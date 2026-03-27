# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["Sparse4D"]


def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


def _extract_sample_indices(data, max_items=8):
    img_metas = data.get("img_metas", [])
    if hasattr(img_metas, "data"):
        img_metas = img_metas.data
    if isinstance(img_metas, tuple):
        img_metas = list(img_metas)
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
        img_metas = img_metas[0]

    sample_indices = []
    if isinstance(img_metas, list):
        for meta in img_metas:
            if isinstance(meta, dict):
                sid = meta.get("sample_idx", None)
                if sid is not None:
                    sample_indices.append(str(sid))
    return sample_indices[:max_items]


def _check_finite(name, obj, data):
    for tensor in _iter_tensors(obj):
        if tensor.numel() == 0:
            continue
        if not torch.isfinite(tensor).all():
            det = tensor.detach()
            bad_count = int((~torch.isfinite(det)).sum().item())
            total = int(det.numel())
            sample_indices = _extract_sample_indices(data)
            msg = (
                f"Non-finite detected in {name}: bad={bad_count}/{total}, "
                f"shape={tuple(det.shape)}, dtype={det.dtype}, sample_idx={sample_indices}"
            )
            raise FloatingPointError(msg)


@DETECTORS.register_module()
class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None, debug_finite=False):
        if debug_finite and isinstance(metas, dict):
            _check_finite("input:img", img, metas)
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if debug_finite and isinstance(metas, dict):
            _check_finite("feature_maps:backbone_out", feature_maps, metas)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
            if debug_finite and isinstance(metas, dict):
                _check_finite("feature_maps:neck_out", feature_maps, metas)
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if debug_finite and isinstance(metas, dict):
            _check_finite("feature_maps:reshaped", feature_maps, metas)
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(
            img, True, data, debug_finite=True
        )
        _check_finite("feature_maps", feature_maps, data)
        if depths is not None:
            _check_finite("depths", depths, data)

        model_outs = self.head(feature_maps, data)
        _check_finite("model_outs", model_outs, data)

        output = self.head.loss(model_outs, data)
        for key, value in output.items():
            _check_finite(f"loss:{key}", value, data)

        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
            _check_finite("loss:loss_dense_depth", output["loss_dense_depth"], data)
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
