import argparse
import importlib
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed, single_gpu_test
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.utils import draw_lidar_bbox3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on val/test split and save visualizations"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="which split in cfg.data to run inference on",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/inference_vis",
        help="directory to save inference outputs and visualizations",
    )
    parser.add_argument(
        "--result-pkl",
        default=None,
        help="result pkl path; default is <out-dir>/results.pkl",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="score threshold for fallback visualizer",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="max number of samples to visualize, 0 means all",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="fuse conv and bn for a bit faster inference",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="set deterministic option for CUDNN backend",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override settings in config, key=value format",
    )
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            module_dir = osp.dirname(cfg.plugin_dir)
        else:
            module_dir = osp.dirname(config_path)
        module_path = module_dir.replace("/", ".")
        importlib.import_module(module_path)


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_img_path(data_root, path):
    if osp.isabs(path):
        return path
    return osp.join(data_root, path)


def _extract_prediction(result):
    pred = result.get("img_bbox", result)
    if not isinstance(pred, dict):
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = pred.get("boxes_3d", None)
    scores = pred.get("scores_3d", None)
    if boxes is None or scores is None:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    boxes_np = _to_numpy(boxes)
    scores_np = _to_numpy(scores)

    if boxes_np is None or scores_np is None or boxes_np.size == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes_np = boxes_np.astype(np.float32)
    scores_np = scores_np.astype(np.float32)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np[None, :]
    return boxes_np, scores_np


def fallback_visualize(dataset, outputs, save_dir, score_thr=0.25, max_samples=0):
    vis_dir = osp.join(save_dir, "visual")
    mmcv.mkdir_or_exist(vis_dir)

    total = len(outputs)
    if max_samples > 0:
        total = min(total, max_samples)

    for idx in range(total):
        info = dataset.data_infos[idx]
        img_files = info.get("img_filename", [])
        lidar2img = info.get("lidar2img", [])

        if len(img_files) == 0 or len(lidar2img) == 0:
            continue

        imgs = []
        for p in img_files:
            img_path = _resolve_img_path(getattr(dataset, "data_root", ""), p)
            if not osp.exists(img_path):
                imgs = []
                break
            imgs.append(mmcv.imread(img_path))
        if len(imgs) == 0:
            continue

        boxes, scores = _extract_prediction(outputs[idx])
        if scores.shape[0] > 0:
            keep = scores >= float(score_thr)
            boxes = boxes[keep]

        vis = draw_lidar_bbox3d(boxes, imgs, lidar2img, color=(255, 120, 0))
        sample_idx = str(info.get("sample_idx", f"sample_{idx:06d}"))
        out_path = osp.join(vis_dir, f"{sample_idx}.jpg")
        mmcv.imwrite(vis, out_path)

    print(f"Fallback visualization saved to: {osp.abspath(vis_dir)}")


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    result_pkl = (
        args.result_pkl
        if args.result_pkl is not None
        else osp.join(args.out_dir, "results.pkl")
    )

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugins(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    set_random_seed(args.seed, deterministic=args.deterministic)

    data_cfg = cfg.data[args.split]
    if isinstance(data_cfg, dict):
        data_cfg.test_mode = True
    samples_per_gpu = data_cfg.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)

    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 2),
        dist=False,
        shuffle=False,
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    elif hasattr(dataset, "CLASSES"):
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, show=False, out_dir=None)

    mmcv.dump(outputs, result_pkl)
    print(f"Inference results saved to: {osp.abspath(result_pkl)}")

    can_use_dataset_show = hasattr(dataset, "show") and callable(dataset.show)
    if can_use_dataset_show:
        vis_pipeline = cfg.get("evaluation", {}).get("pipeline", None)
        dataset.show(outputs, save_dir=args.out_dir, show=False, pipeline=vis_pipeline)
        print(
            "Visualization saved by dataset.show to: "
            f"{osp.abspath(osp.join(args.out_dir, 'visual'))}"
        )
    else:
        fallback_visualize(
            dataset,
            outputs,
            save_dir=args.out_dir,
            score_thr=args.score_thr,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()
