import argparse
import importlib
import json
import os
import os.path as osp

import cv2
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

from projects.mmdet3d_plugin.datasets.utils import (
    box3d_to_corners,
    draw_lidar_bbox3d_on_bev,
    plot_rect3d_on_img,
)


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
    norm_path = osp.normpath(path)
    if osp.exists(norm_path):
        return norm_path

    norm_root = osp.normpath(data_root) if data_root else ""
    if norm_root and (
        norm_path == norm_root
        or norm_path.startswith(norm_root + osp.sep)
    ):
        return norm_path

    if data_root:
        return osp.join(data_root, path)
    return path


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


def _calibration_json_from_img_path(img_path):
    norm = osp.normpath(img_path)
    marker = f"{osp.sep}sensor{osp.sep}camera{osp.sep}"
    pos = norm.find(marker)
    if pos < 0:
        return None, None
    seq_root = norm[:pos]
    cam_name = osp.basename(osp.dirname(norm))
    calib_path = osp.join(seq_root, "sensor", "calibration", "calibration.json")
    return calib_path, cam_name


def _load_camera_calibration(img_path, cache):
    calib_path, cam_name = _calibration_json_from_img_path(img_path)
    if calib_path is None or cam_name is None:
        return None
    key = (calib_path, cam_name)
    if key in cache:
        return cache[key]

    calib = None
    if osp.exists(calib_path):
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cam_cfg = data.get(cam_name, {}) if isinstance(data, dict) else {}
            if isinstance(cam_cfg, dict):
                calib = cam_cfg
        except Exception:
            calib = None
    cache[key] = calib
    return calib


def _camera_meta_for_projection(img_path, cam_intrinsic, lidar2img, calib_cache):
    K4 = np.asarray(cam_intrinsic, dtype=np.float32)
    if K4.shape == (4, 4):
        K = K4[:3, :3]
    elif K4.shape == (3, 3):
        K = K4
    else:
        return None
    if not np.isfinite(K).all():
        return None

    calib = _load_camera_calibration(img_path, calib_cache)
    model = "opencv_pinhole"
    xi = None
    dist = np.zeros((5,), dtype=np.float32)
    focal = [float(K[0, 0]), float(K[1, 1])]
    principal = [float(K[0, 2]), float(K[1, 2])]
    body_to_view = None

    l2i = np.asarray(lidar2img, dtype=np.float32)
    if l2i.shape != (4, 4):
        return None

    if isinstance(calib, dict):
        model = str(calib.get("model", "opencv_pinhole")).lower()
        raw_xi = calib.get("xi", None)
        if isinstance(raw_xi, (int, float)):
            xi = float(raw_xi)
        elif isinstance(raw_xi, (list, tuple)) and len(raw_xi) > 0:
            try:
                xi = float(raw_xi[0])
            except Exception:
                xi = None
        raw_dist = calib.get("distortion_coeffs", None)
        if raw_dist is not None:
            arr = np.asarray(raw_dist, dtype=np.float32).reshape(-1)
            if arr.size >= 4 and np.isfinite(arr).all():
                dist = arr[:5]

        if "focal_length_px" in calib and "principal_point_px" in calib:
            try:
                fx, fy = calib["focal_length_px"][:2]
                cx, cy = calib["principal_point_px"][:2]
                focal = [float(fx), float(fy)]
                principal = [float(cx), float(cy)]
            except Exception:
                pass

        # Follow aiMotive official renderer: compute body_to_view from yaw/pitch/roll + pos.
        if "yaw_pitch_roll_deg" in calib and "pos_meter" in calib:
            try:
                ypr = np.asarray(calib["yaw_pitch_roll_deg"], dtype=np.float64)
                pos = np.asarray(calib["pos_meter"], dtype=np.float64)
                yaw, pitch, roll = np.radians(ypr)
                rt_cam_to_body = _euler_to_matrix(
                    -roll, -pitch, -yaw, order="XYZ"
                )
                rt_cam_to_body[3, :3] = pos[:3]
                a_cam_to_view = np.array(
                    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                    dtype=np.float64,
                )
                body_to_view = _rt_inverse_postmul(rt_cam_to_body) @ a_cam_to_view
            except Exception:
                body_to_view = None

    if body_to_view is None:
        # Fallback from lidar2img decomposition (row-vector convention).
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = K.astype(np.float64)
        try:
            body_to_cam_col = np.linalg.inv(viewpad) @ l2i.astype(np.float64)
            body_to_cam_row = body_to_cam_col.T
            a_cam_to_view = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            body_to_view = body_to_cam_row @ a_cam_to_view
        except np.linalg.LinAlgError:
            return None

    return {
        "model": model if model in ("opencv_pinhole", "mei") else "opencv_pinhole",
        "focal_length_px": np.asarray(focal, dtype=np.float64),
        "principal_point_px": np.asarray(principal, dtype=np.float64),
        "distortion_coeffs": np.asarray(dist, dtype=np.float64),
        "xi": xi,
        "body_to_view": np.asarray(body_to_view, dtype=np.float64),
    }


def _euler_to_matrix(x_rotation, y_rotation, z_rotation, order="XYZ"):
    ax = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x_rotation), np.sin(x_rotation), 0],
            [0, -np.sin(x_rotation), np.cos(x_rotation), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    ay = np.array(
        [
            [np.cos(y_rotation), 0, -np.sin(y_rotation), 0],
            [0, 1, 0, 0],
            [np.sin(y_rotation), 0, np.cos(y_rotation), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    az = np.array(
        [
            [np.cos(z_rotation), np.sin(z_rotation), 0, 0],
            [-np.sin(z_rotation), np.cos(z_rotation), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    if order == "XYZ":
        return ax @ ay @ az
    raise ValueError(f"Unsupported order: {order}")


def _rt_inverse_postmul(rt_mat_postmul):
    r = rt_mat_postmul[:3, :3]
    t = rt_mat_postmul[3, :3]
    mx = np.identity(4, dtype=np.float64)
    mx[:3, :3] = r.T
    mx[3, :3] = -t @ r.T
    return mx


def _pinhole_view_to_image(ray, cam):
    x = ray[0] / np.clip(ray[2], 1e-8, 1e8)
    y = ray[1] / np.clip(ray[2], 1e-8, 1e8)
    d = cam["distortion_coeffs"]
    k1, k2, p1, p2, k3 = d[:5]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    coef = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    qx = x * coef + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    qy = y * coef + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)

    fx, fy = cam["focal_length_px"]
    cx, cy = cam["principal_point_px"]
    qx = qx * fx + cx
    qy = qy * fy + cy

    image_point = np.full((2, ray.shape[1]), -1.0, dtype=np.float64)
    valid = np.isfinite(qx) & np.isfinite(qy)
    image_point[0, valid] = qx[valid]
    image_point[1, valid] = qy[valid]
    return image_point


def _mei_view_to_image(ray, cam):
    x, y, z = ray
    xi = float(cam["xi"] if cam["xi"] is not None else 1.0)
    d = cam["distortion_coeffs"]
    k1, k2, p1, p2, k3 = d[:5]

    norm = np.sqrt(x * x + y * y + z * z)
    norm = np.clip(norm, 1e-8, 1e8)
    x = x / norm
    y = y / norm
    z = z / norm + xi
    z[np.abs(z) <= 1e-5] = 1e-5

    x = x / z
    y = y / z
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    coef = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    qx = x * coef + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    qy = y * coef + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)

    fx, fy = cam["focal_length_px"]
    cx, cy = cam["principal_point_px"]
    qx = qx * fx + cx
    qy = qy * fy + cy

    image_point = np.full((2, ray.shape[1]), -1.0, dtype=np.float64)
    valid = np.isfinite(qx) & np.isfinite(qy)
    image_point[0, valid] = qx[valid]
    image_point[1, valid] = qy[valid]
    return image_point


def _ray_to_image(ray, cam):
    if cam["model"] == "mei":
        return _mei_view_to_image(ray, cam)
    return _pinhole_view_to_image(ray, cam)


def _project_points(points_xyz, cam_meta, w, h):
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    view_pts = (pts_h @ cam_meta["body_to_view"])[:, :3]

    uv = _ray_to_image(view_pts.T, cam_meta).T
    valid = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return uv.astype(np.float32), valid


def _draw_lidar_bbox3d_model_aware(boxes_3d, imgs, cam_metas, color=(255, 120, 0)):
    corners = box3d_to_corners(boxes_3d).astype(np.float32)
    num_bbox = corners.shape[0]

    vis_imgs = []
    flat_corners = corners.reshape(-1, 3)
    for img, cam_meta in zip(imgs, cam_metas):
        cam_img = img.copy()
        h, w = cam_img.shape[:2]
        uv, valid = _project_points(flat_corners, cam_meta, w, h)
        rect = uv.reshape(num_bbox, 8, 2)
        valid_per_box = valid.reshape(num_bbox, 8).any(axis=1)
        rect[~valid_per_box] = -1e6
        cam_img = plot_rect3d_on_img(
            cam_img, num_bbox, rect, color=color, thickness=1
        )
        vis_imgs.append(cam_img)

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        merged = np.concatenate(vis_imgs, axis=1)
    else:
        merged = np.concatenate(
            [
                np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
            ],
            axis=0,
        )
    bev = draw_lidar_bbox3d_on_bev(boxes_3d, merged.shape[0], color=color)
    return np.concatenate([bev, merged], axis=1)


def fallback_visualize(dataset, outputs, save_dir, score_thr=0.25, max_samples=0):
    vis_dir = osp.join(save_dir, "visual")
    mmcv.mkdir_or_exist(vis_dir)

    calib_cache = {}
    saved = 0
    for idx in range(len(outputs)):
        if max_samples > 0 and saved >= max_samples:
            break

        info = dataset.data_infos[idx]
        img_files = info.get("img_filename", [])
        lidar2img = info.get("lidar2img", [])
        cam_intrinsic = info.get("cam_intrinsic", [])

        if len(img_files) == 0 or len(lidar2img) == 0:
            continue

        imgs = []
        cam_metas = []
        for cam_idx, p in enumerate(img_files):
            img_path = _resolve_img_path(getattr(dataset, "data_root", ""), p)
            if not osp.exists(img_path):
                imgs = []
                break
            img = mmcv.imread(img_path)
            l2i = lidar2img[cam_idx] if cam_idx < len(lidar2img) else None
            if l2i is None:
                imgs = []
                break
            k = cam_intrinsic[cam_idx] if cam_idx < len(cam_intrinsic) else None
            if k is None:
                imgs = []
                break
            cam_meta = _camera_meta_for_projection(
                img_path, k, l2i, calib_cache
            )
            if cam_meta is None:
                imgs = []
                break
            imgs.append(img)
            cam_metas.append(cam_meta)
        if len(imgs) == 0:
            continue

        boxes, scores = _extract_prediction(outputs[idx])
        if scores.shape[0] == 0:
            continue
        keep = scores >= float(score_thr)
        if not np.any(keep):
            continue
        boxes = boxes[keep]
        if boxes.shape[0] == 0:
            continue

        vis = _draw_lidar_bbox3d_model_aware(
            boxes, imgs, cam_metas, color=(255, 120, 0)
        )
        sample_idx = str(info.get("sample_idx", f"sample_{idx:06d}"))
        out_path = osp.join(vis_dir, f"{sample_idx}.jpg")
        mmcv.imwrite(vis, out_path)
        saved += 1

    print(
        f"Fallback visualization saved to: {osp.abspath(vis_dir)} "
        f"(saved {saved} samples with detections)"
    )


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
    if osp.exists(result_pkl):
        outputs = mmcv.load(result_pkl)
        print(f"Found existing results, skip inference: {osp.abspath(result_pkl)}")
    else:
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

    if not isinstance(outputs, list):
        raise TypeError("Loaded outputs must be a list.")
    if len(outputs) != len(dataset):
        raise ValueError(
            f"results length ({len(outputs)}) != dataset length ({len(dataset)}). "
            "Please use matching config/split/result-pkl."
        )

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
