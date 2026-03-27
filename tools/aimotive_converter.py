# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import defaultdict

import numpy as np

import mmcv

from projects.mmdet3d_plugin.datasets.aimotive_tl_ts_dataset import (
    AiMotiveTLTSDataset,
)


SCENE_NAMES = ["highway", "night", "rainy", "urban"]
DEFAULT_TL_CLASSES = ["red", "red_yellow", "yellow", "green", "unknown"]
DEFAULT_TS_CLASSES = [
    "speed_limit",
    "yield",
    "stop",
    "no_entry",
    "priority",
    "unknown",
]


def _classes_for_object_type(object_type, traffic_sign_classes=None):
    if object_type == "traffic_light":
        return DEFAULT_TL_CLASSES
    if traffic_sign_classes is not None and len(traffic_sign_classes) > 0:
        return traffic_sign_classes
    return DEFAULT_TS_CLASSES


def _split_counts(num_items, val_ratio, test_ratio):
    """Compute per-scene split counts with basic safeguards."""
    if num_items <= 0:
        return 0, 0, 0
    if num_items == 1:
        return 1, 0, 0
    if num_items == 2:
        return 1, 1, 0

    val_num = int(round(num_items * val_ratio))
    test_num = int(round(num_items * test_ratio))
    val_num = max(1, val_num)
    test_num = max(1, test_num)

    if val_num + test_num >= num_items:
        overflow = val_num + test_num - (num_items - 1)
        reduce_val = min(overflow, max(0, val_num - 1))
        val_num -= reduce_val
        overflow -= reduce_val
        reduce_test = min(overflow, max(0, test_num - 1))
        test_num -= reduce_test

    train_num = num_items - val_num - test_num
    if train_num <= 0:
        train_num = 1
        if val_num >= test_num and val_num > 1:
            val_num -= 1
        elif test_num > 1:
            test_num -= 1

    return train_num, val_num, test_num


def _check_scene_coverage(scene, num_items, train_num, val_num, test_num):
    if num_items >= 3:
        if min(train_num, val_num, test_num) <= 0:
            raise RuntimeError(
                f"Scene {scene} has {num_items} sequences but split cannot keep all train/val/test non-empty."
            )
    elif num_items > 0:
        print(
            f"[WARN] Scene {scene} has only {num_items} sequence(s); "
            "cannot guarantee all train/val/test are non-empty."
        )


def _find_scene_sequence_dirs(root_path, scene_names):
    """Collect sequence dirs grouped by scene name.

    Expected structure:
      root_path/highway/<sequence>/sensor/...
      root_path/night/<sequence>/sensor/...
      root_path/rainy/<sequence>/sensor/...
      root_path/urban/<sequence>/sensor/...
    """
    seq_map = {scene: [] for scene in scene_names}
    for scene in scene_names:
        scene_root = os.path.join(root_path, scene)
        if not os.path.isdir(scene_root):
            continue
        for entry in sorted(os.listdir(scene_root)):
            seq_dir = os.path.join(scene_root, entry)
            if not os.path.isdir(seq_dir):
                continue
            camera_dir = os.path.join(seq_dir, "sensor", "camera")
            if os.path.isdir(camera_dir):
                seq_map[scene].append(seq_dir)
    return seq_map


def _build_infos_for_sequences(dataset, seq_dirs):
    infos = []
    stats = {
        "num_infos_before": 0,
        "num_infos_after": 0,
        "num_infos_dropped_empty": 0,
        "num_boxes_before": 0,
        "num_boxes_after": 0,
    }
    for seq_dir in seq_dirs:
        seq_infos = dataset._build_sequence_infos(seq_dir)
        for info in seq_infos:
            stats["num_infos_before"] += 1
            boxes_before = np.asarray(
                info.get("gt_bboxes_3d", np.zeros((0, 9), dtype=np.float32)),
                dtype=np.float32,
            )
            stats["num_boxes_before"] += int(boxes_before.shape[0])

            ann = dataset._sanitize_annotations(
                info.get("gt_bboxes_3d"),
                info.get("gt_labels_3d"),
                info.get("gt_names"),
                info.get("instance_inds"),
                apply_radius_filter=True,
            )
            if info.get("instance_inds", None) is None:
                sample_uid = str(info.get("sample_idx", ""))
                ann["instance_inds"] = dataset._fallback_instance_ids(
                    sample_uid, int(ann["gt_labels_3d"].shape[0])
                )
            info = dict(info)
            info["gt_bboxes_3d"] = ann["gt_bboxes_3d"]
            info["gt_labels_3d"] = ann["gt_labels_3d"]
            info["gt_names"] = ann["gt_names"]
            info["instance_inds"] = ann["instance_inds"]

            boxes_after = info["gt_bboxes_3d"]
            stats["num_boxes_after"] += int(boxes_after.shape[0])
            if boxes_after.shape[0] == 0:
                stats["num_infos_dropped_empty"] += 1
                continue

            stats["num_infos_after"] += 1
            infos.append(info)
    infos = dataset._remap_timestamps(infos)
    return infos, stats


def _dump_infos(info_prefix, split, infos, metadata):
    out_path = f"{info_prefix}_infos_{split}.pkl"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(dict(infos=infos, metadata=metadata), out_path)
    print(f"Saved {len(infos)} infos to {out_path}")


def create_aimotive_infos_with_split(
    root_path,
    info_prefix,
    object_type="traffic_light",
    cam_order=None,
    seed=42,
    val_ratio=0.2,
    test_ratio=0.2,
    scene_names=None,
    traffic_sign_classes=None,
):
    """Create train/val/test pkl infos with scene-balanced random sampling.

    The split is performed at sequence level for each ODD scene independently,
    then concatenated to global train/val/test sets.
    """
    if scene_names is None:
        scene_names = SCENE_NAMES

    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio+test_ratio < 1")

    dataset = AiMotiveTLTSDataset(
        data_root=root_path,
        pipeline=None,
        object_type=object_type,
        classes=_classes_for_object_type(object_type, traffic_sign_classes),
        cam_order=cam_order,
        load_interval=1,
        test_mode=False,
        with_seq_flag=False,
        lazy_init=True,
    )

    # Avoid reusing pre-built infos from __init__; we split by sequence first.
    scene_seq_map = _find_scene_sequence_dirs(root_path, scene_names)
    rng = np.random.RandomState(seed)

    split_seq_map = defaultdict(list)
    scene_stats = {}
    for scene in scene_names:
        seq_dirs = list(scene_seq_map.get(scene, []))
        if len(seq_dirs) == 0:
            scene_stats[scene] = {
                "num_sequences": 0,
                "train": 0,
                "val": 0,
                "test": 0,
            }
            continue

        perm = rng.permutation(len(seq_dirs))
        seq_dirs = [seq_dirs[i] for i in perm]
        train_num, val_num, test_num = _split_counts(
            len(seq_dirs), val_ratio, test_ratio
        )
        _check_scene_coverage(scene, len(seq_dirs), train_num, val_num, test_num)

        train_dirs = seq_dirs[:train_num]
        val_dirs = seq_dirs[train_num : train_num + val_num]
        test_dirs = seq_dirs[train_num + val_num : train_num + val_num + test_num]

        split_seq_map["train"].extend(train_dirs)
        split_seq_map["val"].extend(val_dirs)
        split_seq_map["test"].extend(test_dirs)

        scene_stats[scene] = {
            "num_sequences": len(seq_dirs),
            "train": len(train_dirs),
            "val": len(val_dirs),
            "test": len(test_dirs),
        }

    split_infos = {}
    split_stats = {}
    for split in ["train", "val", "test"]:
        infos, stats = _build_infos_for_sequences(dataset, split_seq_map[split])
        split_infos[split] = infos
        split_stats[split] = stats

    common_metadata = {
        "dataset": "aimotive_tl_ts",
        "object_type": object_type,
        "data_root": root_path,
        "cam_order": dataset.cam_order,
        "classes": list(dataset.CLASSES),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scene_stats": scene_stats,
        "sanitize_thresholds": {
            "gt_filter_radius": float(dataset.gt_filter_radius)
            if dataset.gt_filter_radius is not None
            else None,
            "min_box_size": float(dataset.min_box_size),
            "max_box_size": float(dataset.max_box_size),
            "max_abs_velocity": float(dataset.max_abs_velocity),
        },
    }

    for split in ["train", "val", "test"]:
        metadata = dict(common_metadata)
        metadata["split"] = split
        metadata["num_infos"] = len(split_infos[split])
        metadata["sanitize_stats"] = split_stats[split]
        _dump_infos(info_prefix, split, split_infos[split], metadata)

    print("Scene-level sequence split stats:")
    for scene in scene_names:
        s = scene_stats[scene]
        print(
            f"  {scene}: total={s['num_sequences']} "
            f"train={s['train']} val={s['val']} test={s['test']}"
        )
    print("Sanitization stats:")
    for split in ["train", "val", "test"]:
        s = split_stats[split]
        print(
            f"  {split}: infos_before={s['num_infos_before']} "
            f"infos_after={s['num_infos_after']} "
            f"dropped_empty={s['num_infos_dropped_empty']} "
            f"boxes_before={s['num_boxes_before']} "
            f"boxes_after={s['num_boxes_after']}"
        )


def create_infos_for_both_tasks(
    root_path,
    info_prefix,
    cam_order=None,
    seed=42,
    val_ratio=0.2,
    test_ratio=0.2,
    scene_names=None,
    independent_sampling=True,
    traffic_sign_classes=None,
):
    """Create two independent pkl sets for TL and TS.

    Output files:
      {info_prefix}_tl_infos_{split}.pkl
      {info_prefix}_ts_infos_{split}.pkl
    """
    task_cfgs = [
        ("traffic_light", f"{info_prefix}_tl", seed),
        (
            "traffic_sign",
            f"{info_prefix}_ts",
            seed + 1 if independent_sampling else seed,
        ),
    ]

    for object_type, task_prefix, task_seed in task_cfgs:
        print(
            "\n============================================================"
        )
        print(
            f"Generating {object_type} infos with seed={task_seed}, "
            f"prefix={task_prefix}"
        )
        create_aimotive_infos_with_split(
            root_path=root_path,
            info_prefix=task_prefix,
            object_type=object_type,
            cam_order=cam_order,
            seed=task_seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            scene_names=scene_names,
            traffic_sign_classes=traffic_sign_classes,
        )


def main():
    parser = argparse.ArgumentParser(description="AiMotive TL/TS converter")
    parser.add_argument(
        "--root_path",
        type=str,
        default="data/aimotive_tl_ts",
        help="ODD root path containing highway/night/rainy/urban",
    )
    parser.add_argument(
        "--info_prefix",
        type=str,
        default="data/aimotive_anno_pkls/aimotive_tl",
        help="Output file prefix, final file is {prefix}_infos_{split}.pkl",
    )
    parser.add_argument(
        "--object_type",
        type=str,
        default="traffic_light",
        choices=["traffic_light", "traffic_sign", "both"],
        help="Target object type",
    )
    parser.add_argument(
        "--cam_order",
        type=str,
        default="",
        help="Comma-separated camera names, leave empty for default order",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio per scene (by sequences)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test split ratio per scene (by sequences)",
    )
    parser.add_argument(
        "--scene_names",
        type=str,
        default="highway,night,rainy,urban",
        help="Comma-separated scene folders under root_path",
    )
    parser.add_argument(
        "--shared_sampling",
        action="store_true",
        help=(
            "When object_type=both, use shared sampling for TL and TS "
            "(same seed for both tasks)."
        ),
    )
    parser.add_argument(
        "--traffic_sign_classes",
        type=str,
        default="speed_limit,yield,stop,no_entry,priority,unknown",
        help=(
            "Comma-separated classes used when object_type is traffic_sign. "
            "Must match training config class_names for traffic_sign."
        ),
    )
    args = parser.parse_args()

    cam_order = None
    if args.cam_order.strip():
        cam_order = [x.strip() for x in args.cam_order.split(",") if x.strip()]

    scene_names = [
        x.strip() for x in args.scene_names.split(",") if x.strip()
    ]
    traffic_sign_classes = [
        x.strip() for x in args.traffic_sign_classes.split(",") if x.strip()
    ]

    if args.object_type == "both":
        create_infos_for_both_tasks(
            root_path=args.root_path,
            info_prefix=args.info_prefix,
            cam_order=cam_order,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            scene_names=scene_names,
            independent_sampling=(not args.shared_sampling),
            traffic_sign_classes=traffic_sign_classes,
        )
    else:
        create_aimotive_infos_with_split(
            root_path=args.root_path,
            info_prefix=args.info_prefix,
            object_type=args.object_type,
            cam_order=cam_order,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            scene_names=scene_names,
            traffic_sign_classes=traffic_sign_classes,
        )


if __name__ == "__main__":
    main()
