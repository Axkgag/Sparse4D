#!/usr/bin/env python3
"""Diagnose AiMotive PKL annotations for potential NaN triggers.

Examples:
  python3 tools/diagnose_aimotive_pkl.py \
    --pkl data/aimotive_anno_pkls/aimotive_tl_infos_train.pkl

  python3 tools/diagnose_aimotive_pkl.py \
    --root data/aimotive_anno_pkls --pattern '*tl*train*.pkl' --topk 50
"""

import argparse
import glob
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import mmcv
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose AiMotive PKL quality")
    parser.add_argument("--pkl", type=str, default="", help="Single PKL path")
    parser.add_argument(
        "--root",
        type=str,
        default="data/aimotive_anno_pkls",
        help="Search root for PKL files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*tl*train*.pkl",
        help="Glob pattern used under --root",
    )
    parser.add_argument("--topk", type=int, default=30, help="Top suspicious samples")
    parser.add_argument(
        "--max-box-size",
        type=float,
        default=20.0,
        help="Threshold to flag too-large boxes",
    )
    parser.add_argument(
        "--max-abs-vel",
        type=float,
        default=10.0,
        help="Threshold to flag too-large velocity",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default="",
        help="Optional output path for machine-readable report",
    )
    return parser.parse_args()


def load_infos(pkl_path: str) -> List[Dict]:
    data = mmcv.load(pkl_path)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported PKL format: {pkl_path}")


def parse_sequence(sample_idx: str, sequence_id: str) -> str:
    if sequence_id:
        return sequence_id
    if sample_idx and "_" in sample_idx:
        return sample_idx.rsplit("_", 1)[0]
    return "unknown"


def safe_array(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def diagnose_one(
    pkl_path: str,
    topk: int,
    max_box_size: float,
    max_abs_vel: float,
) -> Dict:
    infos = load_infos(pkl_path)

    counters = Counter()
    reason_counter = Counter()
    sample_reasons: Dict[str, List[str]] = defaultdict(list)
    seq_ts_map: Dict[str, List[float]] = defaultdict(list)
    global_instid_map: Dict[int, int] = Counter()
    class_counter = Counter()

    matrix_issues = Counter()
    io_issues = Counter()

    box_sizes = []
    box_vels = []

    for idx, info in enumerate(infos):
        counters["num_infos"] += 1

        sample_idx = str(info.get("sample_idx", f"idx_{idx}"))
        sequence_id = str(info.get("sequence_id", ""))
        seq_key = parse_sequence(sample_idx, sequence_id)

        ts = info.get("timestamp", np.nan)
        try:
            ts = float(ts)
        except Exception:
            ts = np.nan
        seq_ts_map[seq_key].append(ts)

        gt_boxes = safe_array(info.get("gt_bboxes_3d", np.zeros((0, 9), dtype=np.float32)), np.float32)
        gt_labels = safe_array(info.get("gt_labels_3d", np.zeros((0,), dtype=np.int64)), np.int64)
        inst_ids = info.get("instance_inds", None)
        if inst_ids is not None:
            inst_ids = safe_array(inst_ids, np.int64)

        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes[None, :]
        if gt_boxes.ndim != 2:
            reason = "bad_box_dim"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1
            counters["num_bad_samples"] += 1
            continue

        n = gt_boxes.shape[0]
        counters["num_boxes"] += int(n)

        for label in gt_labels.tolist():
            class_counter[int(label)] += 1

        for mat in info.get("lidar2img", []):
            arr = np.asarray(mat)
            if arr.shape != (4, 4):
                matrix_issues["lidar2img_shape"] += 1
            if not np.isfinite(arr).all():
                matrix_issues["lidar2img_nonfinite"] += 1

        for mat in info.get("cam_intrinsic", []):
            arr = np.asarray(mat)
            if arr.shape != (4, 4):
                matrix_issues["cam_intrinsic_shape"] += 1
            if not np.isfinite(arr).all():
                matrix_issues["cam_intrinsic_nonfinite"] += 1

        lidar2global = np.asarray(info.get("lidar2global", np.eye(4, dtype=np.float32)))
        if lidar2global.shape != (4, 4):
            matrix_issues["lidar2global_shape"] += 1
        if not np.isfinite(lidar2global).all():
            matrix_issues["lidar2global_nonfinite"] += 1

        missing_img = False
        for img_path in info.get("img_filename", []):
            if not os.path.exists(str(img_path)):
                missing_img = True
                break
        if missing_img:
            io_issues["missing_img_path"] += 1

        if n == 0:
            counters["num_empty_infos"] += 1
            continue

        if gt_boxes.shape[1] < 9:
            reason = "box_dims_lt_9"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1

        if gt_labels.shape[0] != n:
            reason = "label_box_len_mismatch"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1

        if inst_ids is not None and inst_ids.shape[0] != n:
            reason = "instid_box_len_mismatch"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1

        if not np.isfinite(gt_boxes).all():
            reason = "nonfinite_boxes"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1

        if not np.isfinite(gt_labels).all():
            reason = "nonfinite_labels"
            sample_reasons[sample_idx].append(reason)
            reason_counter[reason] += 1

        if gt_boxes.shape[1] >= 6:
            sz = gt_boxes[:, 3:6]
            box_sizes.append(sz)
            bad_nonpos = np.any(sz <= 0, axis=1)
            bad_large = np.any(np.abs(sz) > max_box_size, axis=1)
            if np.any(bad_nonpos):
                reason = "box_size_nonpositive"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += int(np.sum(bad_nonpos))
            if np.any(bad_large):
                reason = "box_size_too_large"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += int(np.sum(bad_large))

        if gt_boxes.shape[1] >= 9:
            vel = gt_boxes[:, 7:9]
            box_vels.append(vel)
            if not np.isfinite(vel).all():
                reason = "nonfinite_velocity"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += 1
            bad_vel = np.any(np.abs(vel) > max_abs_vel, axis=1)
            if np.any(bad_vel):
                reason = "velocity_too_large"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += int(np.sum(bad_vel))

        if gt_boxes.shape[1] >= 7:
            yaw = gt_boxes[:, 6]
            if not np.isfinite(yaw).all():
                reason = "nonfinite_yaw"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += 1

        if inst_ids is not None:
            # In-frame duplicate ids are suspicious for association.
            uniq = np.unique(inst_ids)
            if uniq.shape[0] != inst_ids.shape[0]:
                reason = "duplicate_instid_in_frame"
                sample_reasons[sample_idx].append(reason)
                reason_counter[reason] += 1
            for iid in uniq.tolist():
                global_instid_map[int(iid)] += 1

    # Sequence-level timestamp sanity.
    seq_ts_issues = []
    for seq, ts_list in seq_ts_map.items():
        arr = np.asarray(ts_list, dtype=np.float64)
        if not np.isfinite(arr).all():
            seq_ts_issues.append((seq, "nonfinite_timestamp"))
            reason_counter["nonfinite_timestamp"] += 1
            continue
        if arr.size > 1:
            diffs = np.diff(arr)
            if np.any(diffs < 0):
                seq_ts_issues.append((seq, "timestamp_not_monotonic"))
                reason_counter["timestamp_not_monotonic"] += 1
            if np.any(diffs == 0):
                seq_ts_issues.append((seq, "timestamp_repeated"))
                reason_counter["timestamp_repeated"] += 1

    # Cross-frame instance id reuse can be valid for tracking datasets,
    # but for AiMotive TL/TS without stable IDs it is a useful warning.
    reused_inst_ids = sum(1 for _, c in global_instid_map.items() if c > 20)

    all_dt = []
    for _, ts_list in seq_ts_map.items():
        arr = np.asarray(ts_list, dtype=np.float64)
        if arr.size > 1 and np.isfinite(arr).all():
            arr = np.sort(arr)
            all_dt.append(np.diff(arr))
    if all_dt:
        all_dt = np.concatenate(all_dt)
        dt_stats = {
            "min": float(np.min(all_dt)),
            "p50": float(np.percentile(all_dt, 50)),
            "p99": float(np.percentile(all_dt, 99)),
            "max": float(np.max(all_dt)),
            "num_eq_0": int(np.sum(all_dt == 0)),
            "num_lt_0": int(np.sum(all_dt < 0)),
        }
    else:
        dt_stats = None

    suspicious = []
    for sample_idx, reasons in sample_reasons.items():
        uniq = sorted(set(reasons))
        suspicious.append((sample_idx, len(uniq), uniq))
    suspicious.sort(key=lambda x: (-x[1], x[0]))

    if box_sizes:
        all_sizes = np.concatenate(box_sizes, axis=0)
        size_stats = {
            "min": all_sizes.min(axis=0).tolist(),
            "p99": np.percentile(all_sizes, 99, axis=0).tolist(),
            "max": all_sizes.max(axis=0).tolist(),
        }
    else:
        size_stats = None

    if box_vels:
        all_vels = np.concatenate(box_vels, axis=0)
        vel_stats = {
            "min": all_vels.min(axis=0).tolist(),
            "p99": np.percentile(all_vels, 99, axis=0).tolist(),
            "max": all_vels.max(axis=0).tolist(),
        }
    else:
        vel_stats = None

    report = {
        "pkl": pkl_path,
        "counters": dict(counters),
        "reason_counter": dict(reason_counter),
        "class_counter": dict(class_counter),
        "size_stats": size_stats,
        "velocity_stats": vel_stats,
        "matrix_issues": dict(matrix_issues),
        "io_issues": dict(io_issues),
        "dt_stats": dt_stats,
        "seq_timestamp_issues": seq_ts_issues,
        "num_reused_inst_ids_gt20_frames": reused_inst_ids,
        "top_suspicious_samples": [
            {"sample_idx": s, "num_reasons": n, "reasons": r}
            for s, n, r in suspicious[:topk]
        ],
    }
    return report


def format_report(report: Dict) -> str:
    lines = []
    lines.append(f"\n=== {os.path.basename(report['pkl'])} ===")
    c = report["counters"]
    lines.append(
        f"infos={c.get('num_infos', 0)} boxes={c.get('num_boxes', 0)} "
        f"empty_infos={c.get('num_empty_infos', 0)}"
    )

    if report["size_stats"] is not None:
        s = report["size_stats"]
        lines.append(f"size min={s['min']} p99={s['p99']} max={s['max']}")
    if report["velocity_stats"] is not None:
        v = report["velocity_stats"]
        lines.append(f"vel min={v['min']} p99={v['p99']} max={v['max']}")

    rc = report["reason_counter"]
    if rc:
        lines.append("top reasons:")
        for k, v in sorted(rc.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  - {k}: {v}")

    if report["seq_timestamp_issues"]:
        lines.append(
            f"seq timestamp issues: {len(report['seq_timestamp_issues'])} sequences"
        )

    if report.get("dt_stats") is not None:
        dt = report["dt_stats"]
        lines.append(
            "dt min/p50/p99/max="
            f"{dt['min']}/{dt['p50']}/{dt['p99']}/{dt['max']} "
            f"eq0={dt['num_eq_0']} lt0={dt['num_lt_0']}"
        )

    if report.get("matrix_issues"):
        lines.append("matrix issues:")
        for k, v in sorted(report["matrix_issues"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {k}: {v}")

    if report.get("io_issues"):
        lines.append("io issues:")
        for k, v in sorted(report["io_issues"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {k}: {v}")

    if report.get("class_counter"):
        cls_pairs = sorted(report["class_counter"].items(), key=lambda x: -x[1])
        preview = cls_pairs[:10]
        lines.append(f"class counter (top): {preview}")

    lines.append(
        f"reused inst ids (>20 frames): {report['num_reused_inst_ids_gt20_frames']}"
    )

    top = report["top_suspicious_samples"]
    if top:
        lines.append("top suspicious samples:")
        for item in top[:10]:
            lines.append(
                f"  - {item['sample_idx']}: {item['num_reasons']} reasons -> {item['reasons']}"
            )
    else:
        lines.append("top suspicious samples: none")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if args.pkl:
        pkl_files = [args.pkl]
    else:
        pkl_files = sorted(glob.glob(os.path.join(args.root, args.pattern)))

    if not pkl_files:
        print("No PKL files found.")
        return

    reports = []
    for pkl in pkl_files:
        reports.append(
            diagnose_one(
                pkl_path=pkl,
                topk=args.topk,
                max_box_size=args.max_box_size,
                max_abs_vel=args.max_abs_vel,
            )
        )

    for r in reports:
        print(format_report(r))

    if args.dump_json:
        out = {"reports": reports}
        mmcv.dump(out, args.dump_json)
        print(f"\nSaved report json to: {args.dump_json}")


if __name__ == "__main__":
    main()
