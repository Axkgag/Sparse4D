#!/usr/bin/env python3
"""Debug utility to verify dataset and dataloader from a config file.

Example:
  python tools/debug_dataset_loader.py \
    projects/configs/sparse4dv3_temporal_r50_aimotive_tlts_1x8_bs6_256x704.py \
    --split train
"""

import argparse
import importlib
import os
from typing import Any

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer

from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build dataset+dataloader from config and print one batch"
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split in cfg.data",
    )
    parser.add_argument(
        "--samples-per-gpu",
        type=int,
        default=None,
        help="Override samples_per_gpu from config",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help="Override workers_per_gpu from config",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config settings, format: key=value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataloader sampler",
    )
    return parser.parse_args()


def import_plugin_modules(cfg: Config, config_path: str) -> None:
    if not hasattr(cfg, "plugin") or not cfg.plugin:
        return

    if hasattr(cfg, "plugin_dir"):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)

    module_parts = [p for p in module_dir.split("/") if p]
    if not module_parts:
        return

    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path = module_path + "." + part

    print(f"Import plugin module: {module_path}")
    importlib.import_module(module_path)


def summarize(obj: Any, indent: int = 0, key_name: str = "") -> None:
    prefix = " " * indent
    header = f"{prefix}{key_name}: " if key_name else prefix

    if isinstance(obj, DataContainer):
        print(
            f"{header}DataContainer(stack={obj.stack}, cpu_only={obj.cpu_only}, "
            f"pad_dims={obj.pad_dims})"
        )
        summarize(obj.data, indent + 2, key_name="data")
        return

    if isinstance(obj, torch.Tensor):
        print(
            f"{header}Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, "
            f"device={obj.device})"
        )
        return

    if isinstance(obj, np.ndarray):
        print(f"{header}ndarray(shape={obj.shape}, dtype={obj.dtype})")
        return

    if isinstance(obj, dict):
        print(f"{header}dict(keys={list(obj.keys())})")
        for k, v in obj.items():
            summarize(v, indent + 2, key_name=str(k))
        return

    if isinstance(obj, (list, tuple)):
        print(f"{header}{type(obj).__name__}(len={len(obj)})")
        max_show = min(len(obj), 3)
        for i in range(max_show):
            summarize(obj[i], indent + 2, key_name=f"[{i}]")
        if len(obj) > max_show:
            print(f"{' ' * (indent + 2)}... ({len(obj) - max_show} more)")
        return

    if isinstance(obj, str):
        text = obj if len(obj) < 120 else obj[:117] + "..."
        print(f"{header}str({text})")
        return

    print(f"{header}{type(obj).__name__}({obj})")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin_modules(cfg, args.config)

    split_cfg = cfg.data[args.split]

    if args.samples_per_gpu is None:
        samples_per_gpu = cfg.data.get("samples_per_gpu", 1)
    else:
        samples_per_gpu = args.samples_per_gpu

    if args.workers_per_gpu is None:
        workers_per_gpu = cfg.data.get("workers_per_gpu", 1)
    else:
        workers_per_gpu = args.workers_per_gpu

    if args.split in ["val", "test"] and isinstance(split_cfg, dict):
        split_cfg = split_cfg.copy()
        split_cfg["test_mode"] = True

    print(f"Building dataset for split={args.split} ...")
    dataset = build_dataset(split_cfg)
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Dataset length: {len(dataset)}")
    if hasattr(dataset, "CLASSES"):
        print(f"CLASSES ({len(dataset.CLASSES)}): {list(dataset.CLASSES)}")

    runner_type = "EpochBasedRunner"
    if cfg.get("runner") is not None:
        runner_type = cfg.runner.get("type", runner_type)

    print("Building dataloader ...")
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=(args.split == "train"),
        seed=args.seed,
        runner_type=runner_type,
    )

    print("Fetching one batch ...")
    batch = next(iter(dataloader))

    print("\n===== Batch Summary =====")
    summarize(batch)
    print("===== End Batch Summary =====")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
