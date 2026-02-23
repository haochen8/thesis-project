#!/usr/bin/env python3
"""
Create a deterministic subset dataset manifest from an existing DeepfakeBench JSON.

Example:
  python preprocessing/dataset_json/make_subset_manifest.py \
    --source-file preprocessing/dataset_json/NVIDIA-dataset.json \
    --source-name NVIDIA-dataset \
    --output-file preprocessing/dataset_json/NVIDIA-dataset-mini.json \
    --output-name NVIDIA-dataset-mini \
    --train-per-class 100 --val-per-class 100 --test-per-class 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def take_first_n(split_dict: Dict[str, dict], n: int) -> Dict[str, dict]:
    keys = sorted(split_dict.keys())
    if n > len(keys):
        raise ValueError(f"Requested {n} items but split only has {len(keys)}.")
    chosen = keys[:n]
    return {k: split_dict[k] for k in chosen}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic subset manifest JSON.")
    parser.add_argument("--source-file", required=True, help="Path to source JSON.")
    parser.add_argument("--source-name", required=True, help="Top-level dataset key in source JSON.")
    parser.add_argument("--output-file", required=True, help="Output JSON path.")
    parser.add_argument("--output-name", required=True, help="Top-level dataset key in output JSON.")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=500)
    args = parser.parse_args()

    source_file = Path(args.source_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    source = json.loads(source_file.read_text())
    if args.source_name not in source:
        raise ValueError(f"Dataset key '{args.source_name}' not found in {source_file}.")

    source_root = source[args.source_name]
    subset_root = {}
    for label, label_splits in source_root.items():
        train_split = label_splits.get("train", {})
        val_split = label_splits.get("val", {})
        test_split = label_splits.get("test", {})
        subset_root[label] = {
            "train": take_first_n(train_split, args.train_per_class),
            "val": take_first_n(val_split, args.val_per_class),
            "test": take_first_n(test_split, args.test_per_class),
        }

    out = {args.output_name: subset_root}
    output_file.write_text(json.dumps(out, indent=2))

    labels: List[str] = sorted(subset_root.keys())
    print(f"[done] wrote {output_file}")
    print(f"[done] dataset={args.output_name}")
    for label in labels:
        print(
            f"[done] {label}: "
            f"train={len(subset_root[label]['train'])}, "
            f"val={len(subset_root[label]['val'])}, "
            f"test={len(subset_root[label]['test'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
