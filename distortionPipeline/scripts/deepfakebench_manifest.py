from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.manifest_adapter import (  # noqa: E402
    augmented_manifest_to_clean_subset_datasets,
    augmented_manifest_to_deepfakebench_datasets,
    deepfakebench_dataset_json_to_jsonl,
    pipeline_jsonl_to_deepfakebench_dataset,
    write_dataset_index_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepfakeBench dataset JSON to/from distortion manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_jsonl = subparsers.add_parser("to-jsonl", help="Convert DeepfakeBench dataset JSON to pipeline JSONL.")
    to_jsonl.add_argument("--input", required=True, type=Path)
    to_jsonl.add_argument("--output", required=True, type=Path)
    to_jsonl.add_argument("--dataset_name", default=None)

    to_json = subparsers.add_parser("to-json", help="Convert pipeline JSONL back to DeepfakeBench dataset JSON.")
    to_json.add_argument("--input", required=True, type=Path)
    to_json.add_argument("--output", required=True, type=Path)
    to_json.add_argument("--dataset_name", default=None)

    to_distorted = subparsers.add_parser(
        "to-distorted-json",
        help="Convert an augmented distortion manifest into one DeepfakeBench dataset JSON per treatment.",
    )
    to_distorted.add_argument("--input", required=True, type=Path)
    to_distorted.add_argument("--output_dir", required=True, type=Path)
    to_distorted.add_argument("--index_csv", default=None, type=Path)

    to_clean_subset = subparsers.add_parser(
        "to-clean-subset-json",
        help="Convert an augmented distortion manifest into one clean-subset DeepfakeBench dataset JSON per treatment.",
    )
    to_clean_subset.add_argument("--input", required=True, type=Path)
    to_clean_subset.add_argument("--output_dir", required=True, type=Path)
    to_clean_subset.add_argument("--index_csv", default=None, type=Path)

    args = parser.parse_args()

    if args.command == "to-jsonl":
        deepfakebench_dataset_json_to_jsonl(args.input, args.output, dataset_name=args.dataset_name)
        print(f"Wrote JSONL manifest to {args.output}")
        return

    if args.command == "to-distorted-json":
        rows = augmented_manifest_to_deepfakebench_datasets(args.input, args.output_dir)
        if args.index_csv is not None:
            write_dataset_index_csv(args.index_csv, rows)
        print(f"Wrote {len(rows)} DeepfakeBench dataset JSON file(s) into {args.output_dir}")
        return

    if args.command == "to-clean-subset-json":
        rows = augmented_manifest_to_clean_subset_datasets(args.input, args.output_dir)
        if args.index_csv is not None:
            write_dataset_index_csv(args.index_csv, rows)
        print(f"Wrote {len(rows)} clean-subset DeepfakeBench dataset JSON file(s) into {args.output_dir}")
        return

    pipeline_jsonl_to_deepfakebench_dataset(args.input, args.output, dataset_name=args.dataset_name)
    print(f"Wrote DeepfakeBench dataset JSON to {args.output}")


if __name__ == "__main__":
    main()
