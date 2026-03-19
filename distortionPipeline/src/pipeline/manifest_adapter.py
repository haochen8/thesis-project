from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from src.pipeline.manifest_io import read_jsonl, write_jsonl

PIPELINE_CORE_FIELDS = {
    "image_id",
    "label",
    "path",
    "src_path",
    "image_base64",
}

PIPELINE_JOB_FIELDS = PIPELINE_CORE_FIELDS | {
    "job_id",
    "recipe_id",
    "recipe_instance_id",
    "recipe_label",
    "steps",
    "variant",
    "global_seed",
    "source_metadata",
    "source_dataset_name",
    "source_label",
    "source_split",
    "source_sample_id",
    "source_frame_index",
    "source_frame_count",
    "source_frame_path",
}

PIPELINE_RUNTIME_FIELDS = {
    "seed",
    "cache_key",
    "cache_hit",
    "distorted_path",
}


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def _ensure_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _resolve_dataset_root(
    dataset_json: Mapping[str, Any],
    dataset_name: str | None,
) -> tuple[str, Mapping[str, Any]]:
    if dataset_name is None:
        dataset_names = sorted(dataset_json.keys())
        if len(dataset_names) != 1:
            raise ValueError("dataset_name is required when the JSON contains multiple datasets")
        dataset_name = dataset_names[0]
    if dataset_name not in dataset_json:
        raise KeyError(f"dataset key '{dataset_name}' not found")
    dataset_root = _ensure_mapping(dataset_json[dataset_name], f"dataset '{dataset_name}'")
    return dataset_name, dataset_root


def extract_source_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    raw_metadata = record.get("source_metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update(deepcopy(dict(raw_metadata)))

    excluded_fields = PIPELINE_JOB_FIELDS | PIPELINE_RUNTIME_FIELDS
    for key, value in record.items():
        if key not in excluded_fields:
            metadata[key] = deepcopy(value)
    return metadata


def deepfakebench_dataset_to_pipeline_records(
    dataset_json: Mapping[str, Any],
    dataset_name: str | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
) -> Iterator[Dict[str, Any]]:
    """Flatten a DeepfakeBench dataset JSON into pipeline-ready JSONL records."""

    dataset_name, dataset_root = _resolve_dataset_root(dataset_json, dataset_name)
    for label in sorted(dataset_root.keys()):
        label_root = _ensure_mapping(dataset_root[label], f"label '{label}'")
        for split in splits:
            split_root = label_root.get(split)
            if split_root is None:
                continue
            split_root = _ensure_mapping(split_root, f"split '{split}' for label '{label}'")
            for sample_id in sorted(split_root.keys()):
                sample = _ensure_mapping(split_root[sample_id], f"sample '{sample_id}'")
                frames = sample.get("frames")
                if not isinstance(frames, list) or not frames:
                    raise ValueError(
                        f"sample '{sample_id}' in label '{label}' split '{split}' must include a non-empty frames list"
                    )

                sample_metadata = deepcopy(dict(sample))
                sample_metadata.pop("frames", None)
                for frame_index, frame_path in enumerate(frames):
                    record = {
                        "image_id": sample_id if len(frames) == 1 else f"{sample_id}__f{frame_index:04d}",
                        "label": label,
                        "path": str(frame_path),
                        "src_path": str(frame_path),
                        "source_dataset_name": dataset_name,
                        "source_label": label,
                        "source_split": split,
                        "source_sample_id": sample_id,
                        "source_frame_index": frame_index,
                        "source_frame_count": len(frames),
                        "source_frame_path": str(frame_path),
                        "source_metadata": deepcopy(sample_metadata),
                    }
                    yield record


def deepfakebench_dataset_json_to_jsonl(
    input_json: str | Path,
    output_jsonl: str | Path,
    dataset_name: str | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    input_path = Path(input_json)
    dataset_json = _ensure_mapping(
        json.loads(input_path.read_text(encoding="utf-8")),
        f"DeepfakeBench dataset JSON at {input_path}",
    )
    records = list(deepfakebench_dataset_to_pipeline_records(dataset_json, dataset_name=dataset_name, splits=splits))
    write_jsonl(output_jsonl, records)


def pipeline_records_to_deepfakebench_dataset(
    records: Iterable[Mapping[str, Any]],
    dataset_name: str | None = None,
) -> Dict[str, Any]:
    """Rebuild a DeepfakeBench dataset JSON from pipeline records."""

    grouped: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    dataset_names = set()

    for record in records:
        source_dataset_name = record.get("source_dataset_name")
        if source_dataset_name is not None:
            dataset_names.add(str(source_dataset_name))

        source_label = str(record.get("source_label") or record.get("label") or "unknown")
        source_split = str(record.get("source_split") or "train")
        source_sample_id = str(record.get("source_sample_id") or record.get("image_id") or "sample")
        frame_index = int(record.get("source_frame_index", 0))
        frame_path = record.get("source_frame_path") or record.get("src_path") or record.get("path")
        if frame_path is None:
            raise ValueError("each record must include source_frame_path, src_path, or path")

        key = (source_dataset_name or dataset_name or "", source_label, source_split, source_sample_id)
        bucket = grouped.setdefault(key, {"metadata": None, "frames": {}})
        metadata = extract_source_metadata(record)
        if bucket["metadata"] is None:
            bucket["metadata"] = metadata
        else:
            for meta_key, meta_value in metadata.items():
                bucket["metadata"].setdefault(meta_key, meta_value)
        bucket["frames"][frame_index] = str(frame_path)

    if dataset_name is None:
        if len(dataset_names) > 1:
            raise ValueError("dataset_name is required when pipeline records refer to multiple datasets")
        dataset_name = next(iter(dataset_names), None)
        if dataset_name is None:
            raise ValueError("dataset_name could not be inferred from the pipeline records")

    dataset_root: Dict[str, Any] = {}
    for (_, label, split, sample_id), payload in sorted(grouped.items()):
        label_root = dataset_root.setdefault(label, {})
        split_root = label_root.setdefault(split, {})
        frames = [path for _, path in sorted(payload["frames"].items())]
        sample = deepcopy(payload["metadata"] or {})
        sample["label"] = label
        sample["frames"] = frames
        split_root[sample_id] = sample

    return {dataset_name: dataset_root}


def pipeline_jsonl_to_deepfakebench_dataset(
    input_jsonl: str | Path,
    output_json: str | Path,
    dataset_name: str | None = None,
) -> None:
    records = list(read_jsonl(input_jsonl))
    dataset_json = pipeline_records_to_deepfakebench_dataset(records, dataset_name=dataset_name)
    Path(output_json).write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def build_distorted_dataset_name(source_dataset_name: str, recipe_instance_id: str, variant: int) -> str:
    return sanitize_name(f"{source_dataset_name}__{recipe_instance_id}__v{variant}")


def build_clean_subset_dataset_name(source_dataset_name: str, recipe_instance_id: str, variant: int) -> str:
    return sanitize_name(f"{source_dataset_name}__clean_subset__{recipe_instance_id}__v{variant}")


def _prepared_distorted_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    distorted_path = record.get("distorted_path")
    if not distorted_path:
        raise ValueError("augmented distortion record is missing distorted_path")
    prepared = deepcopy(dict(record))
    prepared["path"] = str(distorted_path)
    prepared["src_path"] = str(distorted_path)
    prepared["source_frame_path"] = str(distorted_path)
    return prepared


def _prepared_clean_subset_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    source_frame_path = record.get("source_frame_path") or record.get("src_path") or record.get("path")
    if not source_frame_path:
        raise ValueError("augmented distortion record is missing source_frame_path/src_path/path for clean subset export")
    prepared = deepcopy(dict(record))
    prepared["path"] = str(source_frame_path)
    prepared["src_path"] = str(source_frame_path)
    prepared["source_frame_path"] = str(source_frame_path)
    return prepared


def augmented_manifest_to_deepfakebench_datasets(
    input_jsonl: str | Path,
    output_dir: str | Path,
) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int], list[Dict[str, Any]]] = {}
    for record in read_jsonl(input_jsonl):
        source_dataset_name = record.get("source_dataset_name")
        recipe_instance_id = record.get("recipe_instance_id")
        variant = record.get("variant")
        if source_dataset_name is None or recipe_instance_id is None or variant is None:
            raise ValueError(
                "augmented manifest records must include source_dataset_name, recipe_instance_id, and variant"
            )
        key = (str(source_dataset_name), str(recipe_instance_id), int(variant))
        grouped.setdefault(key, []).append(_prepared_distorted_record(record))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[Dict[str, Any]] = []
    for (source_dataset_name, recipe_instance_id, variant), records in sorted(grouped.items()):
        dataset_name = build_distorted_dataset_name(source_dataset_name, recipe_instance_id, variant)
        output_json = output_dir / f"{dataset_name}.json"
        dataset_json = pipeline_records_to_deepfakebench_dataset(records, dataset_name=dataset_name)
        output_json.write_text(
            json.dumps(dataset_json, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        first = records[0]
        manifest_rows.append(
            {
                "dataset_name": dataset_name,
                "source_dataset_name": source_dataset_name,
                "recipe_id": str(first.get("recipe_id", "")),
                "recipe_instance_id": recipe_instance_id,
                "recipe_label": str(first.get("recipe_label", "")),
                "variant": variant,
                "manifest_path": str(output_json.resolve()),
                "input_jsonl": str(Path(input_jsonl).resolve()),
            }
        )
    return manifest_rows


def augmented_manifest_to_clean_subset_datasets(
    input_jsonl: str | Path,
    output_dir: str | Path,
) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int], list[Dict[str, Any]]] = {}
    for record in read_jsonl(input_jsonl):
        source_dataset_name = record.get("source_dataset_name")
        recipe_instance_id = record.get("recipe_instance_id")
        variant = record.get("variant")
        if source_dataset_name is None or recipe_instance_id is None or variant is None:
            raise ValueError(
                "augmented manifest records must include source_dataset_name, recipe_instance_id, and variant"
            )
        key = (str(source_dataset_name), str(recipe_instance_id), int(variant))
        grouped.setdefault(key, []).append(_prepared_clean_subset_record(record))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[Dict[str, Any]] = []
    for (source_dataset_name, recipe_instance_id, variant), records in sorted(grouped.items()):
        dataset_name = build_clean_subset_dataset_name(source_dataset_name, recipe_instance_id, variant)
        output_json = output_dir / f"{dataset_name}.json"
        dataset_json = pipeline_records_to_deepfakebench_dataset(records, dataset_name=dataset_name)
        output_json.write_text(
            json.dumps(dataset_json, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        first = records[0]
        manifest_rows.append(
            {
                "dataset_name": dataset_name,
                "source_dataset_name": source_dataset_name,
                "recipe_id": str(first.get("recipe_id", "")),
                "recipe_instance_id": recipe_instance_id,
                "recipe_label": str(first.get("recipe_label", "")),
                "variant": variant,
                "manifest_path": str(output_json.resolve()),
                "input_jsonl": str(Path(input_jsonl).resolve()),
            }
        )
    return manifest_rows


def write_dataset_index_csv(output_csv: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "source_dataset_name",
        "recipe_id",
        "recipe_instance_id",
        "recipe_label",
        "variant",
        "manifest_path",
        "input_jsonl",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
