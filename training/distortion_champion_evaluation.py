#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


def sanitize_name(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def optional_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def format_num(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_champion_detectors(champions_json: Path) -> tuple[List[str], List[str]]:
    payload = json.loads(champions_json.read_text(encoding="utf-8"))
    detectors = [str(item["detector_key"]) for item in payload.get("champions", [])]
    datasets = [str(item) for item in payload.get("datasets", [])]
    if not detectors:
        raise ValueError(f"no champion detectors found in {champions_json}")
    return detectors, datasets


def resolve_experiment_yaml(
    source_yaml: Path,
    output_yaml: Path,
    respect_image_filters: bool,
) -> Dict[str, object]:
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment YAML must be a mapping: {source_yaml}")
    payload = json.loads(json.dumps(payload))
    images_cfg = payload.get("images")
    if not isinstance(images_cfg, dict):
        images_cfg = {}
    if not respect_image_filters:
        images_cfg.pop("include_labels", None)
        images_cfg.pop("max_images_per_label", None)
    payload["images"] = images_cfg
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def run_command(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    dry_run: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("[dry-run]\n" + shlex.join(cmd) + "\n", encoding="utf-8")
        return 0
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    output = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {shlex.join(cmd)}\nsee {log_path}")
    return proc.returncode


def build_comparison_group(source_dataset_name: str, recipe_instance_id: object = "", variant: object = "") -> str:
    recipe_text = str(recipe_instance_id or "").strip()
    variant_text = str(variant if variant not in (None, "") else "").strip()
    if not recipe_text:
        return source_dataset_name
    return f"{source_dataset_name}::{recipe_text}::v{variant_text or '0'}"


def build_dataset_index_rows(
    source_datasets: List[str],
    generated_rows: List[Dict[str, object]],
    dataset_json_folder: Path,
    include_clean: bool,
    clean_rows: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    use_matched_subset = bool(clean_rows)
    if include_clean:
        if clean_rows:
            for row in clean_rows:
                merged = dict(row)
                merged["condition"] = "clean"
                merged["comparison_group"] = build_comparison_group(
                    str(merged.get("source_dataset_name", "")),
                    merged.get("recipe_instance_id", ""),
                    merged.get("variant", ""),
                )
                rows.append(merged)
        else:
            for dataset_name in source_datasets:
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "condition": "clean",
                        "source_dataset_name": dataset_name,
                        "recipe_id": "",
                        "recipe_instance_id": "",
                        "recipe_label": "",
                        "variant": "",
                        "manifest_path": str((dataset_json_folder / f"{dataset_name}.json").resolve()),
                        "input_jsonl": "",
                        "comparison_group": build_comparison_group(dataset_name),
                    }
                )
    for row in generated_rows:
        merged = dict(row)
        merged["condition"] = "distorted"
        if use_matched_subset:
            merged["comparison_group"] = build_comparison_group(
                str(merged.get("source_dataset_name", "")),
                merged.get("recipe_instance_id", ""),
                merged.get("variant", ""),
            )
        else:
            merged["comparison_group"] = build_comparison_group(str(merged.get("source_dataset_name", "")))
        rows.append(merged)
    return rows


def build_comparison_rows(all_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    clean_lookup: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in all_rows:
        if row.get("condition") == "clean" and row.get("status") == "success":
            clean_lookup[(str(row["detector_key"]), str(row.get("comparison_group", row["source_dataset_name"])))] = row

    comparison_rows: List[Dict[str, object]] = []
    for row in all_rows:
        if row.get("condition") != "distorted":
            continue
        clean_row = clean_lookup.get((str(row["detector_key"]), str(row.get("comparison_group", row["source_dataset_name"]))))
        comparison = dict(row)
        if clean_row is None:
            comparison.update(
                {
                    "clean_status": "missing",
                    "clean_auc": "",
                    "clean_ap": "",
                    "clean_acc": "",
                    "clean_eer": "",
                    "clean_runtime_sec": "",
                    "delta_auc": "",
                    "delta_ap": "",
                    "delta_acc": "",
                    "delta_eer": "",
                    "delta_runtime_sec": "",
                }
            )
        else:
            clean_auc = optional_float(clean_row.get("auc"))
            clean_ap = optional_float(clean_row.get("ap"))
            clean_acc = optional_float(clean_row.get("acc"))
            clean_eer = optional_float(clean_row.get("eer"))
            clean_runtime = optional_float(clean_row.get("runtime_sec"))
            row_auc = optional_float(row.get("auc"))
            row_ap = optional_float(row.get("ap"))
            row_acc = optional_float(row.get("acc"))
            row_eer = optional_float(row.get("eer"))
            row_runtime = optional_float(row.get("runtime_sec"))
            comparison.update(
                {
                    "clean_status": clean_row.get("status", ""),
                    "clean_auc": clean_auc if clean_auc is not None else "",
                    "clean_ap": clean_ap if clean_ap is not None else "",
                    "clean_acc": clean_acc if clean_acc is not None else "",
                    "clean_eer": clean_eer if clean_eer is not None else "",
                    "clean_runtime_sec": clean_runtime if clean_runtime is not None else "",
                    "delta_auc": (row_auc - clean_auc) if row_auc is not None and clean_auc is not None else "",
                    "delta_ap": (row_ap - clean_ap) if row_ap is not None and clean_ap is not None else "",
                    "delta_acc": (row_acc - clean_acc) if row_acc is not None and clean_acc is not None else "",
                    "delta_eer": (row_eer - clean_eer) if row_eer is not None and clean_eer is not None else "",
                    "delta_runtime_sec": (row_runtime - clean_runtime)
                    if row_runtime is not None and clean_runtime is not None
                    else "",
                }
            )
        comparison_rows.append(comparison)
    return comparison_rows


def build_detector_summary(comparison_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        if row.get("status") == "success":
            grouped[str(row["detector_key"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for detector_key, rows in sorted(grouped.items()):
        detector_name = str(rows[0].get("detector_name", detector_key))
        family = str(rows[0].get("family", ""))
        distorted_auc = [optional_float(row.get("auc")) for row in rows]
        delta_auc = [optional_float(row.get("delta_auc")) for row in rows]
        delta_ap = [optional_float(row.get("delta_ap")) for row in rows]
        delta_acc = [optional_float(row.get("delta_acc")) for row in rows]
        delta_eer = [optional_float(row.get("delta_eer")) for row in rows]
        summary_rows.append(
            {
                "detector_key": detector_key,
                "detector_name": detector_name,
                "family": family,
                "distortion_runs": len(rows),
                "mean_auc": mean(distorted_auc) or "",
                "worst_auc": min([value for value in distorted_auc if value is not None], default=""),
                "mean_delta_auc": mean(delta_auc) or "",
                "worst_delta_auc": min([value for value in delta_auc if value is not None], default=""),
                "mean_delta_ap": mean(delta_ap) or "",
                "mean_delta_acc": mean(delta_acc) or "",
                "mean_delta_eer": mean(delta_eer) or "",
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -(optional_float(row.get("mean_delta_auc")) if optional_float(row.get("mean_delta_auc")) is not None else -999.0),
            -(optional_float(row.get("mean_auc")) if optional_float(row.get("mean_auc")) is not None else -999.0),
            str(row["detector_key"]),
        )
    )
    return summary_rows


def write_markdown_report(
    report_path: Path,
    source_datasets: List[str],
    detectors: List[str],
    resolved_experiment: Dict[str, object],
    dataset_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    distorted_rows = [row for row in dataset_rows if row.get("condition") == "distorted"]
    failed_rows = [row for row in comparison_rows if row.get("status") not in ("success", "dry_run")]

    lines: List[str] = []
    lines.append("# Distortion Champion Evaluation")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Source datasets: `{', '.join(source_datasets)}`")
    lines.append(f"- Champion detectors: `{', '.join(detectors)}`")
    lines.append(f"- Distorted dataset manifests generated: `{len(distorted_rows)}`")
    lines.append(f"- Comparison rows: `{len(comparison_rows)}`")
    lines.append("")
    lines.append("## Distortion Recipes")
    lines.append("")
    for recipe in resolved_experiment.get("recipes", []):
        if isinstance(recipe, dict):
            lines.append(f"- `{recipe.get('recipe_id', '')}`")
    lines.append("")
    lines.append("## Detector Robustness Summary")
    lines.append("")
    lines.append("| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| `{row['detector_key']}` | {format_num(optional_float(row.get('mean_auc')), 6)} | "
            f"{format_num(optional_float(row.get('worst_auc')), 6)} | "
            f"{format_num(optional_float(row.get('mean_delta_auc')), 6)} | "
            f"{format_num(optional_float(row.get('worst_delta_auc')), 6)} | "
            f"{format_num(optional_float(row.get('mean_delta_ap')), 6)} | "
            f"{format_num(optional_float(row.get('mean_delta_acc')), 6)} | "
            f"{format_num(optional_float(row.get('mean_delta_eer')), 6)} | "
            f"{row['distortion_runs']} |"
        )
    lines.append("")
    if failed_rows:
        lines.append("## Failed Detector Runs")
        lines.append("")
        lines.append("| Dataset | Detector | Status | Detail |")
        lines.append("|---|---|---|---|")
        for row in failed_rows:
            lines.append(
                f"| `{row['dataset']}` | `{row['detector_key']}` | `{row['status']}` | `{row.get('status_detail', '')}` |"
            )
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Dataset index: `{output_dir / 'dataset_index.csv'}`")
    lines.append(f"- Combined raw runs: `{output_dir / 'combined_raw_runs.csv'}`")
    lines.append(f"- Distortion comparison: `{output_dir / 'detector_distortion_comparison.csv'}`")
    lines.append(f"- Distortion detector summary: `{output_dir / 'detector_distortion_summary.csv'}`")
    lines.append(f"- Generated dataset JSON folder: `{output_dir / 'generated_dataset_json'}`")
    lines.append(f"- Evaluation dataset JSON folder: `{output_dir / 'evaluation_dataset_json'}`")
    lines.append(f"- Benchmark runs folder: `{output_dir / 'benchmark_runs'}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate distortion treatments, register them as DeepfakeBench datasets, and evaluate champion detectors."
    )
    parser.add_argument("--champions-json", default="training/results/stage_c_existing_detectors/champions.json")
    parser.add_argument("--detectors", nargs="+", default=None, help="Override champion detector list.")
    parser.add_argument("--source-datasets", nargs="+", default=None, help="Override source datasets from champions.json.")
    parser.add_argument("--split", default="test", help="Manifest split to distort and evaluate.")
    parser.add_argument("--distortion-root", default="distortionPipeline")
    parser.add_argument("--experiment-yaml", default=None, help="Defaults to distortionPipeline/configs/experiments/exp.yaml")
    parser.add_argument("--recipes-dir", default=None, help="Defaults to distortionPipeline/configs/recipes")
    parser.add_argument("--dataset-json-folder", default="preprocessing/dataset_json")
    parser.add_argument("--weights-root", default="training/weights")
    parser.add_argument("--weights-map", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--timeout-minutes", type=float, default=20.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluate-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clean-baseline-mode",
        choices=("full", "matched_subset"),
        default="full",
        help="Use the full source manifest or a clean subset matched to each distortion treatment.",
    )
    parser.add_argument("--export-test-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--respect-experiment-image-filters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-mps-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    distortion_root = Path(args.distortion_root).expanduser()
    if not distortion_root.is_absolute():
        distortion_root = (repo_root / distortion_root).resolve()
    experiment_yaml = Path(args.experiment_yaml).expanduser() if args.experiment_yaml else distortion_root / "configs" / "experiments" / "exp.yaml"
    recipes_dir = Path(args.recipes_dir).expanduser() if args.recipes_dir else distortion_root / "configs" / "recipes"
    dataset_json_folder = Path(args.dataset_json_folder).expanduser()
    if not dataset_json_folder.is_absolute():
        dataset_json_folder = (repo_root / dataset_json_folder).resolve()
    champions_json = Path(args.champions_json).expanduser()
    if not champions_json.is_absolute():
        champions_json = (repo_root / champions_json).resolve()

    champion_detectors, champion_datasets = load_champion_detectors(champions_json)
    detectors = list(dict.fromkeys(args.detectors or champion_detectors))
    source_datasets = list(dict.fromkeys(args.source_datasets or champion_datasets))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (repo_root / output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (repo_root / "training" / "results" / f"distortion_champion_eval_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_logs = output_dir / "orchestrator_logs"
    work_dir = output_dir / "work"
    generated_dataset_json_dir = output_dir / "generated_dataset_json"
    evaluation_dataset_json_dir = output_dir / "evaluation_dataset_json"
    benchmark_runs_dir = output_dir / "benchmark_runs"
    generated_dataset_json_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dataset_json_dir.mkdir(parents=True, exist_ok=True)
    benchmark_runs_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(distortion_root))
    from src.pipeline.manifest_adapter import (  # type: ignore
        augmented_manifest_to_clean_subset_datasets,
        augmented_manifest_to_deepfakebench_datasets,
        deepfakebench_dataset_json_to_jsonl,
        write_dataset_index_csv,
    )

    resolved_experiment_path = work_dir / "resolved_experiment.yaml"
    resolved_experiment = resolve_experiment_yaml(
        experiment_yaml,
        resolved_experiment_path,
        respect_image_filters=args.respect_experiment_image_filters,
    )

    run_config = {
        "generated_at": datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "distortion_root": str(distortion_root),
        "experiment_yaml": str(experiment_yaml.resolve()),
        "resolved_experiment_yaml": str(resolved_experiment_path.resolve()),
        "recipes_dir": str(recipes_dir.resolve()),
        "dataset_json_folder": str(dataset_json_folder),
        "champions_json": str(champions_json),
        "detectors": detectors,
        "source_datasets": source_datasets,
        "split": args.split,
        "weights_root": args.weights_root,
        "weights_map": args.weights_map or "",
        "timeout_minutes": args.timeout_minutes,
        "evaluate_clean": args.evaluate_clean,
        "clean_baseline_mode": args.clean_baseline_mode,
        "export_test_artifacts": args.export_test_artifacts,
        "respect_experiment_image_filters": args.respect_experiment_image_filters,
        "disable_mps_fallback": args.disable_mps_fallback,
        "dry_run": args.dry_run,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    generated_rows: List[Dict[str, object]] = []
    clean_subset_rows: List[Dict[str, object]] = []
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

    for source_dataset in source_datasets:
        source_manifest = dataset_json_folder / f"{source_dataset}.json"
        if not source_manifest.exists():
            raise FileNotFoundError(f"source dataset manifest not found: {source_manifest}")
        export_jsonl = work_dir / "source_jsonl" / f"{sanitize_name(source_dataset)}__{args.split}.jsonl"
        export_jsonl.parent.mkdir(parents=True, exist_ok=True)
        deepfakebench_dataset_json_to_jsonl(
            source_manifest,
            export_jsonl,
            dataset_name=source_dataset,
            splits=(args.split,),
        )

        jobs_jsonl = work_dir / "jobs" / f"{sanitize_name(source_dataset)}.jsonl"
        augmented_jsonl = work_dir / "augmented" / f"{sanitize_name(source_dataset)}.jsonl"
        cache_dir = work_dir / "cache" / sanitize_name(source_dataset)
        distorted_out_dir = work_dir / "distorted_outputs" / sanitize_name(source_dataset)

        generate_cmd = [
            args.python_bin,
            "-m",
            "scripts.generate_manifest",
            "--dataset_jsonl",
            str(export_jsonl),
            "--recipes_dir",
            str(recipes_dir),
            "--experiment_yaml",
            str(resolved_experiment_path),
            "--out",
            str(jobs_jsonl),
        ]
        run_command(
            generate_cmd,
            cwd=distortion_root,
            log_path=orchestrator_logs / f"generate_manifest__{sanitize_name(source_dataset)}.log",
            env=env,
            dry_run=args.dry_run,
        )

        run_distortions_cmd = [
            args.python_bin,
            "-m",
            "scripts.run_distortions",
            "--manifest",
            str(jobs_jsonl),
            "--cache_dir",
            str(cache_dir),
            "--out_dir",
            str(distorted_out_dir),
            "--write_augmented_manifest",
            str(augmented_jsonl),
        ]
        run_command(
            run_distortions_cmd,
            cwd=distortion_root,
            log_path=orchestrator_logs / f"run_distortions__{sanitize_name(source_dataset)}.log",
            env=env,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            continue

        rows = augmented_manifest_to_deepfakebench_datasets(augmented_jsonl, generated_dataset_json_dir)
        for row in rows:
            row["source_manifest"] = str(source_manifest.resolve())
            row["source_jsonl"] = str(export_jsonl.resolve())
            row["jobs_jsonl"] = str(jobs_jsonl.resolve())
            row["augmented_manifest"] = str(augmented_jsonl.resolve())
            row["distorted_output_dir"] = str(distorted_out_dir.resolve())
        generated_rows.extend(rows)
        if args.evaluate_clean and args.clean_baseline_mode == "matched_subset":
            subset_rows = augmented_manifest_to_clean_subset_datasets(augmented_jsonl, generated_dataset_json_dir)
            for row in subset_rows:
                row["source_manifest"] = str(source_manifest.resolve())
                row["source_jsonl"] = str(export_jsonl.resolve())
                row["jobs_jsonl"] = str(jobs_jsonl.resolve())
                row["augmented_manifest"] = str(augmented_jsonl.resolve())
                row["distorted_output_dir"] = ""
            clean_subset_rows.extend(subset_rows)

    dataset_rows = build_dataset_index_rows(
        source_datasets,
        generated_rows,
        dataset_json_folder,
        include_clean=args.evaluate_clean,
        clean_rows=clean_subset_rows if args.clean_baseline_mode == "matched_subset" else None,
    )
    dataset_index_csv = output_dir / "dataset_index.csv"
    if dataset_rows:
        dataset_fieldnames = [
            "dataset_name",
            "condition",
            "comparison_group",
            "source_dataset_name",
            "recipe_id",
            "recipe_instance_id",
            "recipe_label",
            "variant",
            "manifest_path",
            "input_jsonl",
            "source_manifest",
            "source_jsonl",
            "jobs_jsonl",
            "augmented_manifest",
            "distorted_output_dir",
        ]
        write_csv(dataset_index_csv, dataset_rows, dataset_fieldnames)
        # Also emit the compact distortion-only index used by the adapter layer.
        if generated_rows:
            write_dataset_index_csv(output_dir / "generated_dataset_index.csv", generated_rows)
    else:
        dataset_index_csv.write_text("", encoding="utf-8")

    if args.clean_baseline_mode == "full":
        for source_dataset in source_datasets:
            source_manifest = dataset_json_folder / f"{source_dataset}.json"
            if source_manifest.exists():
                shutil.copyfile(source_manifest, evaluation_dataset_json_dir / source_manifest.name)
    for row in generated_rows + clean_subset_rows:
        manifest_path = row.get("manifest_path")
        if manifest_path:
            src = Path(str(manifest_path))
            shutil.copyfile(src, evaluation_dataset_json_dir / src.name)

    all_raw_rows: List[Dict[str, object]] = []
    benchmark_script = repo_root / "training" / "spatial_champion_benchmark.py"
    for dataset_row in dataset_rows:
        dataset_name = str(dataset_row["dataset_name"])
        benchmark_output_dir = benchmark_runs_dir / sanitize_name(dataset_name)
        cmd = [
            args.python_bin,
            str(benchmark_script),
            "--datasets",
            dataset_name,
            "--detectors",
            *detectors,
            "--weights-root",
            args.weights_root,
            "--dataset-json-folder",
            str(evaluation_dataset_json_dir),
            "--python-bin",
            args.python_bin,
            "--timeout-minutes",
            str(args.timeout_minutes),
            "--output-dir",
            str(benchmark_output_dir),
        ]
        if args.weights_map:
            cmd.extend(["--weights-map", args.weights_map])
        if args.export_test_artifacts:
            cmd.append("--export-test-artifacts")
        if args.disable_mps_fallback:
            cmd.append("--disable-mps-fallback")

        run_command(
            cmd,
            cwd=repo_root,
            log_path=orchestrator_logs / f"benchmark__{sanitize_name(dataset_name)}.log",
            env=env,
            timeout_seconds=max(args.timeout_minutes, 0.1) * 60.0 * max(len(detectors), 1),
            dry_run=args.dry_run,
        )

        if args.dry_run:
            continue

        raw_runs_path = benchmark_output_dir / "raw_runs.csv"
        for row in read_csv_rows(raw_runs_path):
            merged = dict(row)
            merged["condition"] = dataset_row.get("condition", "")
            merged["comparison_group"] = dataset_row.get("comparison_group", "")
            merged["source_dataset_name"] = dataset_row.get("source_dataset_name", "")
            merged["recipe_id"] = dataset_row.get("recipe_id", "")
            merged["recipe_instance_id"] = dataset_row.get("recipe_instance_id", "")
            merged["recipe_label"] = dataset_row.get("recipe_label", "")
            merged["variant"] = dataset_row.get("variant", "")
            merged["manifest_path"] = dataset_row.get("manifest_path", "")
            merged["input_jsonl"] = dataset_row.get("input_jsonl", "")
            merged["source_manifest"] = dataset_row.get("source_manifest", "")
            merged["source_jsonl"] = dataset_row.get("source_jsonl", "")
            merged["jobs_jsonl"] = dataset_row.get("jobs_jsonl", "")
            merged["augmented_manifest"] = dataset_row.get("augmented_manifest", "")
            merged["distorted_output_dir"] = dataset_row.get("distorted_output_dir", "")
            merged["benchmark_output_dir"] = str(benchmark_output_dir.resolve())
            all_raw_rows.append(merged)

    combined_raw_runs = output_dir / "combined_raw_runs.csv"
    comparison_csv = output_dir / "detector_distortion_comparison.csv"
    detector_summary_csv = output_dir / "detector_distortion_summary.csv"
    report_md = output_dir / "distortion_champion_report.md"

    if args.dry_run:
        report_md.write_text(
            "# Distortion Champion Evaluation (Dry Run)\n\n"
            f"- Source datasets: `{', '.join(source_datasets)}`\n"
            f"- Detectors: `{', '.join(detectors)}`\n"
            f"- Output dir: `{output_dir}`\n",
            encoding="utf-8",
        )
        return 0

    raw_fieldnames = [
        "condition",
        "comparison_group",
        "source_dataset_name",
        "recipe_id",
        "recipe_instance_id",
        "recipe_label",
        "variant",
        "manifest_path",
        "input_jsonl",
        "source_manifest",
        "source_jsonl",
        "jobs_jsonl",
        "augmented_manifest",
        "distorted_output_dir",
        "benchmark_output_dir",
        "detector_key",
        "detector_name",
        "family",
        "dataset",
        "config_path",
        "weights_path",
        "weight_source",
        "status",
        "status_detail",
        "return_code",
        "device",
        "runtime_sec",
        "acc",
        "auc",
        "eer",
        "ap",
        "video_auc",
        "artifacts_dir",
        "prediction_artifacts_path",
        "metrics_artifacts_path",
        "log_path",
        "command",
    ]
    write_csv(combined_raw_runs, all_raw_rows, raw_fieldnames)

    comparison_rows = build_comparison_rows(all_raw_rows)
    comparison_fieldnames = [
        *raw_fieldnames,
        "clean_status",
        "clean_auc",
        "clean_ap",
        "clean_acc",
        "clean_eer",
        "clean_runtime_sec",
        "delta_auc",
        "delta_ap",
        "delta_acc",
        "delta_eer",
        "delta_runtime_sec",
    ]
    write_csv(comparison_csv, comparison_rows, comparison_fieldnames)

    summary_rows = build_detector_summary(comparison_rows)
    summary_fieldnames = [
        "detector_key",
        "detector_name",
        "family",
        "distortion_runs",
        "mean_auc",
        "worst_auc",
        "mean_delta_auc",
        "worst_delta_auc",
        "mean_delta_ap",
        "mean_delta_acc",
        "mean_delta_eer",
    ]
    write_csv(detector_summary_csv, summary_rows, summary_fieldnames)
    write_markdown_report(
        report_md,
        source_datasets=source_datasets,
        detectors=detectors,
        resolved_experiment=resolved_experiment,
        dataset_rows=dataset_rows,
        comparison_rows=comparison_rows,
        summary_rows=summary_rows,
        output_dir=output_dir,
    )
    print(f"[done] output_dir={output_dir}")
    print(f"[done] dataset_index={dataset_index_csv}")
    print(f"[done] combined_raw_runs={combined_raw_runs}")
    print(f"[done] comparison={comparison_csv}")
    print(f"[done] detector_summary={detector_summary_csv}")
    print(f"[done] report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
