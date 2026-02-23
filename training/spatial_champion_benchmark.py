#!/usr/bin/env python3
"""
Run a spatial-detector leaderboard and pick top-3 champions.

This script runs DeepfakeBench `training/test.py` detector-by-detector and
dataset-by-dataset, parses test metrics, writes CSV artifacts, and ranks
detectors with a reproducible scoring rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


METRIC_KEYS = ("acc", "auc", "eer", "ap", "video_auc")
DEFAULT_DATASETS = ("Dataset-1", "NVIDIA-dataset")
STAGE_DATASETS = {
    "A": ("Dataset-1-mini", "NVIDIA-dataset-mini"),
    "B": ("Dataset-1-stageB", "NVIDIA-dataset-stageB"),
    "C": ("Dataset-1", "NVIDIA-dataset"),
}


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    name: str
    family: str
    config_relpath: str
    aliases: Tuple[str, ...] = ()
    explicit_weight: Optional[str] = None


# Spatial-only pool requested by user:
# - 5 Naive + Spatial detectors
# - Video detectors are intentionally excluded.
SPATIAL_DETECTORS: Tuple[DetectorSpec, ...] = (
    DetectorSpec("xception", "Xception", "naive", "training/config/detector/xception_cpu.yaml", ("xception_cpu",), "xception_best.pth"),
    DetectorSpec("mesonet", "MesoNet", "naive", "training/config/detector/meso4.yaml", ("meso4",), "meso4_best.pth"),
    DetectorSpec("mesoinception", "MesoInception", "naive", "training/config/detector/meso4Inception.yaml", ("meso4inception",), "meso4Incep_best.pth"),
    DetectorSpec("cnn-aug", "CNN-Aug", "naive", "training/config/detector/resnet34.yaml", ("resnet34", "cnn_aug"), "cnnaug_best.pth"),
    DetectorSpec("efficientnet-b4", "EfficientNet-B4", "naive", "training/config/detector/efficientnetb4.yaml", ("efficientnetb4",), "effnb4_best.pth"),
    DetectorSpec("capsule", "Capsule", "spatial", "training/config/detector/capsule_net.yaml", ("capsule_net",), "capsule_best.pth"),
    DetectorSpec("dsp-fwa", "DSP-FWA", "spatial", "training/config/detector/fwa.yaml", ("fwa", "dspfwa"), "fwa_best.pth"),
    DetectorSpec("face-xray", "Face X-ray", "spatial", "training/config/detector/facexray.yaml", ("facexray", "facexray"), "facexray_best.pth"),
    DetectorSpec("ffd", "FFD", "spatial", "training/config/detector/ffd.yaml", ("ffd",), "ffd_best.pth"),
    DetectorSpec("core", "CORE", "spatial", "training/config/detector/core.yaml", ("core",), "core_best.pth"),
    DetectorSpec("recce", "RECCE", "spatial", "training/config/detector/recce.yaml", ("recce",), "recce_best.pth"),
    DetectorSpec("ucf", "UCF", "spatial", "training/config/detector/ucf.yaml", ("ucf",), "ucf_best.pth"),
    DetectorSpec("local-relation", "Local-relation", "spatial", "training/config/detector/lrl.yaml", ("lrl", "localrelation"), "lrl_best.pth"),
    DetectorSpec("iid", "IID", "spatial", "training/config/detector/iid.yaml", ("iid",), "iid_best.pth"),
    DetectorSpec("rfm", "RFM", "spatial", "training/config/detector/rfm.yaml", ("rfm",), "rfm_best.pth"),
    DetectorSpec("sia", "SIA", "spatial", "training/config/detector/sia.yaml", ("sia",), "sia_best.pth"),
    DetectorSpec("sladd", "SLADD", "spatial", "training/config/detector/sladd_detector.yaml", ("sladd", "sladd_detector"), "sladd_best.pth"),
    DetectorSpec("uia-vit", "UIA-ViT", "spatial", "training/config/detector/uia_vit.yaml", ("uia_vit", "uiavit"), "uia_vit_best.pth"),
    DetectorSpec("clip", "CLIP", "spatial", "training/config/detector/clip.yaml", ("clip",), "clip_best.pth"),
    DetectorSpec("sbi", "SBI", "spatial", "training/config/detector/sbi.yaml", ("sbi",), "sbi_best.pth"),
    DetectorSpec("pcl-i2g", "PCL-I2G", "spatial", "training/config/detector/pcl_xception.yaml", ("pcl_xception", "pcli2g"), "pcl_xception_best.pth"),
    DetectorSpec("multi-attention", "Multi-Attention", "spatial", "training/config/detector/multi_attention.yaml", ("multi_attention", "multiattention"), "multi_attention_best.pth"),
    DetectorSpec("lsda", "LSDA", "spatial", "training/config/detector/lsda.yaml", ("lsda",), "lsda_best.pth"),
    DetectorSpec("effort", "Effort", "spatial", "training/config/detector/effort.yaml", ("effort",), "effort_best.pth"),
)

# Optional frequency detectors.
FREQUENCY_DETECTORS: Tuple[DetectorSpec, ...] = (
    DetectorSpec("f3net", "F3Net", "frequency", "training/config/detector/f3net.yaml", ("f3net",), "f3net_best.pth"),
    DetectorSpec("spsl", "SPSL", "frequency", "training/config/detector/spsl.yaml", ("spsl",), "spsl_best.pth"),
    DetectorSpec("srm", "SRM", "frequency", "training/config/detector/srm.yaml", ("srm",), "srm_best.pth"),
)


def norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def load_weights_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("--weights-map JSON must be an object mapping detector key -> weight path")
    return {str(k): str(v) for k, v in data.items()}


def weight_match_score(path: Path, tokens: Iterable[str]) -> int:
    name = path.name.lower()
    stem = norm_token(path.stem)
    score = 0
    for token in tokens:
        if not token:
            continue
        if token in stem:
            score += 10
        elif token in norm_token(name):
            score += 6
    # Only apply tie-break bonuses after at least one detector-token matched.
    if score > 0:
        if "best" in name:
            score += 5
        if "final" in name:
            score += 3
        if "latest" in name:
            score += 2
    return score


def resolve_weight_path(
    detector: DetectorSpec,
    weights_root: Path,
    weights_map: Dict[str, str],
    all_weights: List[Path],
) -> Tuple[Optional[Path], str]:
    if detector.key in weights_map:
        forced = Path(weights_map[detector.key]).expanduser()
        if not forced.is_absolute():
            forced = (weights_root / forced).resolve()
        if forced.exists():
            return forced, "weights_map"
        return None, f"weights_map_missing:{forced}"

    if detector.explicit_weight:
        explicit = (weights_root / detector.explicit_weight).resolve()
        if explicit.exists():
            return explicit, "explicit"

    if detector.key == "xception":
        xcp = (weights_root / "xception_best.pth").resolve()
        if xcp.exists():
            return xcp, "xception_default"

    tokens = {
        norm_token(detector.key),
        norm_token(detector.name),
        norm_token(Path(detector.config_relpath).stem),
    }
    for alias in detector.aliases:
        tokens.add(norm_token(alias))
    tokens = {x for x in tokens if x}

    scored: List[Tuple[int, float, Path]] = []
    for pth in all_weights:
        score = weight_match_score(pth, tokens)
        if score > 0:
            scored.append((score, pth.stat().st_mtime, pth))
    if not scored:
        return None, "not_found"
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2], "auto_match"


def parse_test_output(output: str) -> Dict[str, Optional[float]]:
    parsed: Dict[str, Optional[float]] = {k: None for k in METRIC_KEYS}
    for metric in METRIC_KEYS:
        # Handles lines like: "auc: 0.66567"
        match = re.search(rf"(?m)^{re.escape(metric)}:\s*([^\s]+)\s*$", output)
        parsed[metric] = safe_float(match.group(1)) if match else None
    return parsed


def parse_device(output: str) -> Optional[str]:
    match = re.search(r"===> Using device:\s*([A-Za-z0-9_:-]+)", output)
    if match:
        return match.group(1)
    return None


def parse_dataset_header(output: str) -> Optional[str]:
    match = re.search(r"(?m)^dataset:\s*(.+?)\s*$", output)
    if match:
        return match.group(1)
    return None


def build_detector_lookup(detectors: Iterable[DetectorSpec]) -> Dict[str, DetectorSpec]:
    lookup: Dict[str, DetectorSpec] = {}
    for det in detectors:
        keys = {det.key, det.name, det.key.replace("-", "_")}
        keys.update(det.aliases)
        for item in keys:
            lookup[norm_token(item)] = det
    return lookup


def select_detectors(all_detectors: Tuple[DetectorSpec, ...], requested: List[str]) -> List[DetectorSpec]:
    if not requested:
        return list(all_detectors)
    lookup = build_detector_lookup(all_detectors)
    selected: List[DetectorSpec] = []
    seen = set()
    expanded_tokens: List[str] = []
    for token in requested:
        expanded_tokens.extend([x for x in token.split(",") if x.strip()])
    for token in expanded_tokens:
        normalized = norm_token(token)
        if normalized in ("all", "spatial"):
            for det in all_detectors:
                if det.key not in seen:
                    selected.append(det)
                    seen.add(det.key)
            continue
        det = lookup.get(normalized)
        if det is None:
            raise ValueError(f"Unknown detector '{token}'. Use --list-detectors to inspect available keys.")
        if det.key not in seen:
            selected.append(det)
            seen.add(det.key)
    return selected


def average(values: List[Optional[float]]) -> Optional[float]:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def format_num(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def to_optional_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run spatial detector leaderboard and pick top-3 champions.")
    parser.add_argument("--stage", choices=("A", "B", "C"), default=None, help="Preset stage datasets (A/B/C).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names in dataset_json folder.")
    parser.add_argument("--detectors", nargs="+", default=[], help="Detector keys/names (default: all spatial pool).")
    parser.add_argument("--include-frequency", action="store_true", help="Include frequency detectors (F3Net/SPSL/SRM) in the candidate pool.")
    parser.add_argument("--weights-root", default="training/weights", help="Folder containing detector weight files.")
    parser.add_argument("--weights-map", default=None, help="JSON file mapping detector key -> weight path.")
    parser.add_argument("--output-dir", default=None, help="Output folder for logs/CSVs (default: timestamped folder).")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run training/test.py.")
    parser.add_argument("--timeout-minutes", type=float, default=20.0, help="Per-run timeout in minutes.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved runs and write plan files without executing.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at first failed execution.")
    parser.add_argument("--list-detectors", action="store_true", help="List supported spatial detectors and exit.")
    parser.add_argument("--disable-mps-fallback", action="store_true", help="Do not set PYTORCH_ENABLE_MPS_FALLBACK=1.")
    args = parser.parse_args()

    candidate_pool: Tuple[DetectorSpec, ...] = SPATIAL_DETECTORS + (FREQUENCY_DETECTORS if args.include_frequency else ())

    if args.list_detectors:
        for spec in candidate_pool:
            print(f"{spec.key:16s} | {spec.name:16s} | {spec.family:9s} | {spec.config_relpath}")
        return 0

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    test_script = repo_root / "training" / "test.py"
    weights_root = Path(args.weights_root).expanduser()
    if not weights_root.is_absolute():
        weights_root = (repo_root / weights_root).resolve()

    weights_map = load_weights_map(Path(args.weights_map).expanduser().resolve()) if args.weights_map else {}
    if args.stage and args.datasets:
        raise ValueError("Use either --stage or --datasets, not both.")

    if args.stage:
        datasets = list(STAGE_DATASETS[args.stage])
    elif args.datasets:
        datasets = list(dict.fromkeys(args.datasets))
    else:
        datasets = list(DEFAULT_DATASETS)

    selected = select_detectors(candidate_pool, args.detectors)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (repo_root / output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (repo_root / "training" / "results" / f"spatial_champions_{stamp}").resolve()
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_weights = sorted(weights_root.rglob("*.pth")) if weights_root.exists() else []

    run_config = {
        "generated_at": datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "test_script": str(test_script),
        "stage": args.stage or "",
        "datasets": datasets,
        "selected_detectors": [spec.key for spec in selected],
        "weights_root": str(weights_root),
        "weights_map": weights_map,
        "timeout_minutes": args.timeout_minutes,
        "dry_run": args.dry_run,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    raw_rows: List[Dict[str, object]] = []
    timeout_seconds = max(args.timeout_minutes, 0.1) * 60.0

    for detector in selected:
        config_path = (repo_root / detector.config_relpath).resolve()
        weight_path, weight_source = resolve_weight_path(detector, weights_root, weights_map, all_weights)

        for dataset in datasets:
            row: Dict[str, object] = {
                "detector_key": detector.key,
                "detector_name": detector.name,
                "family": detector.family,
                "dataset": dataset,
                "config_path": str(config_path),
                "weights_path": str(weight_path) if weight_path else "",
                "weight_source": weight_source,
                "status": "",
                "status_detail": "",
                "return_code": "",
                "device": "",
                "runtime_sec": "",
                "acc": "",
                "auc": "",
                "eer": "",
                "ap": "",
                "video_auc": "",
                "log_path": "",
                "command": "",
            }

            dataset_slug = sanitize_name(dataset)
            log_path = logs_dir / f"{detector.key}__{dataset_slug}.log"
            row["log_path"] = str(log_path)

            if not config_path.exists():
                row["status"] = "config_missing"
                row["status_detail"] = str(config_path)
                raw_rows.append(row)
                continue
            if weight_path is None:
                row["status"] = "missing_weight"
                if detector.explicit_weight:
                    row["status_detail"] = f"expected:{weights_root / detector.explicit_weight}"
                else:
                    row["status_detail"] = "No matching .pth found"
                raw_rows.append(row)
                continue

            cmd = [
                args.python_bin,
                str(test_script),
                "--detector_path",
                str(config_path),
                "--weights_path",
                str(weight_path),
                "--test_dataset",
                dataset,
            ]
            row["command"] = shlex.join(cmd)

            if args.dry_run:
                row["status"] = "dry_run"
                raw_rows.append(row)
                continue

            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
            env.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
            if not args.disable_mps_fallback:
                env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                elapsed = time.perf_counter() - t0
                combined_output = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
                log_path.write_text(combined_output)

                row["return_code"] = proc.returncode
                row["runtime_sec"] = round(elapsed, 3)
                row["device"] = parse_device(combined_output) or ""

                metrics = parse_test_output(combined_output)
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value if metric_value is not None else ""

                parsed_dataset = parse_dataset_header(combined_output)
                if parsed_dataset and parsed_dataset != dataset:
                    row["status_detail"] = f"dataset_mismatch:{parsed_dataset}"

                if proc.returncode == 0 and metrics["auc"] is not None and metrics["ap"] is not None and metrics["eer"] is not None:
                    row["status"] = "success"
                else:
                    row["status"] = "failed"
                    if not row["status_detail"]:
                        row["status_detail"] = "nonzero_exit_or_metrics_missing"
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - t0
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                combined_output = (stdout or "") + ("\n" if stdout and stderr else "") + (stderr or "")
                log_path.write_text(combined_output)

                row["status"] = "timeout"
                row["status_detail"] = f"timeout_after_{int(timeout_seconds)}s"
                row["runtime_sec"] = round(elapsed, 3)
                row["device"] = parse_device(combined_output) or ""

            raw_rows.append(row)

            if args.stop_on_error and row["status"] not in ("success", "dry_run"):
                break
        if args.stop_on_error and raw_rows and raw_rows[-1]["status"] not in ("success", "dry_run"):
            break

    raw_fieldnames = [
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
        "log_path",
        "command",
    ]
    write_csv(output_dir / "raw_runs.csv", raw_rows, raw_fieldnames)

    by_detector: Dict[str, List[Dict[str, object]]] = {}
    for row in raw_rows:
        by_detector.setdefault(str(row["detector_key"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for spec in selected:
        rows = by_detector.get(spec.key, [])
        success_rows = [r for r in rows if r.get("status") == "success"]
        success_by_dataset = {str(r["dataset"]): r for r in success_rows}

        auc_values = [safe_float(str(success_by_dataset[d]["auc"])) if d in success_by_dataset else None for d in datasets]
        ap_values = [safe_float(str(success_by_dataset[d]["ap"])) if d in success_by_dataset else None for d in datasets]
        eer_values = [safe_float(str(success_by_dataset[d]["eer"])) if d in success_by_dataset else None for d in datasets]
        acc_values = [safe_float(str(success_by_dataset[d]["acc"])) if d in success_by_dataset else None for d in datasets]
        runtime_values = [safe_float(str(success_by_dataset[d]["runtime_sec"])) if d in success_by_dataset else None for d in datasets]

        auc_mean = average(auc_values)
        ap_mean = average(ap_values)
        eer_mean = average(eer_values)
        acc_mean = average(acc_values)
        runtime_total = sum(v for v in runtime_values if v is not None) if any(v is not None for v in runtime_values) else None

        auc_gap = None
        if len(datasets) >= 2 and auc_values[0] is not None and auc_values[1] is not None:
            auc_gap = abs(auc_values[0] - auc_values[1])

        summary_rows.append(
            {
                "detector_key": spec.key,
                "detector_name": spec.name,
                "family": spec.family,
                "runs_expected": len(datasets),
                "runs_success": len(success_rows),
                "complete": int(len(success_rows) == len(datasets)),
                "acc_mean": acc_mean if acc_mean is not None else "",
                "auc_mean": auc_mean if auc_mean is not None else "",
                "eer_mean": eer_mean if eer_mean is not None else "",
                "ap_mean": ap_mean if ap_mean is not None else "",
                "auc_gap": auc_gap if auc_gap is not None else "",
                "runtime_total_sec": runtime_total if runtime_total is not None else "",
                "status_note": "ok" if len(success_rows) == len(datasets) else "incomplete_or_failed",
            }
        )

    complete_rows = [r for r in summary_rows if int(r["complete"]) == 1 and r["auc_mean"] != "" and r["ap_mean"] != "" and r["eer_mean"] != "" and r["runtime_total_sec"] != ""]
    runtimes = [float(r["runtime_total_sec"]) for r in complete_rows]
    rt_min = min(runtimes) if runtimes else None
    rt_max = max(runtimes) if runtimes else None

    for row in complete_rows:
        runtime = float(row["runtime_total_sec"])
        if rt_min is None or rt_max is None or abs(rt_max - rt_min) < 1e-12:
            runtime_norm = 0.0
        else:
            runtime_norm = (runtime - rt_min) / (rt_max - rt_min)
        auc_gap = float(row["auc_gap"]) if row["auc_gap"] != "" else 0.0
        score = (
            0.55 * float(row["auc_mean"])
            + 0.20 * float(row["ap_mean"])
            + 0.10 * float(row["acc_mean"])
            - 0.15 * float(row["eer_mean"])
            - 0.08 * runtime_norm
            - 0.12 * auc_gap
        )
        row["runtime_norm"] = runtime_norm
        row["score"] = score

    # Populate score/runtime_norm for all rows for CSV consistency.
    complete_by_key = {str(r["detector_key"]): r for r in complete_rows}
    for row in summary_rows:
        comp = complete_by_key.get(str(row["detector_key"]))
        row["runtime_norm"] = comp["runtime_norm"] if comp is not None else ""
        row["score"] = comp["score"] if comp is not None else ""

    summary_rows.sort(
        key=lambda r: (
            -float(r["score"]) if r["score"] != "" else float("inf"),
            -float(r["auc_mean"]) if r["auc_mean"] != "" else float("inf"),
            str(r["detector_key"]),
        )
    )

    summary_fieldnames = [
        "detector_key",
        "detector_name",
        "family",
        "runs_expected",
        "runs_success",
        "complete",
        "acc_mean",
        "auc_mean",
        "eer_mean",
        "ap_mean",
        "auc_gap",
        "runtime_total_sec",
        "runtime_norm",
        "score",
        "status_note",
    ]
    write_csv(output_dir / "detector_summary.csv", summary_rows, summary_fieldnames)

    ranked = [r for r in summary_rows if r["score"] != ""]
    ranked.sort(key=lambda r: float(r["score"]), reverse=True)
    champions = ranked[:3]

    champions_payload = {
        "generated_at": datetime.now().isoformat(),
        "datasets": datasets,
        "scoring_formula": "0.55*AUC + 0.20*AP + 0.10*ACC - 0.15*EER - 0.08*runtime_norm - 0.12*auc_gap",
        "champions": [
            {
                "rank": idx + 1,
                "detector_key": row["detector_key"],
                "detector_name": row["detector_name"],
                "family": row["family"],
                "score": row["score"],
                "auc_mean": row["auc_mean"],
                "ap_mean": row["ap_mean"],
                "eer_mean": row["eer_mean"],
                "acc_mean": row["acc_mean"],
                "auc_gap": row["auc_gap"],
                "runtime_total_sec": row["runtime_total_sec"],
            }
            for idx, row in enumerate(champions)
        ],
        "num_ranked_complete_detectors": len(ranked),
    }
    (output_dir / "champions.json").write_text(json.dumps(champions_payload, indent=2))

    md_lines: List[str] = []
    md_lines.append("# Spatial Detector Champion Report")
    md_lines.append("")
    md_lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    md_lines.append(f"- Datasets: `{', '.join(datasets)}`")
    md_lines.append(f"- Selected detectors: `{len(selected)}`")
    md_lines.append(f"- Ranked complete detectors: `{len(ranked)}`")
    md_lines.append(f"- Output dir: `{output_dir}`")
    md_lines.append("")
    md_lines.append("## Top 3 Champions")
    md_lines.append("")
    md_lines.append("| Rank | Detector | Family | Score | AUC(mean) | AP(mean) | EER(mean) | ACC(mean) | AUC gap | Runtime(s) |")
    md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    if champions:
        for idx, row in enumerate(champions, start=1):
            md_lines.append(
                f"| {idx} | {row['detector_name']} (`{row['detector_key']}`) | {row['family']} | "
                f"{format_num(to_optional_float(row['score']), 6)} | {format_num(to_optional_float(row['auc_mean']), 6)} | "
                f"{format_num(to_optional_float(row['ap_mean']), 6)} | {format_num(to_optional_float(row['eer_mean']), 6)} | "
                f"{format_num(to_optional_float(row['acc_mean']), 6)} | {format_num(to_optional_float(row['auc_gap']), 6)} | "
                f"{format_num(to_optional_float(row['runtime_total_sec']), 3)} |"
            )
    else:
        md_lines.append("| - | - | - | - | - | - | - | - | - | - |")
    md_lines.append("")

    failed = [r for r in raw_rows if r["status"] not in ("success", "dry_run")]
    if failed:
        md_lines.append("## Failed or Missing Runs")
        md_lines.append("")
        md_lines.append("| Detector | Dataset | Status | Detail | Log |")
        md_lines.append("|---|---|---|---|---|")
        for row in failed:
            md_lines.append(
                f"| `{row['detector_key']}` | `{row['dataset']}` | `{row['status']}` | "
                f"`{row['status_detail']}` | `{row['log_path']}` |"
            )
        md_lines.append("")

    md_lines.append("## Artifacts")
    md_lines.append("")
    md_lines.append(f"- Raw run table: `{output_dir / 'raw_runs.csv'}`")
    md_lines.append(f"- Detector summary: `{output_dir / 'detector_summary.csv'}`")
    md_lines.append(f"- Champion JSON: `{output_dir / 'champions.json'}`")
    md_lines.append(f"- Run config: `{output_dir / 'run_config.json'}`")
    md_lines.append("")

    (output_dir / "leaderboard.md").write_text("\n".join(md_lines))

    print(f"[done] output_dir={output_dir}")
    print(f"[done] raw_runs={output_dir / 'raw_runs.csv'}")
    print(f"[done] summary={output_dir / 'detector_summary.csv'}")
    print(f"[done] champions={output_dir / 'champions.json'}")
    if champions:
        print("[done] top_3=" + ", ".join(str(x["detector_key"]) for x in champions))
    else:
        print("[done] top_3=none (insufficient complete detector results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
