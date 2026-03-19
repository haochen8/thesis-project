# Distortion Integration Manual Test Steps

Generated: `2026-03-19`
Updated after the Python 3.11 MPS migration.

## Purpose

These steps verify the current canonical distortion-to-detector workflow using the migrated MPS-capable environment.

## Preconditions

- Conda env: `DeepfakeBench311`
- Working directory: `/Users/Hao/thesis-project`
- Run these commands from a normal terminal session on macOS. Inside the Codex sandbox, `mps_available` may appear false even when the environment is correct.

## Step 0: Confirm MPS In `DeepfakeBench311`

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench311

python - <<'PY'
import sys, platform, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('mac_ver', platform.mac_ver())
print('mps_built', torch.backends.mps.is_built())
print('mps_available', torch.backends.mps.is_available())
print(torch.zeros(1, device='mps').device)
PY
```

Expected result:

- `python 3.11.15`
- `torch 2.12.0.dev20260318`
- `mps_built True`
- `mps_available True`
- final line prints `mps:0`

## Step 1: Verify The Benchmark Entry Point

This reproduces the clean Xception mini test in the migrated environment.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench311

MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/.cache \
python /Users/Hao/thesis-project/training/test.py \
  --detector_path /Users/Hao/thesis-project/training/config/detector/xception_cpu_mini.yaml \
  --weights_path /Users/Hao/thesis-project/training/weights/xception_best.pth \
  --test_dataset Dataset-1-mini \
  --dataset_json_folder /Users/Hao/thesis-project/preprocessing/dataset_json
```

Expected result:

- `===> Using device: mps`
- `acc: 0.631`
- `auc: 0.6656719999999999`
- `eer: 0.392`
- `ap: 0.7117617465412283`
- `===> Test Done!`

## Step 2: Run The Top-3 Matched-Subset Distortion Benchmark On `Dataset-1-mini`

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench311

MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/.cache \
python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets Dataset-1-mini \
  --detectors mesonet mesoinception xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small.yaml \
  --clean-baseline-mode matched_subset \
  --respect-experiment-image-filters \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311
```

Files to inspect:

- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/combined_raw_runs.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/distortion_champion_report.md`

Expected result summary:

- all `6` runs succeed (`3` clean subset + `3` distorted)
- all runs use `device=mps`
- Gaussian blur matched-subset deltas:
  - `mesonet`: `delta_auc=-0.0312`
  - `mesoinception`: `delta_auc=+0.0276`
  - `xception`: `delta_auc=-0.0616`

## Step 3: Run The Same Matched-Subset Benchmark On `NVIDIA-dataset-mini`

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench311

MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/.cache \
python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets NVIDIA-dataset-mini \
  --detectors mesonet mesoinception xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small.yaml \
  --clean-baseline-mode matched_subset \
  --respect-experiment-image-filters \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311
```

Files to inspect:

- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/combined_raw_runs.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/distortion_champion_report.md`

Expected result summary:

- all `6` runs succeed (`3` clean subset + `3` distorted)
- all runs use `device=mps`
- Gaussian blur matched-subset deltas:
  - `mesonet`: `delta_auc=+0.0592`
  - `mesoinception`: `delta_auc=-0.0864`
  - `xception`: `delta_auc=-0.0608`

## Step 4: Inspect Per-Detector Prediction Artifacts

Each detector run writes per-sample outputs.

Examples:

- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/benchmark_runs/Dataset-1-mini__gaussian_blur_v1__4a1525bdf7__v0/test_artifacts/xception__Dataset-1-mini__gaussian_blur_v1__4a1525bdf7__v0/Dataset-1-mini__gaussian_blur_v1__4a1525bdf7__v0/predictions.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/benchmark_runs/NVIDIA-dataset-mini__gaussian_blur_v1__4a1525bdf7__v0/test_artifacts/mesonet__NVIDIA-dataset-mini__gaussian_blur_v1__4a1525bdf7__v0/NVIDIA-dataset-mini__gaussian_blur_v1__4a1525bdf7__v0/metrics.json`

Use these to verify:

- image-level probabilities were exported
- the clean and distorted dataset names are different
- the comparison CSVs were derived from structured artifacts rather than scraped logs

## Recommended Next Expansion

1. Add `jpeg`, `noise`, and `text_overlay` experiment YAMLs using the same matched-subset path.
2. Reuse `DeepfakeBench311` for those runs.
3. Keep the detector set fixed until distortion robustness ranking stabilizes.
