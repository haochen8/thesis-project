# Distortion Integration Manual Test Steps

Generated: `2026-03-19`

## Purpose

These steps verify the new distortion-to-detector integration in increasing order of cost.

## Preconditions

- Conda env: `DeepfakeBench`
- Working directory: `/Users/Hao/thesis-project`

## Step 0: Check whether MPS is currently available

Run this first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench

python - <<'PY'
import torch
print('torch', torch.__version__)
print('mps_built', torch.backends.mps.is_built())
print('mps_available', torch.backends.mps.is_available())
PY
```

Interpretation:

- If `mps_available` is `True`, use the full mini smoke steps below.
- If `mps_available` is `False`, the benchmark will run on CPU. In that case, use the quickcheck and the small paired smoke step first.
- Current observed state on `2026-03-19`: `torch 2.8.0`, `mps_built=True`, `mps_available=False`.

## Step 1: Fast end-to-end quickcheck

Use a tiny distortion run with one detector and no clean baseline comparison.
Start with `xception` because it is the most predictable first validator in this codebase.

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench

python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets Dataset-1-mini \
  --detectors xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_quickcheck.yaml \
  --respect-experiment-image-filters \
  --no-evaluate-clean \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception
```

What this should do:

1. Export `Dataset-1-mini` test manifest into distortion JSONL
2. Generate 10 distortion jobs (5 real + 5 fake, one recipe)
3. Write one distorted dataset manifest
4. Run `xception` on that generated dataset
5. Write benchmark and comparison artifacts

Files to check:

- `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception/run_config.json`
- `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception/dataset_index.csv`
- `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception/generated_dataset_json/`
- `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception/benchmark_runs/`
- `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception/combined_raw_runs.csv`

Success criteria:

- `generated_dataset_json` contains one `.json` file
- `benchmark_runs` contains one dataset run folder
- `combined_raw_runs.csv` contains `xception`
- `status` is `success`

## Step 2: Small paired smoke run

Use a reduced paired run so clean vs distorted deltas are generated quickly even on CPU.

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench

python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets Dataset-1-mini \
  --detectors xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small.yaml \
  --clean-baseline-mode matched_subset \
  --respect-experiment-image-filters \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched
```

What this should do:

1. Generate a clean subset manifest that matches the sampled distortion jobs
2. Generate one reduced distorted dataset with `50` real + `50` fake images
3. Evaluate `xception` on the distorted dataset
4. Evaluate `xception` on the matched clean subset
5. Write clean-vs-distorted deltas

Files to check:

- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/evaluation_dataset_json/`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/combined_raw_runs.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/detector_distortion_comparison.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/detector_distortion_summary.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/distortion_champion_report.md`

Success criteria:

- `combined_raw_runs.csv` contains one clean row and one distorted row for `xception`
- the clean row dataset name contains `__clean_subset__`
- `detector_distortion_comparison.csv` contains non-empty `clean_auc` and `delta_auc`
- `prediction_artifacts_path` and `metrics_artifacts_path` are populated

## Step 3: Meaningful mini-dataset smoke run

Use the full `Dataset-1-mini` size for one distortion recipe so clean vs distorted comparison is valid.
Keep `xception` for this first full clean-vs-distorted smoke run, then expand to the other champions.
If `mps_available` is still `False`, expect this step to be much slower because both clean and distorted passes will run on CPU.

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench

python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets Dataset-1-mini \
  --detectors xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_mini.yaml \
  --respect-experiment-image-filters \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception
```

What this should do:

1. Evaluate clean `Dataset-1-mini`
2. Generate one distorted dataset with the same sample count
3. Evaluate `xception` on the distorted dataset
4. Write clean-vs-distorted deltas

Files to check:

- `/Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception/evaluation_dataset_json/`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception/combined_raw_runs.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception/detector_distortion_comparison.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception/detector_distortion_summary.csv`
- `/Users/Hao/thesis-project/training/results/distortion_smoke_dataset1mini_xception/distortion_champion_report.md`

Success criteria:

- `combined_raw_runs.csv` contains one clean row and one distorted row for `xception`
- `detector_distortion_comparison.csv` contains non-empty `clean_auc` and `delta_auc`
- `prediction_artifacts_path` and `metrics_artifacts_path` are populated

## Step 4: Expand detector coverage

After Step 2 succeeds, expand to the three chosen champions.
If `mps_available` is still `False`, keep using the small matched-subset setup first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench

python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --source-datasets Dataset-1-mini \
  --detectors mesonet mesoinception xception \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small.yaml \
  --clean-baseline-mode matched_subset \
  --respect-experiment-image-filters \
  --output-dir /Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched
```

## Step 5: Move to NVIDIA mini

Repeat Step 2 with:

- `--source-datasets NVIDIA-dataset-mini`

This verifies the same integration on the second dataset before any full-scale run.

## Recommended order

1. `champion_quickcheck.yaml`
2. `champion_smoke_small.yaml` with one detector
3. `champion_smoke_mini.yaml` with one detector
4. `champion_smoke_mini.yaml` with all three champions
5. `NVIDIA-dataset-mini`
