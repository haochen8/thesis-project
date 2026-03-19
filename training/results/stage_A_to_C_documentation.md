# Stage A to Stage C Documentation (DeepfakeBench on Custom Datasets)

## 1. Purpose and Scope

This document describes how Stage A, Stage B, and Stage C were executed in this project for detector evaluation on custom datasets (`Dataset-1`, `NVIDIA-dataset`) using the benchmark runner:

- `/Users/Hao/thesis-project/training/spatial_champion_benchmark.py`

The target detector pool was image-based detectors only (no video detectors), with final Stage C restricted to detectors that had available compatible checkpoints.

---

## 2. Benchmark Runner Behavior

The runner executes detector evaluation with `training/test.py` detector-by-detector and dataset-by-dataset, then aggregates metrics.

### 2.1 Per-run execution
- Builds command:
  - `python training/test.py --detector_path <config> --weights_path <ckpt> --test_dataset <dataset_name>`
- Parses test output metrics:
  - `acc`, `auc`, `eer`, `ap`, `video_auc`
- Records runtime in seconds, status, and log path.

### 2.2 Status logic
- `success`: exit code `0` and `auc/ap/eer` all parsed.
- `missing_weight`: checkpoint not found.
- `failed`: non-zero exit or missing required metrics.
- `timeout`: subprocess exceeded timeout.

### 2.3 Aggregation logic
For each detector across datasets:
- `acc_mean`, `auc_mean`, `eer_mean`, `ap_mean`
- `auc_gap = |auc_dataset1 - auc_dataset2|`
- `runtime_total_sec`

Champion score formula:

`score = 0.55*AUC + 0.20*AP + 0.10*ACC - 0.15*EER - 0.08*runtime_norm - 0.12*auc_gap`

where `runtime_norm` is min-max normalized runtime among complete detectors.

---

## 3. Dataset Manifests and Stage Mapping

Stage mapping in runner:
- Stage A -> `Dataset-1-mini`, `NVIDIA-dataset-mini`
- Stage B -> `Dataset-1-stageB`, `NVIDIA-dataset-stageB`
- Stage C -> `Dataset-1`, `NVIDIA-dataset`

Manifest files:
- `/Users/Hao/thesis-project/preprocessing/dataset_json/Dataset-1-mini.json`
- `/Users/Hao/thesis-project/preprocessing/dataset_json/NVIDIA-dataset-mini.json`
- `/Users/Hao/thesis-project/preprocessing/dataset_json/Dataset-1-stageB.json`
- `/Users/Hao/thesis-project/preprocessing/dataset_json/NVIDIA-dataset-stageB.json`
- `/Users/Hao/thesis-project/preprocessing/dataset_json/Dataset-1.json`
- `/Users/Hao/thesis-project/preprocessing/dataset_json/NVIDIA-dataset.json`

Subset generator:
- `/Users/Hao/thesis-project/preprocessing/dataset_json/make_subset_manifest.py`

Key subset sizes used:
- `Dataset-1-mini`: per class `train=100, val=100, test=500`
- `NVIDIA-dataset-mini`: per class `train=100, val=100, test=500`
- `Dataset-1-stageB`: per class `train=100, val=100, test=2500`
- `NVIDIA-dataset-stageB`: per class `train=100, val=100, test=5000`

---

## 4. Stage A (Feasibility Gate)

### 4.1 Objective
Quickly verify pipeline viability on mini subsets before larger runs.

### 4.2 Run artifact directory
- `/Users/Hao/thesis-project/training/results/stage_a_spatial`

### 4.3 Effective config
- Generated at: `2026-02-23T16:33:27.590576`
- Datasets: `Dataset-1-mini`, `NVIDIA-dataset-mini`
- Selected detectors: `24` (naive + spatial pool)
- Timeout per run: `20 min`

### 4.4 Outcome
- Raw runs: `48`
- Status counts:
  - `success`: `2`
  - `missing_weight`: `46`
- Complete ranked detectors: `1`
- Stage A top detector: `xception`

Interpretation:
- The evaluation stack worked.
- Most detectors were blocked by unavailable checkpoints at this stage.

---

## 5. Stage B (Medium-Scale Selection)

### 5.1 Objective
Evaluate candidate pool on larger subsets to shortlist detectors robustly and identify blockers.

### 5.2 Base Stage B run
Artifact directory:
- `/Users/Hao/thesis-project/training/results/stage_b_spatial`

Config:
- Generated at: `2026-02-23T16:52:17.380128`
- Datasets: `Dataset-1-stageB`, `NVIDIA-dataset-stageB`
- Selected detectors: `24`
- Timeout per run: `20 min`

Base outcome:
- Raw runs: `48`
- Status:
  - `success`: `12`
  - `failed`: `8`
  - `missing_weight`: `28`
- Ranked complete detectors: `6`
- Base top-3: `mesonet`, `xception`, `mesoinception`

### 5.3 Failure analysis and remediation
Main blockers in Stage B base run:

1. Missing backbone pretrains (not detector checkpoints):
- Missing files such as:
  - `xception-b5690688.pth`
  - `efficientnet-b4-6ed6700e.pth`

2. UCF inference output mismatch:
- `KeyError: 'prob'` in test path.
- Fixed in:
  - `/Users/Hao/thesis-project/training/detectors/ucf_detector.py`
  - In inference mode, `forward()` now returns `'prob'`.

3. Detector checkpoint availability limits:
- Several spatial detectors had no compatible released checkpoints and remained `missing_weight`.

### 5.4 Targeted reruns after fixes

#### 5.4.1 Targeted detector rerun
Artifact:
- `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained`

Scope:
- `efficientnet-b4`, `ffd`, `core`, `ucf`

Outcome:
- Raw runs: `8`
- Status:
  - `success`: `6`
  - `failed`: `2` (UCF before patch)

#### 5.4.2 UCF-only verification after patch
Artifact:
- `/Users/Hao/thesis-project/training/results/stage_b_ucf_fix`

Outcome:
- Raw runs: `2`
- Status:
  - `success`: `2`

### 5.5 Stage B merged final view
Merged artifact:
- `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix`

Final Stage B outcome:
- Raw runs: `48`
- Status:
  - `success`: `20`
  - `missing_weight`: `28`
- Ranked complete detectors: `10`
- Top-3:
  - `mesonet`
  - `xception`
  - `mesoinception`

---

## 6. Stage C (Full Dataset Final Evaluation)

### 6.1 Objective
Run final full-dataset evaluation with detectors that had compatible available checkpoints.

### 6.2 Why detector subset was restricted
By this point, known `missing_weight` detectors were excluded from execution scope to avoid invalid comparisons.

Stage C detector list (`10`):
- `xception`
- `mesonet`
- `mesoinception`
- `cnn-aug`
- `efficientnet-b4`
- `capsule`
- `ffd`
- `core`
- `recce`
- `ucf`

### 6.3 Stage C run
Artifact directory:
- `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors`

Config:
- Generated at: `2026-02-27T17:16:13.758423`
- Stage: `C`
- Datasets: `Dataset-1`, `NVIDIA-dataset`
- Selected detectors: `10`
- Timeout per run: `45 min`

### 6.4 Stage C outcome
- Raw runs: `20`
- Status:
  - `success`: `20`
- Ranked complete detectors: `10`
- Top-3 champions:
  1. `mesonet`
  2. `mesoinception`
  3. `xception`

### 6.5 Final documentation cleanup for Stage C
Leaderboard and metadata were updated to explicitly reflect tested-only policy and excluded detectors:

- Updated:
  - `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/leaderboard.md`
  - `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/champions.json`
  - `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/run_config.json`
- Added:
  - `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/excluded_missing_weights.md`

Excluded spatial detectors due to missing weights (`14`):
- `dsp-fwa`, `face-xray`, `local-relation`, `iid`, `rfm`, `sia`, `sladd`, `uia-vit`, `clip`, `sbi`, `pcl-i2g`, `multi-attention`, `lsda`, `effort`

---

## 7. Key Engineering Decisions Across Stages

1. Stage-gated execution:
- Stage A: quick feasibility
- Stage B: medium-scale screening + remediation
- Stage C: full-scale final evaluation

2. Strict artifact traceability:
- Every stage wrote `run_config.json`, `raw_runs.csv`, `detector_summary.csv`, `champions.json`, `leaderboard.md`.

3. Separation of blocking causes:
- Missing detector checkpoints vs runtime/code bugs vs missing backbone pretrains were treated separately.

4. Fair final comparison:
- Stage C final ranking used only detectors that could actually be evaluated end-to-end with compatible weights.

---

## 8. Reproducibility Pointers

Core runner:
- `/Users/Hao/thesis-project/training/spatial_champion_benchmark.py`

Primary result folders:
- Stage A:
  - `/Users/Hao/thesis-project/training/results/stage_a_spatial`
- Stage B merged:
  - `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix`
- Stage C final:
  - `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors`

Use each folder's `run_config.json` + `raw_runs.csv` as canonical provenance for the reported results.

