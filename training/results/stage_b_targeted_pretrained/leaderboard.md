# Spatial Detector Champion Report

- Generated: `2026-02-23T18:05:00.556598`
- Datasets: `Dataset-1-stageB, NVIDIA-dataset-stageB`
- Selected detectors: `4`
- Ranked complete detectors: `3`
- Output dir: `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained`

## Top 3 Champions

| Rank | Detector | Family | Score | AUC(mean) | AP(mean) | EER(mean) | ACC(mean) | AUC gap | Runtime(s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | FFD (`ffd`) | spatial | 0.358004 | 0.548702 | 0.542384 | 0.465500 | 0.516850 | 0.219279 | 452.639 |
| 2 | EfficientNet-B4 (`efficientnet-b4`) | naive | 0.235408 | 0.446061 | 0.506367 | 0.537600 | 0.505000 | 0.008826 | 531.854 |
| 3 | CORE (`core`) | spatial | 0.219725 | 0.337991 | 0.429446 | 0.624400 | 0.480550 | 0.053784 | 444.074 |

## Failed or Missing Runs

| Detector | Dataset | Status | Detail | Log |
|---|---|---|---|---|
| `ucf` | `Dataset-1-stageB` | `failed` | `nonzero_exit_or_metrics_missing` | `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/logs/ucf__Dataset-1-stageB.log` |
| `ucf` | `NVIDIA-dataset-stageB` | `failed` | `nonzero_exit_or_metrics_missing` | `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/logs/ucf__NVIDIA-dataset-stageB.log` |

## Artifacts

- Raw run table: `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/raw_runs.csv`
- Detector summary: `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/detector_summary.csv`
- Champion JSON: `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/champions.json`
- Run config: `/Users/Hao/thesis-project/training/results/stage_b_targeted_pretrained/run_config.json`
