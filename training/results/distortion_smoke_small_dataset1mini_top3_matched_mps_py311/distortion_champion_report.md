# Distortion Champion Evaluation

- Generated: `2026-03-19T14:06:36.090332`
- Source datasets: `Dataset-1-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `gaussian_blur_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.525600 | 0.525600 | 0.027600 | 0.027600 | 0.031177 | 0.030000 | -0.020000 | 1 |
| `mesonet` | 0.695600 | 0.695600 | -0.031200 | -0.031200 | -0.020584 | NA | 0.020000 | 1 |
| `xception` | 0.620400 | 0.620400 | -0.061600 | -0.061600 | -0.006693 | NA | -0.020000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_top3_matched_mps_py311/benchmark_runs`
