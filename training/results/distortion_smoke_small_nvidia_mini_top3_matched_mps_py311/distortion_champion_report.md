# Distortion Champion Evaluation

- Generated: `2026-03-19T15:05:20.800768`
- Source datasets: `NVIDIA-dataset-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `gaussian_blur_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesonet` | 0.577600 | 0.577600 | 0.059200 | 0.059200 | 0.070831 | NA | -0.040000 | 1 |
| `xception` | 0.353200 | 0.353200 | -0.060800 | -0.060800 | -0.066190 | -0.080000 | 0.080000 | 1 |
| `mesoinception` | 0.305600 | 0.305600 | -0.086400 | -0.086400 | -0.042645 | -0.170000 | 0.140000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_nvidia_mini_top3_matched_mps_py311/benchmark_runs`
