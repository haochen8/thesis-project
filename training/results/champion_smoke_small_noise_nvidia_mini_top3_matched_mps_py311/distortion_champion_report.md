# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.174844`
- Source datasets: `NVIDIA-dataset-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `noise_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xception` | 0.447600 | 0.447600 | 0.033600 | 0.033600 | -0.000375 | -0.020000 | 0.000000 | 1 |
| `mesonet` | 0.539200 | 0.539200 | 0.020800 | 0.020800 | 0.036672 | 0.000000 | -0.020000 | 1 |
| `mesoinception` | 0.403600 | 0.403600 | 0.011600 | 0.011600 | 0.027377 | -0.030000 | 0.000000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311/benchmark_runs`
