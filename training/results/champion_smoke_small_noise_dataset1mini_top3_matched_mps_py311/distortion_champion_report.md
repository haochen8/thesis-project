# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.171389`
- Source datasets: `Dataset-1-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `noise_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.618400 | 0.618400 | 0.120400 | 0.120400 | 0.085117 | 0.160000 | -0.100000 | 1 |
| `mesonet` | 0.792400 | 0.792400 | 0.065600 | 0.065600 | 0.119142 | 0.000000 | 0.000000 | 1 |
| `xception` | 0.441200 | 0.441200 | -0.240800 | -0.240800 | -0.197489 | -0.090000 | 0.140000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311/benchmark_runs`
