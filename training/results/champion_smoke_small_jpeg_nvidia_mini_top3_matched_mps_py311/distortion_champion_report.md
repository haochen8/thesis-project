# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.173697`
- Source datasets: `NVIDIA-dataset-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `jpeg_compress_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xception` | 0.418400 | 0.418400 | 0.004400 | 0.004400 | -0.002564 | 0.010000 | -0.020000 | 1 |
| `mesoinception` | 0.388400 | 0.388400 | -0.003600 | -0.003600 | -0.009138 | -0.100000 | 0.060000 | 1 |
| `mesonet` | 0.506800 | 0.506800 | -0.011600 | -0.011600 | -0.011222 | 0.000000 | 0.000000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311/benchmark_runs`
