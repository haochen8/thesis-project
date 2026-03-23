# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.175971`
- Source datasets: `NVIDIA-dataset-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `snapchat_text_overlay_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesonet` | 0.600400 | 0.600400 | 0.082000 | 0.082000 | 0.074443 | 0.020000 | -0.100000 | 1 |
| `mesoinception` | 0.442800 | 0.442800 | 0.050800 | 0.050800 | 0.048069 | -0.040000 | 0.020000 | 1 |
| `xception` | 0.434000 | 0.434000 | 0.020000 | 0.020000 | 0.036530 | -0.030000 | 0.000000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311/benchmark_runs`
