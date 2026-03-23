# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.172547`
- Source datasets: `Dataset-1-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `snapchat_text_overlay_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.487600 | 0.487600 | -0.010400 | -0.010400 | -0.033228 | 0.000000 | 0.020000 | 1 |
| `mesonet` | 0.698800 | 0.698800 | -0.028000 | -0.028000 | 0.008328 | 0.060000 | 0.040000 | 1 |
| `xception` | 0.576800 | 0.576800 | -0.105200 | -0.105200 | -0.022754 | 0.070000 | 0.060000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311/benchmark_runs`
