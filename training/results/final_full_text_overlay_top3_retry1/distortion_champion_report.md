# Distortion Champion Evaluation

- Generated: `2026-03-24T12:32:33.931856`
- Source datasets: `Dataset-1, NVIDIA-dataset`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `2`
- Comparison rows: `6`

## Distortion Recipes

- `snapchat_text_overlay_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xception` | 0.545826 | 0.466550 | -0.000006 | -0.036473 | 0.017991 | 0.031238 | 0.011000 | 2 |
| `mesoinception` | 0.511009 | 0.486646 | -0.003221 | -0.025312 | -0.004133 | -0.007746 | 0.008600 | 2 |
| `mesonet` | 0.592893 | 0.519914 | -0.011690 | -0.023914 | 0.012694 | 0.019669 | 0.009600 | 2 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/final_full_text_overlay_top3_retry1/benchmark_runs`
