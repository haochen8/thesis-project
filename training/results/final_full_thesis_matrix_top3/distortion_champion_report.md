# Distortion Champion Evaluation

- Generated: `2026-03-24T15:11:17.071606`
- Source datasets: `Dataset-1, NVIDIA-dataset`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `8`
- Comparison rows: `24`

## Distortion Recipes

- `gaussian_blur_v1`
- `jpeg_compress_v1`
- `noise_v1`
- `snapchat_text_overlay_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesonet` | 0.618061 | 0.518126 | 0.013478 | -0.023914 | 0.022999 | 0.004797 | -0.010562 | 8 |
| `mesoinception` | 0.524606 | 0.453084 | 0.010376 | -0.029457 | 0.013190 | -0.001288 | -0.005125 | 8 |
| `xception` | 0.495138 | 0.406905 | -0.050693 | -0.254671 | -0.044223 | -0.011210 | 0.035600 | 8 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3/benchmark_runs`
