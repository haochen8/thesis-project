# Distortion Champion Evaluation

- Generated: `2026-03-23T14:55:19.475563`
- Source datasets: `Dataset-1, NVIDIA-dataset`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `2`
- Comparison rows: `6`

## Distortion Recipes

- `gaussian_blur_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesonet` | 0.619964 | 0.537880 | 0.015380 | 0.012261 | 0.022983 | -0.001208 | -0.013250 | 2 |
| `mesoinception` | 0.492156 | 0.453084 | -0.022074 | -0.029457 | -0.008828 | -0.039131 | 0.018350 | 2 |
| `xception` | 0.505602 | 0.423298 | -0.040229 | -0.073669 | -0.042254 | -0.018340 | 0.018600 | 2 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/final_full_gaussian_blur_top3/benchmark_runs`
