# Distortion Champion Evaluation

- Generated: `2026-03-23T16:47:17.919591`
- Source datasets: `Dataset-1, NVIDIA-dataset`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `2`
- Comparison rows: `6`

## Distortion Recipes

- `noise_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.572853 | 0.488818 | 0.058623 | 0.021043 | 0.057270 | 0.038587 | -0.041900 | 2 |
| `mesonet` | 0.648664 | 0.528587 | 0.044080 | 0.009207 | 0.051712 | 0.000029 | -0.033300 | 2 |
| `xception` | 0.447048 | 0.406905 | -0.098784 | -0.254671 | -0.111619 | -0.049350 | 0.066750 | 2 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/final_full_noise_top3_prepare/benchmark_runs`
