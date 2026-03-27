# Distortion Champion Evaluation

- Generated: `2026-03-23T16:04:39.211955`
- Source datasets: `Dataset-1, NVIDIA-dataset`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `2`
- Comparison rows: `6`

## Distortion Recipes

- `jpeg_compress_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.522407 | 0.462916 | 0.008177 | -0.004859 | 0.008449 | 0.003136 | -0.005550 | 2 |
| `mesonet` | 0.610724 | 0.518126 | 0.006140 | -0.001255 | 0.004608 | 0.000698 | -0.005300 | 2 |
| `xception` | 0.482077 | 0.416176 | -0.063754 | -0.113597 | -0.041010 | -0.008390 | 0.046050 | 2 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/final_full_jpeg_top3/benchmark_runs`
