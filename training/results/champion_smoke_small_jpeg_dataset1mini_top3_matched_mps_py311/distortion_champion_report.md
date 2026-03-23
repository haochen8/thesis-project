# Distortion Champion Evaluation

- Generated: `2026-03-23T13:45:01.170145`
- Source datasets: `Dataset-1-mini`
- Champion detectors: `mesonet, mesoinception, xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `3`

## Distortion Recipes

- `jpeg_compress_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mesoinception` | 0.531200 | 0.531200 | 0.033200 | 0.033200 | -0.001208 | 0.010000 | -0.020000 | 1 |
| `mesonet` | 0.726800 | 0.726800 | 0.000000 | 0.000000 | -0.002817 | 0.000000 | -0.020000 | 1 |
| `xception` | 0.674800 | 0.674800 | -0.007200 | -0.007200 | 0.030242 | 0.060000 | 0.000000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311/benchmark_runs`
