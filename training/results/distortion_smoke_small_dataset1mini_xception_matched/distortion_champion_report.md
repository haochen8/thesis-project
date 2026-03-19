# Distortion Champion Evaluation

- Generated: `2026-03-19T13:11:29.425094`
- Source datasets: `Dataset-1-mini`
- Champion detectors: `xception`
- Distorted dataset manifests generated: `1`
- Comparison rows: `1`

## Distortion Recipes

- `gaussian_blur_v1`

## Detector Robustness Summary

| Detector | Mean distorted AUC | Worst distorted AUC | Mean ΔAUC | Worst ΔAUC | Mean ΔAP | Mean ΔACC | Mean ΔEER | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xception` | 0.620400 | 0.620400 | -0.061600 | -0.061600 | -0.006693 | NA | -0.020000 | 1 |

## Artifacts

- Dataset index: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/dataset_index.csv`
- Combined raw runs: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/combined_raw_runs.csv`
- Distortion comparison: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/detector_distortion_comparison.csv`
- Distortion detector summary: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/detector_distortion_summary.csv`
- Generated dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/generated_dataset_json`
- Evaluation dataset JSON folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/evaluation_dataset_json`
- Benchmark runs folder: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched/benchmark_runs`
