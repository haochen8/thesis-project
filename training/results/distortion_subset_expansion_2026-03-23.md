# Distortion Subset Expansion Notes

Generated: `2026-03-23`

## Goal

Expand the subset-stage distortion coverage beyond Gaussian blur and verify that the additional distortion pipelines run correctly through the full detector evaluation path.

## Added Experiment Configs

New subset configs added under `/Users/Hao/thesis-project/distortionPipeline/configs/experiments`:

- `/Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small_jpeg.yaml`
- `/Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small_noise.yaml`
- `/Users/Hao/thesis-project/distortionPipeline/configs/experiments/champion_smoke_small_text_overlay.yaml`

Each uses:

- `variants: 1`
- `include_labels: ["roop_Real", "roop_Fake"]`
- `max_images_per_label: 50`

So every run remains a matched subset of `100` images total per source dataset.

## Distortions Checked

1. `jpeg_compress_v1`
2. `noise_v1`
3. `snapchat_text_overlay_v1`

## Datasets Checked

1. `Dataset-1-mini`
2. `NVIDIA-dataset-mini`

## Detector Set

The same three chosen champion detectors were reused:

- `mesonet`
- `mesoinception`
- `xception`

## Run Outcome

All expanded subset runs completed successfully.

Coverage:

- `3` distortions
- `2` datasets
- `3` detectors
- `2` conditions per detector (`clean_subset`, `distorted`)

Total:

- `36` successful detector evaluations
- all reported `device=mps`
- no orchestrator failures

## Result Folders

`Dataset-1-mini`:

- `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_dataset1mini_top3_matched_mps_py311`
- `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_dataset1mini_top3_matched_mps_py311`
- `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_dataset1mini_top3_matched_mps_py311`

`NVIDIA-dataset-mini`:

- `/Users/Hao/thesis-project/training/results/champion_smoke_small_jpeg_nvidia_mini_top3_matched_mps_py311`
- `/Users/Hao/thesis-project/training/results/champion_smoke_small_noise_nvidia_mini_top3_matched_mps_py311`
- `/Users/Hao/thesis-project/training/results/champion_smoke_small_text_overlay_nvidia_mini_top3_matched_mps_py311`

Each folder contains:

- `combined_raw_runs.csv`
- `detector_distortion_comparison.csv`
- `detector_distortion_summary.csv`
- `distortion_champion_report.md`
- per-run prediction and metric artifacts under `benchmark_runs/`

## Verification Summary

### `Dataset-1-mini`

#### JPEG

- `mesonet`: `delta_auc=0.0000`
- `mesoinception`: `delta_auc=+0.0332`
- `xception`: `delta_auc=-0.0072`

#### Noise

- `mesonet`: `delta_auc=+0.0656`
- `mesoinception`: `delta_auc=+0.1204`
- `xception`: `delta_auc=-0.2408`

#### Text Overlay

- `mesonet`: `delta_auc=-0.0280`
- `mesoinception`: `delta_auc=-0.0104`
- `xception`: `delta_auc=-0.1052`

### `NVIDIA-dataset-mini`

#### JPEG

- `mesonet`: `delta_auc=-0.0116`
- `mesoinception`: `delta_auc=-0.0036`
- `xception`: `delta_auc=+0.0044`

#### Noise

- `mesonet`: `delta_auc=+0.0208`
- `mesoinception`: `delta_auc=+0.0116`
- `xception`: `delta_auc=+0.0336`

#### Text Overlay

- `mesonet`: `delta_auc=+0.0820`
- `mesoinception`: `delta_auc=+0.0508`
- `xception`: `delta_auc=+0.0200`

## Reporting Bug Fixed During Verification

A summary-generation bug was found in `/Users/Hao/thesis-project/training/distortion_champion_evaluation.py`:

- exact `0.0` means were being written as blank strings in `detector_distortion_summary.csv`
- cause: `value or ""` logic treated zero as false

Fix applied:

- added `blank_if_none(...)`
- summary fields now preserve exact zero values

Affected summary/report files were regenerated for:

- the earlier Gaussian blur top-3 subset result folders
- all six new subset expansion result folders

## Interpretation

At this stage, the important conclusion is not which detector is globally best under these distortions.
The important conclusion is that:

1. `jpeg`, `noise`, and `text_overlay` all execute cleanly through the full integration path
2. the matched-subset protocol still works on both mini datasets
3. the detectors show distortion-dependent and dataset-dependent behavior, which justifies moving to the full-dataset distortion stage later
