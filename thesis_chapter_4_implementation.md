# 4 Research Project - Implementation

This chapter describes the implementation used to collect the experimental data for the study. The project was implemented as an evaluation pipeline around DeepfakeBench, with additional scripts for detector selection, distortion generation, result aggregation, and prediction export. The purpose of the implementation was to evaluate pretrained image-based deepfake detectors on two custom datasets and then evaluate the selected detectors under image distortions.

## 4.1 Implementation Overview

The implementation consisted of three main parts. First, the datasets were represented as DeepfakeBench-compatible JSON manifests. Second, a benchmark runner was implemented to execute pretrained detectors on selected datasets and collect metrics. Third, a distortion evaluation pipeline was implemented to generate distorted versions of the test data and rerun the selected detectors on those datasets.

The main scripts used for data collection were:

| Component | File | Purpose |
|---|---|---|
| Detector benchmark runner | `training/spatial_champion_benchmark.py` | Runs detectors on datasets, parses metrics, records status, and ranks detectors. |
| Detector test script | `training/test.py` | Loads one detector checkpoint, runs inference, computes metrics, and exports predictions. |
| Distortion orchestrator | `training/distortion_champion_evaluation.py` | Generates distorted datasets and evaluates selected detectors on clean and distorted conditions. |
| Distortion pipeline | `distortionPipeline/` | Applies deterministic image distortions from recipe files. |

Table 4.1: Main implemented components used for data collection.

## 4.2 Dataset Preparation

Two datasets were used in the final evaluation: `Dataset-1` and `NVIDIA-dataset`. Both datasets were converted into the JSON structure expected by DeepfakeBench. Each manifest stores the dataset name, class labels, split names, sample identifiers, labels, and frame paths.

`Dataset-1` contains two labels: `roop_Real` and `roop_Fake`. The full manifest contains 5000 real and 4630 fake samples in each of the train, validation, and test splits. `NVIDIA-dataset` also contains `roop_Real` and `roop_Fake`. The full manifest contains 50000 real and 50000 fake training samples, 10000 real and 10000 fake validation samples, and 10000 real and 10000 fake test samples.

Subset manifests were also created for staged execution. Stage A used mini subsets with 100 train, 100 validation, and 500 test samples per class. Stage B used larger subsets with 100 train and 100 validation samples per class. The Stage B test split used 2500 samples per class for `Dataset-1-stageB` and 5000 samples per class for `NVIDIA-dataset-stageB`. Stage C used the full `Dataset-1` and `NVIDIA-dataset` manifests.

The staged datasets were used to reduce the cost of early debugging and to avoid running the full experiment before the detector configurations, checkpoint paths, and inference outputs had been verified.

## 4.3 Detector Benchmark Implementation

The detector benchmark was implemented in `training/spatial_champion_benchmark.py`. The script wraps the DeepfakeBench test script and runs it detector by detector and dataset by dataset. For each detector run, the benchmark runner builds a command with the detector configuration file, checkpoint path, and test dataset name. The script records the command output, runtime, device, checkpoint source, status, and parsed metrics.

The detector pool was limited to image-based detectors. Video detectors were excluded because the datasets and distortion pipeline were image-based. The original detector list included naive and spatial detectors, but the final Stage C run was restricted to detectors with available compatible pretrained checkpoints. This produced the following Stage C detector set: Xception, MesoNet, MesoInception, CNN-Aug, EfficientNet-B4, Capsule, FFD, CORE, RECCE, and UCF.

The benchmark runner used four run statuses:

| Status | Meaning |
|---|---|
| `success` | The test process exited successfully and the required metrics were parsed. |
| `missing_weight` | No compatible checkpoint was available for the detector. |
| `failed` | The process returned a non-zero exit code or did not produce required metrics. |
| `timeout` | The process exceeded the configured timeout. |

Table 4.2: Run statuses used by the benchmark runner.

For each successful run, the benchmark collected ACC, AUC, EER, AP, video AUC, and runtime. Aggregated detector rows were created by averaging metrics across datasets and calculating the absolute AUC gap between `Dataset-1` and `NVIDIA-dataset`. The ranking score was implemented as:

`score = 0.55*AUC + 0.20*AP + 0.10*ACC - 0.15*EER - 0.08*runtime_norm - 0.12*auc_gap`

where `runtime_norm` is the min-max normalized runtime among complete detectors. This score was used only to rank the evaluated detectors and select the top three detectors for the distortion evaluation.

## 4.4 Staged Evaluation Procedure

The detector evaluation was carried out in three stages. Stage A was a feasibility run on the mini manifests. The purpose was to verify that the dataset manifests, detector configuration files, checkpoint loading, metric parsing, and result writing worked together.

Stage B used larger subset manifests and was used to identify detector failures before the full run. During this stage, missing checkpoints and missing backbone pretrained files were separated from implementation errors. One detector-specific inference issue was found in UCF, where the test path expected a `prob` output. The UCF detector implementation was adjusted so that inference returned the expected `prob` value.

Stage C was the final full-dataset detector evaluation. It used only detectors that could be evaluated end to end with available compatible checkpoints. The Stage C output directory contains the raw detector-dataset runs, detector summary, champion detector JSON, leaderboard, run configuration, and notes about excluded detectors.

## 4.5 Prediction and Metric Export

The original DeepfakeBench test flow was extended in `training/test.py` to support artifact export. When `--export_artifacts_dir` is provided, the script writes one `predictions.csv` and one `metrics.json` file for each evaluated dataset.

Each exported prediction row contains the sample index, image reference, ground-truth label, predicted probability, and thresholded prediction. The metrics JSON stores the dataset name, number of samples, and the computed metrics. These exported artifacts were used later to calculate real-class accuracy, fake-class accuracy, and balanced accuracy for clean and distorted conditions.

## 4.6 Distortion Pipeline Design

The distortion part of the project was implemented as a deterministic image-processing pipeline in `distortionPipeline/`. The pipeline uses JSON recipe files and an experiment YAML file to define which distortions are applied. It converts source dataset manifests into JSONL records, expands the configured recipes into job manifests, applies each distortion, and writes new image outputs and augmented manifests.

The full thesis distortion matrix used one variant per image and a global seed of `12345`. Four distortion recipes were used:

| Recipe | Parameters |
|---|---|
| `gaussian_blur_v1` | Gaussian blur with sigma 1.2. |
| `jpeg_compress_v1` | JPEG recompression with quality 60. |
| `noise_v1` | Gaussian noise with standard deviation 10.0. |
| `snapchat_text_overlay_v1` | Centered text overlay with random words, black stroke, and semi-transparent full-width background. |

Table 4.3: Distortion recipes used in the final distortion evaluation.

The pipeline was implemented to be reproducible. Seeds are derived from the global seed, image identifier, recipe identifier, and variant. The cache key is based on the source path, normalized distortion steps, seed, and variant. If an output already exists in the cache, it can be reused instead of regenerated.

## 4.7 Distortion Evaluation Implementation

The distortion evaluation was implemented in `training/distortion_champion_evaluation.py`. The script reads the champion detector list from the Stage C output, prepares distorted versions of the selected datasets, registers the generated manifests as DeepfakeBench datasets, and then runs the detector benchmark on both clean and distorted conditions.

The distortion orchestrator performs the following steps:

1. Read the selected detectors and source datasets from `champions.json`.
2. Convert each source dataset test split into JSONL format.
3. Generate distortion jobs from the selected recipes.
4. Apply the distortions and write distorted image outputs.
5. Convert the augmented manifests back into DeepfakeBench dataset JSON files.
6. Copy clean and distorted manifests into the evaluation dataset folder.
7. Run the benchmark runner on each clean and distorted dataset.
8. Write combined raw runs, clean-versus-distorted comparisons, detector summaries, and a Markdown report.

This design reused the same benchmark runner for both clean and distorted datasets. As a result, clean and distorted runs used the same detector configurations, checkpoint resolution logic, metric parsing, and artifact export behavior.

## 4.8 Collected Artifacts

The implementation wrote all collected data to result folders under `training/results/`. The Stage C detector evaluation wrote `raw_runs.csv`, `detector_summary.csv`, `champions.json`, `leaderboard.md`, and `run_config.json`. The final distortion matrix wrote `dataset_index.csv`, `combined_raw_runs.csv`, `detector_distortion_comparison.csv`, `detector_distortion_summary.csv`, generated dataset manifests, evaluation manifests, benchmark logs, prediction CSV files, and metric JSON files.

These artifacts form the data basis for the results presented in Chapter 5. The raw run files provide detector-level metrics, the comparison files provide clean-versus-distorted metric differences, and the exported prediction files provide the class-level breakdown between real and fake samples.

