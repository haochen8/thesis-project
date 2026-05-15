# Thesis Project: Robustness Evaluation for Deepfake Detectors

This repository contains the code and experiment artifacts for a thesis project on
deepfake detector robustness. It builds on the detector and evaluation structure of
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench), then adds project-specific
dataset manifests, staged detector selection, distortion generation, and clean-vs-distorted
evaluation workflows.

The main research workflow is:

1. Prepare custom DeepfakeBench-style dataset manifests.
2. Select compatible image-based detectors through staged benchmark runs.
3. Generate controlled image distortions such as JPEG compression, blur, noise, and text overlays.
4. Re-register distorted images as DeepfakeBench-compatible datasets.
5. Compare detector performance on clean and distorted inputs.

## Repository Layout

- `training/`: DeepfakeBench-based training, testing, detector, metric, and orchestration code.
- `training/spatial_champion_benchmark.py`: staged detector benchmark runner.
- `training/distortion_champion_evaluation.py`: end-to-end clean-vs-distorted evaluation runner.
- `training/results/`: experiment notes and generated result summaries.
- `preprocessing/`: dataset preprocessing utilities and dataset JSON manifests.
- `preprocessing/dataset_json/`: manifests for `Dataset-1`, `NVIDIA-dataset`, and staged subsets.
- `distortionPipeline/`: deterministic image-distortion pipeline and manifest bridge.
- `distortionPipeline/configs/recipes/`: distortion recipe definitions.
- `distortionPipeline/configs/experiments/`: experiment-level distortion configurations.
- `analysis/`: plotting and post-processing utilities.
- `datasets/`: local dataset metadata and placeholders. Raw datasets are not included.

## Project-Specific Additions

The project extends the upstream benchmark with:

- staged detector selection over custom datasets:
  - Stage A: small feasibility subsets
  - Stage B: medium-scale detector screening
  - Stage C: full-dataset evaluation of compatible detectors
- a champion scoring workflow for spatial/image-based detectors;
- structured export of benchmark artifacts, including per-sample predictions and metrics;
- a deterministic distortion pipeline with reusable recipes;
- a manifest adapter between distortion pipeline JSONL files and DeepfakeBench dataset JSON files;
- clean-vs-distorted comparison reports for selected champion detectors.

The final Stage C detector pool used compatible available checkpoints for:

- `xception`
- `mesonet`
- `mesoinception`
- `cnn-aug`
- `efficientnet-b4`
- `capsule`
- `ffd`
- `core`
- `recce`
- `ucf`

## Setup

The original DeepfakeBench environment targets Python 3.7-era dependencies. This project
also includes local notes and requirement snapshots under `training/results/` for later
Python environments used during the thesis experiments.

For the DeepfakeBench-style environment:

```bash
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh
```

For the distortion pipeline:

```bash
cd distortionPipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [distortionPipeline/README.md](distortionPipeline/README.md) for the full distortion
pipeline workflow. The distortion system is also maintained as a standalone project at
[litmas/distortionPipeline](https://github.com/litmas/distortionPipeline). It provides the
deterministic recipe-based image distortion tooling used here to generate controlled
robustness treatments such as compression, blur, noise, resizing, color filters, and
social-media-style text/UI overlays.

## Dataset Manifests

Dataset manifests live in `preprocessing/dataset_json/`. The code keeps short local
manifest names, but the source datasets are:

- `Dataset-1`: Kaggle
  [Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset)
- `NVIDIA-dataset`: Kaggle
  [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

The staged benchmark runner maps stages to manifest names:

- Stage A: `Dataset-1-mini`, `NVIDIA-dataset-mini`
- Stage B: `Dataset-1-stageB`, `NVIDIA-dataset-stageB`
- Stage C: `Dataset-1`, `NVIDIA-dataset`

Raw video/image datasets are not redistributed in this repository. Follow the terms of
the original dataset providers before downloading, preprocessing, or sharing data.

## Results Runs

This project currently documents two related result runs:

1. **First run: this repository.** The local thesis run evaluates `MesoNet`,
   `MesoInception`, and `Xception` on the project manifests for `Dataset-1`
   and `NVIDIA-dataset`. It uses Gaussian blur, JPEG compression, noise, and text overlay
   distortions generated through the local `distortionPipeline` integration. The main
   artifacts are in `training/results/final_full_thesis_matrix_top3`.
2. **Second run: standalone distortion pipeline repository.** The follow-up run is from
   [`litmas/distortionPipeline`](https://github.com/litmas/distortionPipeline/tree/main).
   It evaluates `UCF`, `SPSL`, and `F3Net` on Celeb-DF-v1 screening data plus full
   FaceForensics++ and DFDCP test sets, using Snapchat-, Instagram-, and TikTok-style
   social-media distortions. Its results package is published under
   [`results/`](https://github.com/litmas/distortionPipeline/tree/main/results) in that
   repository.

The two runs should be read side by side rather than merged into one leaderboard because
they use different datasets, detectors, and distortion families. A combined interpretation
is available in
[training/results/combined_results_summary.md](training/results/combined_results_summary.md).

## Running Detector Selection

Run a staged benchmark from the repository root:

```bash
python training/spatial_champion_benchmark.py \
  --stage C \
  --output-dir training/results/stage_c_existing_detectors
```

Useful options include:

- `--detectors`: restrict the run to selected detectors.
- `--datasets`: override the datasets selected by the stage.
- `--timeout-minutes`: set the per-detector timeout.
- `--export-test-artifacts`: export structured predictions and metrics from `training/test.py`.
- `--dataset-json-folder`: evaluate against a generated manifest folder instead of the default one.

For details on the completed staged runs, see
[training/results/stage_A_to_C_documentation.md](training/results/stage_A_to_C_documentation.md).

## Running Distortion Evaluation

The distortion workflow connects the project-specific pipeline to DeepfakeBench manifests:

```bash
python training/distortion_champion_evaluation.py \
  --champions-json training/results/stage_c_existing_detectors/champions.json \
  --distortion-root distortionPipeline \
  --experiment-yaml distortionPipeline/configs/experiments/exp.yaml
```

This writes artifacts such as:

- `dataset_index.csv`
- `generated_dataset_index.csv`
- `combined_raw_runs.csv`
- `detector_distortion_comparison.csv`
- `detector_distortion_summary.csv`
- `distortion_champion_report.md`

More implementation notes are in
[training/results/distortion_champion_integration_2026-03-18.md](training/results/distortion_champion_integration_2026-03-18.md).

The distortion generation component is based on
[litmas/distortionPipeline](https://github.com/litmas/distortionPipeline), a small
config-driven pipeline for converting image datasets into JSONL manifests, expanding
distortion recipes into reproducible jobs, applying the transformations, and writing
augmented manifests that can be bridged back into DeepfakeBench-style dataset JSON files.

## Attribution and Licensing

This repository is a research project built from and around
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench):

- DeepfakeBench paper: [A Comprehensive Benchmark of Deepfake Detection](https://arxiv.org/abs/2307.01426)
- Upstream repository: <https://github.com/SCLBD/DeepfakeBench>
- Upstream release weights: <https://github.com/SCLBD/DeepfakeBench/releases>

DeepfakeBench-derived files are attributed to the DeepfakeBench authors and remain subject
to the upstream license terms preserved in [LICENSE copy](LICENSE%20copy). Project-specific
additions are covered by the root [LICENSE](LICENSE) where they are separable from the
DeepfakeBench-derived material.

Datasets, pretrained weights, and third-party model components may have their own licenses
and redistribution rules. Check the original providers before use.

## Citation

If this repository or its DeepfakeBench-derived components are useful, cite the upstream
DeepfakeBench work:

```bibtex
@inproceedings{yan2023deepfakebench,
  title={DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```
