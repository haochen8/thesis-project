# Distortion Champion Integration Notes

Generated: `2026-03-18`

## Goal

Integrate the distortion treatment pipeline with the chosen champion detectors without coupling detector code to the distortion generator.

Implemented architecture:

1. DeepfakeBench dataset manifest -> distortion pipeline JSONL
2. distortion pipeline job generation and image writing
3. augmented distortion manifest -> DeepfakeBench dataset manifests
4. standardized detector evaluation via `training/spatial_champion_benchmark.py`
5. clean-vs-distorted comparison artifacts

## Why Integration Work Was Needed

Before these changes, the two subsystems did not connect cleanly:

1. The distortion pipeline produced flat JSONL manifests plus distorted image paths.
2. DeepfakeBench testing expected nested dataset JSON files in `preprocessing/dataset_json` style.
3. The benchmark runner only knew how to call `training/test.py` against dataset names that already existed in that manifest folder.
4. `training/test.py` printed aggregate metrics to stdout, but did not export structured per-sample outputs for later robustness analysis.

That meant the project had three missing pieces:

1. A manifest bridge between the distortion pipeline and DeepfakeBench dataset format.
2. A standardized evaluation layer that could consume generated treatment datasets without hand-editing dataset manifests each time.
3. Structured result exports so distorted-vs-clean comparisons could be analyzed later without scraping logs.

The files added or modified below exist to solve one of those three gaps.

## Files Changed

### Distortion bridge

- `/Users/Hao/thesis-project/distortionPipeline/src/pipeline/manifest_adapter.py`
  - why it had to be added:
    - the distortion pipeline and DeepfakeBench use different manifest formats
    - without a dedicated adapter, every experiment would require manual JSON rewriting
    - the integration needed a stable round-trip layer that preserved sample ids, labels, split membership, and metadata
  - added DeepfakeBench JSON -> pipeline record flattening
  - added pipeline JSONL -> DeepfakeBench JSON reconstruction
  - added grouped augmented-manifest -> per-treatment dataset JSON generation
  - added compact dataset index CSV writer

- `/Users/Hao/thesis-project/distortionPipeline/scripts/generate_manifest.py`
  - why it had to be modified:
    - converting into pipeline JSONL is not enough if source identifiers are lost during job generation
    - the augmented manifest needs to retain the original sample identity so distorted outputs can be reassembled into the correct DeepfakeBench sample structure
  - now preserves source identifiers needed for round-trip reconstruction:
    - `source_dataset_name`
    - `source_label`
    - `source_split`
    - `source_sample_id`
    - `source_frame_index`
    - `source_frame_count`
    - `source_frame_path`
  - continues to preserve extra metadata in `source_metadata`

- `/Users/Hao/thesis-project/distortionPipeline/scripts/deepfakebench_manifest.py`
  - why it had to be added:
    - the adapter logic needed a usable CLI entry point
    - integration would be brittle if the only way to run conversions was by importing internal Python modules manually
    - this script makes the bridge reproducible from the terminal and suitable for documentation
  - added `to-distorted-json` subcommand for augmented-manifest -> DeepfakeBench dataset JSON conversion

- `/Users/Hao/thesis-project/distortionPipeline/README.md`
  - why it had to be modified:
    - new bridge commands and orchestration entry points must be documented where the distortion pipeline is already described
    - otherwise the added integration layer would exist in code but not in the operational workflow
  - documented the new bridge commands
  - documented the end-to-end champion evaluation entry point

### Distortion tests

- `/Users/Hao/thesis-project/distortionPipeline/tests/test_manifest_adapter.py`
  - why it had to be added:
    - manifest conversion is a structural integration point and easy to break silently
    - a roundtrip test is the quickest way to catch dropped metadata, wrong frame order, or malformed reconstructed dataset JSON
  - roundtrip DeepfakeBench <-> pipeline conversion
  - grouped augmented-manifest -> dataset JSON generation

- `/Users/Hao/thesis-project/distortionPipeline/tests/test_generate_manifest_metadata.py`
  - why it had to be added:
    - preserving metadata through `build_jobs()` is required for the reverse conversion step
    - if those fields disappear here, the end-to-end bridge fails later even though distortion generation itself still runs
  - verifies that `build_jobs()` carries source identifiers and metadata forward

### Evaluation layer

- `/Users/Hao/thesis-project/training/test.py`
  - why it had to be modified:
    - the old version only printed summary metrics to stdout
    - robustness experiments need structured outputs per dataset and per sample
    - the script also needed a configurable dataset manifest folder so generated distortion datasets could be evaluated without overwriting the clean manifest directory
  - added `--export_artifacts_dir`
  - added `--dataset_json_folder`
  - writes per-dataset:
    - `predictions.csv`
    - `metrics.json`
  - default stdout behavior remains unchanged

- `/Users/Hao/thesis-project/training/spatial_champion_benchmark.py`
  - why it had to be modified:
    - the benchmark already handled detector orchestration well, so replacing it would have duplicated working logic
    - instead, it needed small extensions so it could:
      - request structured exports from `training/test.py`
      - point `training/test.py` at a generated evaluation manifest folder instead of only the default clean manifest folder
  - added `--export-test-artifacts`
  - added `--dataset-json-folder`
  - records artifact paths in `raw_runs.csv`

- `/Users/Hao/thesis-project/training/distortion_champion_evaluation.py`
  - why it had to be added:
    - there was no single script that connected treatment generation, manifest registration, detector execution, and comparison artifacts
    - running these steps manually for every distortion recipe would be slow, error-prone, and hard to reproduce
    - the project needed one orchestrator that defines the full experiment workflow in the correct order
  - new orchestration script
  - resolves champion detectors from `champions.json` by default
  - exports source manifests into distortion JSONL
  - runs distortion generation
  - registers generated treatments as DeepfakeBench datasets
  - creates a merged evaluation manifest folder so clean and distorted datasets can be benchmarked through the same interface
  - runs `spatial_champion_benchmark.py` once per dataset
  - writes:
    - `dataset_index.csv`
    - `generated_dataset_index.csv`
    - `combined_raw_runs.csv`
    - `detector_distortion_comparison.csv`
    - `detector_distortion_summary.csv`
    - `distortion_champion_report.md`

## Why It Is Structured This Way

- The distortion pipeline remains a treatment generator.
- The detector benchmark remains the standardized evaluator.
- The bridge sits at the manifest layer, which is the stable interface both systems share.
- Running the benchmark once per dataset avoids misusing the old two-dataset ranking formula across many distortion variants.
- Small patches were preferred over a rewrite because the existing Stage A-C benchmark already worked and had verified detector-weight resolution, timeout handling, and artifact generation.
- The merged evaluation manifest folder was necessary because generated distortion datasets should be evaluated alongside clean datasets without overwriting `preprocessing/dataset_json`.
- Structured exports were added at `training/test.py` level because that is where sample predictions already exist; doing this only at the benchmark wrapper level would have forced more fragile log parsing.

## New Commands

### DeepfakeBench JSON -> pipeline JSONL

```bash
python -m scripts.deepfakebench_manifest to-jsonl \
  --input ../preprocessing/dataset_json/Dataset-1.json \
  --output manifests/dataset_1_test.jsonl \
  --dataset_name Dataset-1
```

### Augmented manifest -> per-treatment DeepfakeBench dataset JSON

```bash
python -m scripts.deepfakebench_manifest to-distorted-json \
  --input manifests/jobs_with_paths.jsonl \
  --output_dir ../training/results/generated_dataset_json \
  --index_csv ../training/results/generated_dataset_index.csv
```

### End-to-end run

```bash
python training/distortion_champion_evaluation.py \
  --champions-json training/results/stage_c_existing_detectors/champions.json \
  --distortion-root distortionPipeline \
  --experiment-yaml distortionPipeline/configs/experiments/exp.yaml
```

## Verification

Completed:

- `python3 -m py_compile /Users/Hao/thesis-project/distortionPipeline/src/pipeline/manifest_adapter.py /Users/Hao/thesis-project/distortionPipeline/scripts/deepfakebench_manifest.py /Users/Hao/thesis-project/distortionPipeline/scripts/generate_manifest.py /Users/Hao/thesis-project/training/test.py /Users/Hao/thesis-project/training/spatial_champion_benchmark.py /Users/Hao/thesis-project/training/distortion_champion_evaluation.py`
- adapter smoke test in the `DeepfakeBench` conda env:
  - DeepfakeBench sample manifest -> pipeline records
  - `build_jobs()` metadata propagation
  - augmented manifest -> generated DeepfakeBench dataset JSON
- dry run of:

```bash
python /Users/Hao/thesis-project/training/distortion_champion_evaluation.py \
  --dry-run \
  --source-datasets Dataset-1 \
  --detectors mesonet \
  --distortion-root /Users/Hao/thesis-project/distortionPipeline \
  --experiment-yaml /Users/Hao/thesis-project/distortionPipeline/configs/experiments/exp.yaml \
  --output-dir /tmp/distortion_champion_eval_dryrun_final
```

Dry-run artifacts written:

- `/tmp/distortion_champion_eval_dryrun_final/run_config.json`
- `/tmp/distortion_champion_eval_dryrun_final/dataset_index.csv`
- `/tmp/distortion_champion_eval_dryrun_final/distortion_champion_report.md`
- `/tmp/distortion_champion_eval_dryrun_final/evaluation_dataset_json/Dataset-1.json`
- `/tmp/distortion_champion_eval_dryrun_final/orchestrator_logs/generate_manifest__Dataset-1.log`
- `/tmp/distortion_champion_eval_dryrun_final/orchestrator_logs/run_distortions__Dataset-1.log`
- `/tmp/distortion_champion_eval_dryrun_final/orchestrator_logs/benchmark__Dataset-1.log`

Not available in the current interpreters:

- `python3 -m pytest ...` failed because `pytest` is not installed in `/usr/local/opt/python@3.14/bin/python3.14`
- `python -m pytest ...` inside the `DeepfakeBench` conda env also failed because `pytest` is not installed there

## Notes

- The current `distortionPipeline/configs/experiments/exp.yaml` was sample-oriented (`include_labels`, `max_images_per_label`). The orchestration script normalizes those image filters away by default unless `--respect-experiment-image-filters` is used.
- The benchmark artifact export is opt-in at the benchmark level but defaults to enabled in the new distortion orchestration script.

## 2026-03-19 Continuation

### Why Another Integration Change Was Needed

- The first reduced smoke configuration (`champion_smoke_small.yaml`) made distortion generation fast, but it exposed a methodological problem in the orchestration layer: the distorted dataset was a sampled subset while the clean baseline still pointed at the full source manifest.
- That meant any `delta_auc` or `delta_ap` from the reduced smoke run would mix two effects:
  - distortion robustness
  - sample selection mismatch
- This had to be fixed in code, not just documented away, otherwise the fast smoke path would have produced misleading clean-vs-distorted comparisons.

### What Was Added

1. `distortionPipeline/src/pipeline/manifest_adapter.py`
- Added `build_clean_subset_dataset_name(...)`.
- Added `_prepared_clean_subset_record(...)`.
- Added `augmented_manifest_to_clean_subset_datasets(...)`.
- Why: the augmented distortion manifest already knows exactly which original images were sampled for each treatment. This new export path turns that same sampled set back into a DeepfakeBench clean manifest, so the detector can be evaluated on the identical clean images used to generate the distorted set.

2. `distortionPipeline/scripts/deepfakebench_manifest.py`
- Added the `to-clean-subset-json` CLI command.
- Why: the clean-subset export should be available as a first-class adapter operation, not hidden inside the Python module only.

3. `training/distortion_champion_evaluation.py`
- Added `--clean-baseline-mode` with `full` and `matched_subset` modes.
- Added `comparison_group` handling so distorted rows are matched against the correct clean baseline.
- In `matched_subset` mode, the script now generates clean-subset dataset manifests from the augmented distortion manifest and evaluates those instead of the full source manifest.
- Why: this keeps the fast smoke workflow scientifically valid while preserving the old full-baseline behavior for full-size runs.

4. `distortionPipeline/configs/experiments/champion_smoke_small.yaml`
- Added a reduced recipe config with `50` real + `50` fake images.
- Why: with MPS currently unavailable, the full mini benchmark is too slow for tight integration feedback loops.

5. `distortionPipeline/tests/test_manifest_adapter.py`
- Added a focused test for `augmented_manifest_to_clean_subset_datasets(...)`.
- Why: the new baseline mode depends on the adapter writing original clean paths, not distorted paths.

6. `training/results/distortion_manual_test_next_steps_2026-03-19.md`
- Added an explicit MPS check step.
- Added the matched-subset small smoke command.
- Why: the correct manual validation path now depends on whether the current environment can actually use MPS.

### Environment Observation

Checked in `DeepfakeBench` on `2026-03-19`:

```text
torch 2.8.0
mps_built=True
mps_available=False
```

Interpretation:
- The installed torch build includes MPS support, but the backend is not currently available in the active environment/session.
- For now, distortion evaluation runs are falling back to CPU.

### Verified Runs

1. Distorted-only quickcheck
- Command shape: `Dataset-1-mini` + `xception` + `champion_quickcheck.yaml` + `--no-evaluate-clean`
- Output: `/Users/Hao/thesis-project/training/results/distortion_quickcheck_dataset1mini_xception`
- Verified result:
  - `status=success`
  - `device=cpu`
  - `runtime_sec=6.592`
  - `auc=0.64`
  - predictions and metrics artifacts were written

2. Matched clean-vs-distorted small smoke
- Command shape: `Dataset-1-mini` + `xception` + `champion_smoke_small.yaml` + `--clean-baseline-mode matched_subset`
- Output: `/Users/Hao/thesis-project/training/results/distortion_smoke_small_dataset1mini_xception_matched`
- Verified result:
  - clean subset row succeeded
  - distorted row succeeded
  - clean dataset name contains `__clean_subset__`
  - comparison CSV contains populated delta fields
  - observed deltas:
    - `clean_auc=0.6820`
    - `distorted_auc=0.6204`
    - `delta_auc=-0.0616`
    - `delta_ap=-0.006693`
    - `delta_eer=-0.0200`

### Runs Intentionally Stopped

- The full `Dataset-1-mini` clean-vs-distorted smoke run was started and then stopped after confirming that the current environment was CPU-only.
- Reason: it would have consumed substantially more time without adding new integration confidence once the matched-subset small smoke was available and verified.
