# Spatial Detector Champion Report (Existing Weights Only)

- Generated: `2026-02-27T19:27:05.077058`
- Datasets: `Dataset-1, NVIDIA-dataset`
- Policy: `Only detectors with available, compatible checkpoints were evaluated and ranked.`
- Tested detectors: `10`
- Excluded spatial detectors (missing weights): `14`

## Tested Detectors

`mesonet`, `mesoinception`, `xception`, `ffd`, `efficientnet-b4`, `capsule`, `ucf`, `recce`, `core`, `cnn-aug`

## Excluded Spatial Detectors (Missing Weights)

`dsp-fwa`, `face-xray`, `local-relation`, `iid`, `rfm`, `sia`, `sladd`, `uia-vit`, `clip`, `sbi`, `pcl-i2g`, `multi-attention`, `lsda`, `effort`

These detectors were not tested because compatible pretrained detector checkpoints were not available locally or in official release assets used for this run.

## Top 3 Champions

| Rank | Detector | Family | Score | AUC(mean) | AP(mean) | EER(mean) | ACC(mean) | AUC gap | Runtime(s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet (`mesonet`) | naive | 0.406179 | 0.604584 | 0.540071 | 0.420450 | 0.491602 | 0.170406 | 175.199 |
| 2 | MesoInception (`mesoinception`) | naive | 0.349883 | 0.514229 | 0.496763 | 0.483300 | 0.518993 | 0.092909 | 183.784 |
| 3 | Xception (`xception`) | naive | 0.348919 | 0.545831 | 0.578570 | 0.475900 | 0.557563 | 0.231488 | 542.993 |

## Full Ranking (Tested Detectors Only)

| Rank | Detector | Family | Score | AUC(mean) | AP(mean) | EER(mean) | ACC(mean) | AUC gap | Runtime(s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet (`mesonet`) | naive | 0.406179 | 0.604584 | 0.540071 | 0.420450 | 0.491602 | 0.170406 | 175.199 |
| 2 | MesoInception (`mesoinception`) | naive | 0.349883 | 0.514229 | 0.496763 | 0.483300 | 0.518993 | 0.092909 | 183.784 |
| 3 | Xception (`xception`) | naive | 0.348919 | 0.545831 | 0.578570 | 0.475900 | 0.557563 | 0.231488 | 542.993 |
| 4 | FFD (`ffd`) | spatial | 0.322812 | 0.544937 | 0.531875 | 0.469600 | 0.520668 | 0.224252 | 767.458 |
| 5 | EfficientNet-B4 (`efficientnet-b4`) | naive | 0.258110 | 0.442321 | 0.495424 | 0.541300 | 0.511543 | 0.012864 | 996.149 |
| 6 | Capsule (`capsule`) | spatial | 0.243792 | 0.427002 | 0.477325 | 0.555800 | 0.472551 | 0.086396 | 799.360 |
| 7 | UCF (`ucf`) | spatial | 0.218618 | 0.430682 | 0.476916 | 0.548000 | 0.512624 | 0.022522 | 1422.209 |
| 8 | RECCE (`recce`) | spatial | 0.194443 | 0.393265 | 0.443672 | 0.578950 | 0.498999 | 0.088955 | 1156.746 |
| 9 | CORE (`core`) | spatial | 0.176786 | 0.335421 | 0.417861 | 0.625850 | 0.485788 | 0.060426 | 778.712 |
| 10 | CNN-Aug (`cnn-aug`) | naive | 0.134722 | 0.295221 | 0.384140 | 0.643500 | 0.355110 | 0.246470 | 391.670 |

## Artifacts

- `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/raw_runs.csv`
- `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/detector_summary.csv`
- `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/champions.json`
- `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/run_config.json`
