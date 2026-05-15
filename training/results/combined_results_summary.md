# Combined Results Summary

Generated: `2026-05-15`

This summary combines the two final result packages at a high level:

- **Run 1:** this thesis project, from
  `training/results/final_full_thesis_matrix_top3`
- **Run 2:** [`litmas/distortionPipeline`](https://github.com/litmas/distortionPipeline/tree/main),
  inspected at commit `6b92135`

The two runs should be interpreted side by side, not merged into a single leaderboard.
They use different datasets, detector pools, and distortion families.

## Run Scope

| Run | Datasets | Detectors | Distortions | Primary Metric | Rows |
|---|---|---|---|---|---:|
| Run 1, this project | `Dataset-1`, `NVIDIA-dataset` | `MesoNet`, `MesoInception`, `Xception` | Gaussian blur, JPEG, noise, text overlay | AUC / video AUC | 24 |
| Run 2, distortionPipeline | Celeb-DF-v1 screening subset, FaceForensics++, DFDCP | `UCF`, `SPSL`, `F3Net` | Snapchat, Instagram, TikTok-style distortions | video AUC | 9 |

## Run 1: Local Thesis Matrix

Run 1 evaluated three available champion detectors from this project across two custom
datasets and four image distortions. The strongest average distorted performance came
from `MesoNet`, while `Xception` was the most fragile under the tested distortions.

| Detector | Mean Distorted AUC | Worst Distorted AUC | Mean Delta AUC | Worst Delta AUC |
|---|---:|---:|---:|---:|
| MesoNet | 0.6181 | 0.5181 | 0.0135 | -0.0239 |
| MesoInception | 0.5246 | 0.4531 | 0.0104 | -0.0295 |
| Xception | 0.4951 | 0.4069 | -0.0507 | -0.2547 |

By distortion type, blur and JPEG produced the largest average drops. Noise had a near-zero
average effect across all rows, but it contained the single worst detector-specific drop
because `Xception` degraded sharply in one condition.

| Distortion | Mean Distorted AUC | Mean Delta AUC | Worst Delta AUC |
|---|---:|---:|---:|
| Gaussian blur | 0.5392 | -0.0156 | -0.0737 |
| JPEG | 0.5384 | -0.0165 | -0.1136 |
| Noise | 0.5562 | 0.0013 | -0.2547 |
| Text overlay | 0.5499 | -0.0050 | -0.0365 |

The dataset split matters. `Dataset-1` had the higher mean distorted AUC but also the
larger mean degradation. `NVIDIA-dataset` stayed closer to its clean baseline, although
its absolute mean AUC was lower.

| Dataset | Mean Distorted AUC | Mean Delta AUC | Worst Distorted AUC |
|---|---:|---:|---:|
| Dataset-1 | 0.6094 | -0.0279 | 0.4069 |
| NVIDIA-dataset | 0.4824 | 0.0100 | 0.4162 |

## Run 2: distortionPipeline Results

Run 2 evaluates stronger frequency/spatial detectors on a broader benchmark-style setup.
For the full test sets only, `F3Net` had the highest average champion score and highest
average distorted video AUC, while `UCF` had the best FaceForensics++ result in the
per-dataset table.

| Detector | Mean Champion Score | Mean Base Video AUC | Mean Avg Distorted Video AUC | Mean Worst Distorted Video AUC | Mean Drop |
|---|---:|---:|---:|---:|---:|
| F3Net | 0.8532 | 0.8680 | 0.8433 | 0.8190 | 0.0247 |
| UCF | 0.8382 | 0.8556 | 0.8266 | 0.8085 | 0.0290 |
| SPSL | 0.8368 | 0.8523 | 0.8265 | 0.8030 | 0.0258 |

Across the full test sets, Instagram-style filtering was the least damaging condition on
average, while Snapchat and TikTok-style treatments caused larger drops.

| Distortion | Mean Full-Test Video AUC | Worst Full-Test Video AUC |
|---|---:|---:|
| Instagram | 0.8533 | 0.7109 |
| TikTok | 0.8255 | 0.6597 |
| Snapchat | 0.8176 | 0.6746 |

The Celeb-DF-v1 rows in Run 2 are screening-subset results, not full final evaluation rows.
They are useful for confirmation-stage discussion, but the thesis main comparison should
lean on FaceForensics++ and DFDCP when describing full-test behavior.

## Combined Interpretation

The main combined finding is that robustness depends more on detector family and dataset
than on the presence of distortion alone. In Run 1, the available naive detectors operate
near the decision boundary on the custom datasets; small image treatments can change AUC
in either direction, and `Xception` is vulnerable to large condition-specific drops. In
Run 2, the stronger detectors keep substantially higher video AUC on full benchmark test
sets, but still show consistent degradation under social-media-style transformations.

The two runs also point to different practical risks:

- **Run 1 risk:** low absolute AUC on the custom datasets means robustness claims should be
  framed carefully; a small delta does not necessarily imply a strong detector.
- **Run 2 risk:** high clean performance does not remove distortion sensitivity; worst-case
  social-media treatments still reduce detector margins.
- **Shared pattern:** average robustness can hide brittle cases, so the thesis should report
  both average distorted AUC and worst distorted AUC/drop.

For thesis writing, the cleanest combined statement is:

> The local thesis matrix shows that the available naive detectors are fragile and often
> close to chance on the custom datasets, with `MesoNet` most stable among the local trio
> and `Xception` showing the largest worst-case degradation. The independent
> `distortionPipeline` run confirms the same robustness concern on stronger detectors and
> standard benchmark datasets: `F3Net`, `UCF`, and `SPSL` remain much stronger in absolute
> video AUC, but social-media-style distortions still lower performance, especially in
> worst-case conditions.

## Source Files

- Run 1 report:
  `training/results/final_full_thesis_matrix_top3/distortion_champion_report.md`
- Run 1 summary CSV:
  `training/results/final_full_thesis_matrix_top3/detector_distortion_summary.csv`
- Run 1 comparison CSV:
  `training/results/final_full_thesis_matrix_top3/detector_distortion_comparison.csv`
- Run 2 results package:
  <https://github.com/litmas/distortionPipeline/tree/main/results>
- Run 2 summary CSV:
  `results/csv/final_results_summary.csv` in `litmas/distortionPipeline`
