# 5 Results

This chapter presents the measured results from the detector evaluation and the distortion evaluation. The reported metrics are accuracy (ACC), area under the receiver operating characteristic curve (AUC), equal error rate (EER), average precision (AP), and runtime in seconds. The evaluation used the full `Dataset-1` and `NVIDIA-dataset` test sets. The detector evaluation included detectors with available compatible pretrained checkpoints.

## 5.1 Stage C Detector Evaluation

Stage C evaluated 10 detectors on two full datasets. The run produced 20 detector-dataset results, all with status `success`. Table 5.1 shows the aggregated detector results across `Dataset-1` and `NVIDIA-dataset`.

| Rank | Detector | Family | Score | AUC mean | AP mean | EER mean | ACC mean | AUC gap | Runtime (s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet | Naive | 0.406179 | 0.604584 | 0.540071 | 0.420450 | 0.491602 | 0.170406 | 175.199 |
| 2 | MesoInception | Naive | 0.349883 | 0.514229 | 0.496763 | 0.483300 | 0.518993 | 0.092909 | 183.784 |
| 3 | Xception | Naive | 0.348919 | 0.545831 | 0.578570 | 0.475900 | 0.557563 | 0.231488 | 542.993 |
| 4 | FFD | Spatial | 0.322812 | 0.544937 | 0.531875 | 0.469600 | 0.520668 | 0.224252 | 767.458 |
| 5 | EfficientNet-B4 | Naive | 0.258110 | 0.442321 | 0.495424 | 0.541300 | 0.511543 | 0.012864 | 996.149 |
| 6 | Capsule | Spatial | 0.243792 | 0.427002 | 0.477325 | 0.555800 | 0.472551 | 0.086396 | 799.360 |
| 7 | UCF | Spatial | 0.218618 | 0.430682 | 0.476916 | 0.548000 | 0.512624 | 0.022522 | 1422.209 |
| 8 | RECCE | Spatial | 0.194443 | 0.393265 | 0.443672 | 0.578950 | 0.498999 | 0.088955 | 1156.746 |
| 9 | CORE | Spatial | 0.176786 | 0.335421 | 0.417861 | 0.625850 | 0.485788 | 0.060426 | 778.712 |
| 10 | CNN-Aug | Naive | 0.134722 | 0.295221 | 0.384140 | 0.643500 | 0.355110 | 0.246470 | 391.670 |

Table 5.1: Stage C detector results aggregated across `Dataset-1` and `NVIDIA-dataset`.

Table 5.2 shows the detector metrics separately for `Dataset-1` and `NVIDIA-dataset`.

| Rank | Detector | Family | AUC Dataset-1 | AUC NVIDIA | AP Dataset-1 | AP NVIDIA | EER Dataset-1 | EER NVIDIA | ACC Dataset-1 | ACC NVIDIA |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet | Naive | 0.689787 | 0.519380 | 0.579608 | 0.500534 | 0.356000 | 0.484900 | 0.482555 | 0.500650 |
| 2 | MesoInception | Naive | 0.560684 | 0.467775 | 0.521077 | 0.472449 | 0.444000 | 0.522600 | 0.560436 | 0.477550 |
| 3 | Xception | Naive | 0.661575 | 0.430087 | 0.698850 | 0.458291 | 0.392000 | 0.559800 | 0.642575 | 0.472550 |
| 4 | FFD | Spatial | 0.657063 | 0.432811 | 0.618560 | 0.445190 | 0.387800 | 0.551400 | 0.587435 | 0.453900 |
| 5 | EfficientNet-B4 | Naive | 0.435889 | 0.448753 | 0.524329 | 0.466520 | 0.547200 | 0.535400 | 0.550987 | 0.472100 |
| 6 | Capsule | Spatial | 0.470200 | 0.383804 | 0.522528 | 0.432123 | 0.524000 | 0.587600 | 0.532503 | 0.412600 |
| 7 | UCF | Spatial | 0.441943 | 0.419421 | 0.506195 | 0.447638 | 0.536800 | 0.559200 | 0.547248 | 0.478000 |
| 8 | RECCE | Spatial | 0.348788 | 0.437742 | 0.431198 | 0.456145 | 0.615400 | 0.542500 | 0.526999 | 0.471000 |
| 9 | CORE | Spatial | 0.305207 | 0.365634 | 0.422438 | 0.413284 | 0.652000 | 0.599700 | 0.535826 | 0.435750 |
| 10 | CNN-Aug | Naive | 0.171986 | 0.418456 | 0.320240 | 0.448040 | 0.734800 | 0.552200 | 0.258671 | 0.451550 |

Table 5.2: Stage C detector results separated by dataset.

## 5.2 Distortion Evaluation

The distortion evaluation used the three detectors selected from Stage C: MesoNet, MesoInception, and Xception. Four distortion recipes were applied to both source datasets: `gaussian_blur_v1`, `jpeg_compress_v1`, `noise_v1`, and `snapchat_text_overlay_v1`. This produced eight distorted dataset manifests and 24 detector-distortion runs. Table 5.3 shows the summary results for the distorted runs.

| Detector | Distortion runs | Mean distorted AUC | Worst distorted AUC | Mean delta AUC | Worst delta AUC | Mean delta AP | Mean delta ACC | Mean delta EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MesoNet | 8 | 0.618061 | 0.518126 | 0.013478 | -0.023914 | 0.022999 | 0.004797 | -0.010562 |
| MesoInception | 8 | 0.524606 | 0.453084 | 0.010376 | -0.029457 | 0.013190 | -0.001288 | -0.005125 |
| Xception | 8 | 0.495138 | 0.406905 | -0.050693 | -0.254671 | -0.044223 | -0.011210 | 0.035600 |

Table 5.3: Detector results on distorted datasets. Delta values are measured against the corresponding clean dataset result.

## 5.3 Real and Fake Class Breakdown

The exported prediction files were used to calculate real-class and fake-class accuracy for clean and distorted conditions. Table 5.4 shows the clean-only class breakdown.

| Detector | Real total | Real correct | Real accuracy | Fake total | Fake correct | Fake accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MesoNet | 15000 | 41 | 0.002733 | 14630 | 14619 | 0.999248 | 0.500991 |
| MesoInception | 15000 | 8327 | 0.555133 | 14630 | 6621 | 0.452563 | 0.503848 |
| Xception | 15000 | 11710 | 0.780667 | 14630 | 3929 | 0.268558 | 0.524612 |

Table 5.4: Real and fake class accuracy for the clean datasets.

Table 5.5 shows the class breakdown for all distorted datasets combined.

| Detector | Real total | Real correct | Real accuracy | Fake total | Fake correct | Fake accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MesoNet | 60000 | 969 | 0.016150 | 58520 | 58067 | 0.992259 | 0.504205 |
| MesoInception | 60000 | 27683 | 0.461383 | 58520 | 32227 | 0.550701 | 0.506042 |
| Xception | 60000 | 49398 | 0.823300 | 58520 | 12663 | 0.216388 | 0.519844 |

Table 5.5: Real and fake class accuracy for all distorted datasets combined.

Table 5.6 shows the balanced accuracy by distortion type.

| Detector | Gaussian blur | JPEG | Noise | Text overlay |
|---|---:|---:|---:|---:|
| MesoNet | 0.500000 | 0.501492 | 0.500958 | 0.514367 |
| MesoInception | 0.479995 | 0.503891 | 0.536395 | 0.503887 |
| Xception | 0.508437 | 0.519332 | 0.498505 | 0.553101 |

Table 5.6: Balanced accuracy by distortion type.

