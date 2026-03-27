# Real-vs-Fake Performance Breakdown

Computed from exported `predictions.csv` files in the completed full matrix benchmark:

- `/Users/Hao/thesis-project/training/results/final_full_thesis_matrix_top3`

Interpretation:
- `real_accuracy` = correctly classified real images / total real images
- `fake_accuracy` = correctly classified fake images / total fake images
- `balanced_accuracy` = mean of real and fake accuracy
- `better_class` indicates which class the detector handled better

## Clean Only

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.002733 | 0.999248 | 0.500991 | fake |
| MesoInception | 0.555133 | 0.452563 | 0.503848 | real |
| Xception | 0.780667 | 0.268558 | 0.524612 | real |

## Distorted Only

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.016150 | 0.992259 | 0.504205 | fake |
| MesoInception | 0.461383 | 0.550701 | 0.506042 | fake |
| Xception | 0.823300 | 0.216388 | 0.519844 | real |

## Gaussian Blur

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.000000 | 1.000000 | 0.500000 | fake |
| MesoInception | 0.227933 | 0.732057 | 0.479995 | fake |
| Xception | 0.705733 | 0.311141 | 0.508437 | real |

## JPEG

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.003600 | 0.999385 | 0.501492 | fake |
| MesoInception | 0.570667 | 0.437116 | 0.503891 | real |
| Xception | 0.844200 | 0.194463 | 0.519332 | real |

## Noise

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.002600 | 0.999316 | 0.500958 | fake |
| MesoInception | 0.467800 | 0.604990 | 0.536395 | fake |
| Xception | 0.993933 | 0.003076 | 0.498505 | real |

## Text Overlay

| Detector | Real Accuracy | Fake Accuracy | Balanced Accuracy | Better Class |
|---|---:|---:|---:|---|
| MesoNet | 0.058400 | 0.970335 | 0.514367 | fake |
| MesoInception | 0.579133 | 0.428640 | 0.503887 | real |
| Xception | 0.749333 | 0.356869 | 0.553101 | real |

