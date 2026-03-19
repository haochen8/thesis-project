# Stage C Separate Dataset Results (Direct Comparison)

This report compares detector performance on full `Dataset-1` vs full `NVIDIA-dataset` from Stage C.

- Source raw file: `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/raw_runs.csv`
- Split files: `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/raw_runs_dataset1.csv`, `/Users/Hao/thesis-project/training/results/stage_c_existing_detectors/raw_runs_nvidia.csv`

## Side-by-Side Metrics

| Rank | Detector | Family | AUC D1 | AUC NVIDIA | ΔAUC (NV-D1) | AP D1 | AP NVIDIA | ΔAP (NV-D1) | EER D1 | EER NVIDIA | ΔEER (NV-D1) | ACC D1 | ACC NVIDIA | Runtime D1(s) | Runtime NVIDIA(s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet (`mesonet`) | naive | 0.689787 | 0.519380 | -0.170406 | 0.579608 | 0.500534 | -0.079074 | 0.356000 | 0.484900 | +0.128900 | 0.482555 | 0.500650 | 84.815 | 90.384 |
| 2 | MesoInception (`mesoinception`) | naive | 0.560684 | 0.467775 | -0.092909 | 0.521077 | 0.472449 | -0.048628 | 0.444000 | 0.522600 | +0.078600 | 0.560436 | 0.477550 | 87.289 | 96.495 |
| 3 | Xception (`xception`) | naive | 0.661575 | 0.430087 | -0.231488 | 0.698850 | 0.458291 | -0.240559 | 0.392000 | 0.559800 | +0.167800 | 0.642575 | 0.472550 | 189.592 | 353.401 |
| 4 | FFD (`ffd`) | spatial | 0.657063 | 0.432811 | -0.224252 | 0.618560 | 0.445190 | -0.173370 | 0.387800 | 0.551400 | +0.163600 | 0.587435 | 0.453900 | 269.682 | 497.776 |
| 5 | EfficientNet-B4 (`efficientnet-b4`) | naive | 0.435889 | 0.448753 | +0.012864 | 0.524329 | 0.466520 | -0.057808 | 0.547200 | 0.535400 | -0.011800 | 0.550987 | 0.472100 | 337.431 | 658.718 |
| 6 | Capsule (`capsule`) | spatial | 0.470200 | 0.383804 | -0.086396 | 0.522528 | 0.432123 | -0.090404 | 0.524000 | 0.587600 | +0.063600 | 0.532503 | 0.412600 | 340.166 | 459.194 |
| 7 | UCF (`ucf`) | spatial | 0.441943 | 0.419421 | -0.022522 | 0.506195 | 0.447638 | -0.058558 | 0.536800 | 0.559200 | +0.022400 | 0.547248 | 0.478000 | 440.545 | 981.664 |
| 8 | RECCE (`recce`) | spatial | 0.348788 | 0.437742 | +0.088955 | 0.431198 | 0.456145 | +0.024947 | 0.615400 | 0.542500 | -0.072900 | 0.526999 | 0.471000 | 412.951 | 743.795 |
| 9 | CORE (`core`) | spatial | 0.305207 | 0.365634 | +0.060426 | 0.422438 | 0.413284 | -0.009154 | 0.652000 | 0.599700 | -0.052300 | 0.535826 | 0.435750 | 277.839 | 500.873 |
| 10 | CNN-Aug (`cnn-aug`) | naive | 0.171986 | 0.418456 | +0.246470 | 0.320240 | 0.448040 | +0.127800 | 0.734800 | 0.552200 | -0.182600 | 0.258671 | 0.451550 | 157.130 | 234.540 |

Interpretation: positive `ΔAUC`/`ΔAP` means better on NVIDIA; positive `ΔEER` means worse on NVIDIA.

