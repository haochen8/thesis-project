# Spatial Detector Champion Report

- Generated: `2026-02-23T18:22:59.918263`
- Datasets: `Dataset-1-stageB, NVIDIA-dataset-stageB`
- Selected detectors: `24`
- Ranked complete detectors: `10`
- Output dir: `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix`

## Top 3 Champions

| Rank | Detector | Family | Score | AUC(mean) | AP(mean) | EER(mean) | ACC(mean) | AUC gap | Runtime(s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MesoNet (`mesonet`) | naive | 0.410195 | 0.608183 | 0.553272 | 0.4207 | 0.5010 | 0.183003 | 168.326 |
| 2 | Xception (`xception`) | naive | 0.357339 | 0.544289 | 0.585693 | 0.4783 | 0.5514 | 0.241861 | 289.944 |
| 3 | MesoInception (`mesoinception`) | naive | 0.350807 | 0.514404 | 0.506890 | 0.4832 | 0.5163 | 0.101023 | 173.000 |

## Failed or Missing Runs

| Detector | Dataset | Status | Detail | Log |
|---|---|---|---|---|
| `dsp-fwa` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/dsp-fwa__Dataset-1-stageB.log` |
| `dsp-fwa` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/dsp-fwa__NVIDIA-dataset-stageB.log` |
| `face-xray` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/face-xray__Dataset-1-stageB.log` |
| `face-xray` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/face-xray__NVIDIA-dataset-stageB.log` |
| `local-relation` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/local-relation__Dataset-1-stageB.log` |
| `local-relation` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/local-relation__NVIDIA-dataset-stageB.log` |
| `iid` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/iid__Dataset-1-stageB.log` |
| `iid` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/iid__NVIDIA-dataset-stageB.log` |
| `rfm` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/rfm__Dataset-1-stageB.log` |
| `rfm` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/rfm__NVIDIA-dataset-stageB.log` |
| `sia` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sia__Dataset-1-stageB.log` |
| `sia` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sia__NVIDIA-dataset-stageB.log` |
| `sladd` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sladd__Dataset-1-stageB.log` |
| `sladd` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sladd__NVIDIA-dataset-stageB.log` |
| `uia-vit` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/uia-vit__Dataset-1-stageB.log` |
| `uia-vit` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/uia-vit__NVIDIA-dataset-stageB.log` |
| `clip` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/clip__Dataset-1-stageB.log` |
| `clip` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/clip__NVIDIA-dataset-stageB.log` |
| `sbi` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sbi__Dataset-1-stageB.log` |
| `sbi` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/sbi__NVIDIA-dataset-stageB.log` |
| `pcl-i2g` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/pcl-i2g__Dataset-1-stageB.log` |
| `pcl-i2g` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/pcl-i2g__NVIDIA-dataset-stageB.log` |
| `multi-attention` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/multi-attention__Dataset-1-stageB.log` |
| `multi-attention` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/multi-attention__NVIDIA-dataset-stageB.log` |
| `lsda` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/lsda__Dataset-1-stageB.log` |
| `lsda` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/lsda__NVIDIA-dataset-stageB.log` |
| `effort` | `Dataset-1-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/effort__Dataset-1-stageB.log` |
| `effort` | `NVIDIA-dataset-stageB` | `missing_weight` | `No matching .pth found` | `/Users/Hao/thesis-project/training/results/stage_b_spatial/logs/effort__NVIDIA-dataset-stageB.log` |

## Artifacts

- Raw run table: `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix/raw_runs.csv`
- Detector summary: `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix/detector_summary.csv`
- Champion JSON: `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix/champions.json`
- Run config: `/Users/Hao/thesis-project/training/results/stage_b_spatial_merged_pretrained_fix/run_config.json`
