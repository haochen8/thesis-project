# MPS Restore Investigation

Generated: `2026-03-19`

## Goal

Restore MPS acceleration before running the next distortion benchmark stages.

## Current DeepfakeBench Env

Environment:
- Env: `DeepfakeBench`
- Python: `3.9.23`
- Torch: `2.8.0`
- Machine: `arm64`
- macOS: `26.3.1`

Observed commands:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench
python - <<'PY'
import platform, torch
print('mac_ver', platform.mac_ver())
print('torch', torch.__version__)
print('mps_built', torch.backends.mps.is_built())
print('mps_available', torch.backends.mps.is_available())
print(torch.zeros(1, device='mps').device)
PY
```

Observed result:

```text
mac_ver ('26.3.1', ('', '', ''), 'arm64')
torch 2.8.0
mps_built True
mps_available False
RuntimeError: The MPS backend is supported on MacOS 13.0+.Current OS version can be queried using `sw_vers`
```

Additional probe:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench
python - <<'PY'
import torch
for major, minor in [(13,0),(14,0),(15,0),(26,0)]:
    print((major, minor), torch.backends.mps.is_macos_or_newer(major, minor))
PY
```

Observed result:

```text
/Users/Hao/miniconda3/envs/DeepfakeBench/lib/python3.9/site-packages/torch/backends/mps/__init__.py:31: UserWarning: Checking for unexpected MacOS 26.0 returning false (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSHooks.mm:59.)
(13, 0) True
(14, 0) True
(15, 0) True
(26, 0) False
```

Interpretation:
- The current PyTorch build is hitting an upstream macOS 26 MPS gating issue.
- This is not caused by `training/test.py` or the distortion integration code.

## Fresh Official PyTorch Test Env

Created temporary env:
- `torch-mps-check`

Installed:
- Python `3.11.15`
- `torch 2.12.0.dev20260318`
- `torchvision 0.26.0.dev20260318`
- `torchaudio 2.11.0.dev20260319`

Validation command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-mps-check
python - <<'PY'
import platform, sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('mac_ver', platform.mac_ver())
print('mps_built', torch.backends.mps.is_built())
print('mps_available', torch.backends.mps.is_available())
print(torch.zeros(1, device='mps').device)
PY
```

Observed result:

```text
python 3.11.15
torch 2.12.0.dev20260318
mac_ver ('26.3.1', ('', '', ''), 'arm64')
mps_built True
mps_available True
mps:0
```

Interpretation:
- MPS works on this machine with a newer official nightly PyTorch build.
- Therefore the blocker is not hardware or project code.

## Why DeepfakeBench Was Not Repaired In-Place

Attempted package replacement path:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DeepfakeBench
python -m pip install --upgrade --force-reinstall --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cpu
```

Observed result:

```text
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
ERROR: No matching distribution found for torch
```

Interpretation:
- Official nightly wheels are not available for the current `DeepfakeBench` Python `3.9` environment.
- The working repair path currently requires Python `3.11`.

## Conclusion

MPS is not restorable in the current `DeepfakeBench` env (`Python 3.9 + torch 2.8.0`) with a simple torch upgrade.

The only verified working path found today is:
1. Python `3.11`
2. current official nightly PyTorch build

## Supporting Artifact

Pre-change package snapshot:
- `/Users/Hao/thesis-project/training/results/deepfakebench_conda_list_before_mps_restore_2026-03-19.txt`
