#!/usr/bin/env bash
# [UAV port] MonoCD training entry. Usage:
#   bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [pseudo.pkl ...] --out OUT --seed 0 --epochs 19
# Optional env: GPU_FRAC (default 0.2; official batch 8 needs ~0.6, see NOTES.md), DLA34_PTH (ImageNet DLA-34 weights).
export PYTHONUTF8=1
HERE="$(cd "$(dirname "$0")" && pwd)"
exec D:/Miniconda3/envs/city/python.exe "$HERE/train.py" "$@"
