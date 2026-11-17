#!/usr/bin/env bash
# [UAV port] MonoDGP training on the cmpfull data. Usage:
#   bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [pseudo.pkl@D:/cmpfull/real_train ...] --out <dir> --seed 444 --epochs 48 [--batch-size 8]
# GPU_FRAC (default 0.2) caps per-process GPU memory. The original train.sh (KITTI) is kept as train_kitti.sh.
export PYTHONUTF8=1
HERE="$(cd "$(dirname "$0")" && pwd)"
exec D:/Miniconda3/envs/city/python.exe "$HERE/tools/train_uav.py" "$@"
