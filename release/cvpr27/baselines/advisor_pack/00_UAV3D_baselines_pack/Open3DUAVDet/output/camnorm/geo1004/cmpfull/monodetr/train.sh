#!/usr/bin/env bash
# [UAV port] MonoDETR training on the cmpfull data. Usage:
#   bash train.sh --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl [pseudo.pkl@@@ROOT@@/data/cmpfull/real_train ...] --out <dir> --seed 444 --epochs 37 --batch-size 2 [--resume] [--max-iters N]
# GPU_FRAC (default 0.2) caps per-process GPU memory. The original train.sh (KITTI) is kept as train_kitti.sh.
export PYTHONUTF8=1
export TORCH_HOME="${TORCH_HOME:-@@ROOT@@/torch_home}"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PY:-python}" "$HERE/tools/train_uav.py" "$@"
