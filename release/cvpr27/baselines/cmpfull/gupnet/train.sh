#!/usr/bin/env bash
# GUPNet（官方代码最小改动）在无人机统一数据上训练。参数原样传给 tools/train_uav.py：
#   bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [伪标签.pkl ...] --out <dir> --seed 444 --epochs 140
# 需要官方 ImageNet DLA-34 权重：export GUPNET_DLA34_WEIGHTS=<.../dla34-ba72cf86.pth>（或让官方代码自己联网下载）
set -e
export PYTHONUTF8=1
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-D:/Miniconda3/envs/city/python.exe}"
cd "$HERE/code"
"$PY" tools/train_uav.py "$@"
