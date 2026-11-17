#!/usr/bin/env bash
# 一键装环境（Linux + NVIDIA 显卡；Windows 用 Git Bash 也行）。用法：bash setup/install.sh
# 1) 建 conda 环境 uav3d（Python 3.10）  2) 装 torch 2.11 + CUDA 12.8 版  3) 装其余依赖  4) 把代码里的路径换成本包位置
set -e
cd "$(dirname "$0")/.."
ENV=${ENV:-uav3d}
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.10
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV"
# 原机器用的是 torch 2.11.0+cu128。显卡驱动太老装不上时，改成 cu126 / cu121 对应版本即可（代码不依赖特定版本特性）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r setup/requirements.txt
python setup/fix_paths.py
python -c "import torch; print('torch', torch.__version__, 'CUDA 可用:', torch.cuda.is_available(), '显卡数:', torch.cuda.device_count())"
echo "环境装好了。以后每次先运行：conda activate $ENV"
