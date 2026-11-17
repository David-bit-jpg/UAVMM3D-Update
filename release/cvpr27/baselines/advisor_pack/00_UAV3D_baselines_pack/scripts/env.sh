# 每个脚本开头都会 source 这个文件：统一的根目录和环境变量
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && (pwd -W 2>/dev/null || pwd))"
export PKG
export R="$PKG/Open3DUAVDet"
export G="$R/output/camnorm/geo1004"
export CF="$G/cmpfull"
export DATA="$PKG/data/cmpfull"
export RUNS="$PKG/runs"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
case "$OSTYPE" in msys*|cygwin*) SEP=';' ;; *) SEP=':' ;; esac   # Windows 的 Git Bash 也能用
export PYTHONPATH="$R${PYTHONPATH:+$SEP$PYTHONPATH}"
export TORCH_HOME="$PKG/torch_home"
export DLA34_PTH="$TORCH_HOME/hub/checkpoints/dla34-ba72cf86.pth"
export GUPNET_DLA34_WEIGHTS="$DLA34_PTH"
export GPU_FRAC=${GPU_FRAC:-0.95}      # 每个进程最多用多少显存（独占服务器就用满）
export PY=${PY:-python}
