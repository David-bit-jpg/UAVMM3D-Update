# 每个 release/*.sh 开头都会 source 这个文件：统一的根目录和环境变量。
# 先运行一次 python release/fix_paths.py，把代码里的路径占位符换成本机路径。
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && (pwd -W 2>/dev/null || pwd))"
export R
export G="$R/output/camnorm/geo1004"
export O="$G/proto"
export INST="$G/proto/inst"
export S="$R/output/camnorm/night1003/scratch"
export REPL="$R/output/camnorm/night1003/repl"
export M="$R/output/models/uavdet_3d/camnorm"
export CN="@@MMCACHE@@/mav6d_cn_obj"
export RUNS="$R/runs"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
case "$OSTYPE" in msys*|cygwin*) SEP=';' ;; *) SEP=':' ;; esac   # Windows 的 Git Bash 也能用
export PYTHONPATH="$R${PYTHONPATH:+$SEP$PYTHONPATH}"
export TORCH_HOME="@@TORCH_HOME@@"
PY=${PY:-python}
case "$CN" in *@@*) echo "还没换路径：先运行 python release/fix_paths.py"; exit 1 ;; esac
