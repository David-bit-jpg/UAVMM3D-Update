#!/bin/bash
# nose_det 串行驱动：driver_sim.sh <tag> <ckpt> <zoom...>   每个 zoom 跑一次 run_heading.py（GPU，串行）
TAG=$1; CKPT=$2; shift 2
cd E:/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=D:/Miniconda3/envs/city/python.exe
OUT=E:/Open3DUAVDet/output/camnorm/annot_audit/nose_det
for z in "$@"; do
  echo "=== $TAG zoom $z start $(date)" >> "$OUT/driver.log"
  "$PY" annot_audit/nose_det/run_heading.py --ckpt "$CKPT" --tag "$TAG" --zoom "$z" --interval 1 >> "$OUT/driver.log" 2>&1 \
    || echo "FAILED $TAG zoom $z" >> "$OUT/driver.log"
  echo "=== $TAG zoom $z end $(date)" >> "$OUT/driver.log"
done
echo "=== CHUNK DONE $TAG $(date)" >> "$OUT/driver.log"
