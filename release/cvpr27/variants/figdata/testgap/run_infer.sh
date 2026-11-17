#!/bin/bash
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
T=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/figdata/testgap; PY=D:/Miniconda3/envs/city/python.exe
for j in 5 6; do
  [ -f $T/st2_s${j}_det0_test.pkl ] || $PY $T/tg_refiner_infer.py $j test $T/st2_s${j}_det0_test.pkl || exit 2
done
echo INFER_DONE
