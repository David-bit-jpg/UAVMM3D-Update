#!/bin/bash
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
N=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean; PY=D:/Miniconda3/envs/city/python.exe
for j in 5 6; do
  [ -f $N/chk_s${j}_train300.pkl ] || $PY $N/n1c_refiner_infer.py $j train $N/chk_s${j}_train300.pkl 300 || exit 2
  [ -f $N/st2_s${j}_det0_val.pkl ] || $PY $N/n1c_refiner_infer.py $j val $N/st2_s${j}_det0_val.pkl || exit 2
done
echo INFER_DONE
