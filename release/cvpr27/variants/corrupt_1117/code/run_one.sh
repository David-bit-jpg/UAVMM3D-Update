#!/bin/bash
# usage: bash run_one.sh <arm> <cond>   (one test-time pass; GPU through gpu_slot light)
A=$1; C=$2; R=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home UAV_PNP_EPNP_FALLBACK=1
mkdir -p $R/$A/$C
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh light D:/Miniconda3/envs/city/python.exe E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117/code/corrupt_ori_x.py $A $C > $R/$A/$C/stdout.log 2>&1
echo "exit $? $A $C"
