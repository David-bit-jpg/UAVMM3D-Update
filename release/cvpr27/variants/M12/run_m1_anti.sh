#!/bin/bash
# [M12] AntiUAV3D v2 (out2antifs) single-model no-TTA PST: self-trained detector DET1_s0 without flip, refiner stage-3 seed 5 only,
# orientation student S_ab0 only without mirror view, height network unchanged; w = deployed w_sim.json; same chain_x deploy rules.
# Writes only under M12/anti/single (lib copies in M12/antilib redirect sx.O / sx.W).
M=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12; MB=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12
X=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs; PY=D:/Miniconda3/envs/city/python.exe
AO=$M/anti/single; AB=$MB/anti/single; L=$AB/logs
KPS=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_out_r34_kp_w12_zoom_irlid_kps_out/OKZkps_s0/ckpt/checkpoint_epoch_8.pth
export PYTHONUTF8=1 SX_SRC=out SX_TGT=antifs SX_TEACH=full M12_AO=$AO UAV_PNP_EPNP_FALLBACK=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $L/run_m1_anti.log; }
cd $MB/antilib
until [ -f $AB/recs/det1.pkl ]; do sleep 30; done
grep -E "s_bar|frames" $L/det1_infer.log | tee -a $L/run_m1_anti.log
for x in "val val1" "test te1"; do set -- $x; [ -f $AB/work/oc_$2.pkl ] || $PY ori_x.py crops $AO/recs/det1.pkl $1 1 $2 >> $L/run_m1_anti.log 2>&1 || { say "crops $2 failed"; exit 2; }; done
for n in val1 te1; do [ -f $AB/ou/read_${n}_stu.pkl ] || ORI_MIRROR=0 $PY $M/memcap.py $M/antilib/ori_x.py read $n $AO/ou/read_${n}_stu.pkl $X/ou/S_ab0.pth >> $L/run_m1_anti.log 2>&1 || { say "read $n failed"; exit 2; }; done
[ -f $AB/kps_te1.pkl ] || $PY $M/memcap.py $M/antilib/ori_x.py kps $KPS tr1,val1,te1 "$AO/kps_%s.pkl" 80 >> $L/run_m1_anti.log 2>&1 || { say "kps failed"; exit 2; }
for sp in train val test; do [ -f $AB/ref/inf_fin_det1_$sp.pkl ] || $PY $M/memcap.py $M/antilib/refiner_x.py infer $X/ref/ref_st3_s5.pt $AO/recs/det1.pkl $sp $AO/ref/inf_fin_det1_$sp.pkl >> $L/run_m1_anti.log 2>&1 || { say "refiner $sp failed"; exit 2; }; done
WW=$($PY -c "import json;print(json.load(open('$X/w_sim.json'))['w'])"); say "w (deployed) = $WW"
[ -f $AB/deploy_single.json ] || $PY chain_x.py deploy $WW single >> $L/run_m1_anti.log 2>&1 || { say "deploy failed"; exit 2; }
$PY score_x.py $AO/preds/single_fused_val.pkl val >> $L/run_m1_anti.log 2>&1
$PY score_x.py $AO/preds/single_fused_test.pkl antifs_test 2>&1 | tee -a $L/run_m1_anti.log
say "ALL DONE"
