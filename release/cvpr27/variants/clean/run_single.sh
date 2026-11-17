#!/bin/bash
# [clean item 1] single-model no-TTA PST on the clean in2mav chain (M12 recipe, = run_m1_anti.sh with in2mav inputs):
# self-trained detector DET1_s0 without flip; refiner stage-3 seed 5 only; orientation student S_ab0 only, no mirror view;
# height network unchanged (KPSs0; BN re-estimated on deployed tr1 crops, the only reused train-split reading);
# w = deployed w_sim.json (0.1903); constants s_bar / colour centres / c_k / r_k / b_k recomputed by chain_x deploy from this model's own unlabeled images.
# Writes only under review1113/clean/single. GPU: gpu_slot light + memory fraction 0.2.
C=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean; CB=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean
X=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav; PY=D:/Miniconda3/envs/city/python.exe
SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh light"; MC=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/memcap.py
AO=$C/single; AB=$CB/single; L=$AB/logs; LIB=$C/lib
KPS=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_kps/KPSs0/ckpt/checkpoint_epoch_8.pth
DET=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/srcx/st_in2mav/DET1_s0/ckpt/checkpoint_epoch_3.pth
export PYTHONUTF8=1 SX_SRC=in SX_TGT=mav SX_TEACH=full CL_AO=$AO MEMFRAC=0.2 UAV_PNP_EPNP_FALLBACK=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $L/run_single.log; }
mkdir -p $AB/recs $AB/ref $AB/ou $AB/preds $AB/work
cd $CB/lib
say start
[ -f $AB/recs/det1.pkl ] || M12_NOFLIP=1 $SLOT $PY $MC $LIB/det_infer_x.py --ckpt $DET --splits train val test --out $AO/recs/det1.pkl --size-fallback $X/recs/det0.pkl > $L/det1_infer.log 2>&1 || { say "det failed"; exit 2; }
grep -E "s_bar|frames" $L/det1_infer.log | tee -a $L/run_single.log
for x in "val val1" "test te1"; do set -- $x; [ -f $AB/work/oc_$2.pkl ] || $PY $LIB/ori_x.py crops $AO/recs/det1.pkl $1 1 $2 >> $L/run_single.log 2>&1 || { say "crops $2 failed"; exit 2; }; done
for n in val1 te1; do [ -f $AB/ou/read_${n}_stu.pkl ] || ORI_MIRROR=0 $SLOT $PY $MC $LIB/ori_x.py read $n $AO/ou/read_${n}_stu.pkl $X/ou/S_ab0.pth >> $L/run_single.log 2>&1 || { say "read $n failed"; exit 2; }; done
[ -f $AB/kps_te1.pkl ] || $SLOT $PY $MC $LIB/ori_x.py kps $KPS tr1,val1,te1 "$AO/kps_%s.pkl" 80 >> $L/run_single.log 2>&1 || { say "kps failed"; exit 2; }
for sp in val test train; do [ -f $AB/ref/inf_fin_det1_$sp.pkl ] || $SLOT $PY $MC $LIB/refiner_x.py infer $X/ref/ref_st3_s5.pt $AO/recs/det1.pkl $sp $AO/ref/inf_fin_det1_$sp.pkl >> $L/run_single.log 2>&1 || { say "refiner $sp failed"; exit 2; }; done
WW=$($PY -c "import json;print(json.load(open('$X/w_sim.json'))['w'])"); say "w (deployed) = $WW"
[ -f $AB/deploy_single.json ] || $PY $LIB/chain_x.py deploy $WW single >> $L/run_single.log 2>&1 || { say "deploy failed"; exit 2; }
for t in fused nofuse; do $PY $LIB/score_x.py $AO/preds/single_${t}_val.pkl val >> $L/run_single.log 2>&1; done
for t in fused nofuse; do $PY $LIB/score_x.py $AO/preds/single_${t}_test.pkl real_test 2>&1 | tee -a $L/run_single.log; done
say "ALL DONE"
