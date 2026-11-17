#!/bin/bash
# [clean2 item 2] NOT RUN. Commands to retrain the clean chain's step-3 depth students with omega = 0 (targets = IR depth only:
# delta = log(z_pl / z_det0), no corner term), seeds 5 and 6, everything else unchanged, then deploy + score.
# Uses the inference-capable copies in review1113/clean/lib (sx.O/sx.W redirected to $CL_AO), so nothing under srcx/in2mav is written.
# Training batch sizes are untouched in that copy (64 sim + 64 real); only refiner INFERENCE batch is 8 there (128 in srcx/lib).
set -e
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
L=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/lib
X=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav; XW=D:/S2MUAV3D/srcx_work/in2mav
export CL_AO=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0; A=$CL_AO
mkdir -p $A/recs $A/ref $A/ou $A/preds $A/logs $A/work
# 0) read-only inputs of the deployed chain (copies; ~1.3 GB)
cp $X/recs/det0.pkl $X/recs/det1.pkl $A/recs/
for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st2_s5.pt ref_st2_s6.pt inf_st2_det1_train.pkl; do cp $X/ref/$f $A/ref/; done
cp $X/ou/read_tr1_stu.pkl $X/ou/read_val1_stu.pkl $X/ou/read_te1_stu.pkl $A/ou/
cp $X/kps_tr1.pkl $X/kps_val1.pkl $X/kps_te1.pkl $A/
cp $XW/oc_tr1.pkl $XW/oc_val1.pkl $XW/oc_te1.pkl $A/work/          # chain_x.zL_table reads OO.W/oc_*.pkl directly (pkl only; no crop arrays needed)
# 1) omega = 0 stage-3 targets (CPU, < 1 min): same 4,751 crops, corner term dropped
$PY $L/chain_x.py targets 0 > $A/logs/targets_w0.log 2>&1
# 2) stage-3 training, seeds 5 and 6 in parallel (as run_x.sh), 2,000 steps each (~15 min each, concurrent)
for j in 5 6; do ( REF_REAL=d0 REF_DP=$A/ref/dp_fused.npy $SLOT light $PY $L/refiner_x.py train 3 $j > $A/logs/ref_st3_s$j.log 2>&1 ) & sleep 30; done; wait
# 3) inference of both seeds on det1 crops (train for c_k, val for b'_k, test), seed average (~18 min at batch 128; slower at batch 8)
for sp in train val test; do
  for j in 5 6; do $SLOT light $PY $L/refiner_x.py infer $A/ref/ref_st3_s$j.pt $A/recs/det1.pkl $sp $A/ref/fin_s${j}_$sp.pkl >> $A/logs/infer.log 2>&1; done
  $PY E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/avg_inf.py $A/ref/inf_fin_det1_$sp.pkl $A/ref/fin_s5_$sp.pkl $A/ref/fin_s6_$sp.pkl >> $A/logs/infer.log 2>&1
done
# 4) deployment (CPU): deployment fusion unchanged at the frozen omega = 0.190 (also writes the nofuse arm), then score
$PY $L/chain_x.py deploy 0.190282282581008 w0 > $A/logs/deploy.log 2>&1
for p in w0_fused_val w0_nofuse_val; do $PY $L/score_x.py $A/preds/$p.pkl val; done
for p in w0_fused_test w0_nofuse_test; do $PY $L/score_x.py $A/preds/$p.pkl real_test; done   # test: score once
