#!/bin/bash
# [xdomain_1117 A] cross-target transfer without re-adaptation (inference only). Same inference/scoring code as ourdet_baseproto
# (lib/ourdet_infer.py unchanged: single model, no flip, predicted size, top-1 box per frame); only SX_SRC/SX_TGT change, so the decode config
# is srcx real_<training source>2<TEST target>.yaml = test target's real data config (its cache, input norm, intrinsics) + the TRAINING SOURCE's
# decoder statistics (DEPTH/SIZE mean/std the network was trained with). Each cell scored once on the test target (score_x.py).
# usage: bash run_A.sh   (skips finished cells; re-run resumes)
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; XD=$X/review1113/xdomain_1117; XW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/xdomain_1117
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home UAV_PNP_EPNP_FALLBACK=1 SX_TEACH=full
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $X/gpu_slot.sh"; INF=$X/review1113/ourdet_baseproto/lib/ourdet_infer.py
MR=D:/S2MUAV3D/ourdet_baseproto_models/models/uavdet_3d/camnorm/ourdet_baseproto
say() { echo "$(date '+%m-%d %H:%M:%S') [A] $1" | tee -a $XD/logs/run.log; }
# name | training source (decoder stats) | test target | checkpoint
CELLS="in2mav_pst_ON_antifs|in|antifs|$MR/in2mav_pst/DET_s0/ckpt/checkpoint_epoch_5.pth
out2antifs_pst_ON_mav|out|mav|$MR/out2antifs_pst/DET_s0/ckpt/checkpoint_epoch_5.pth
in_sim_s0_ON_antifs|in|antifs|E:/Open3DUAVDet/output/release_20260920/ramp_s0.pth
out_sim_s0_ON_mav|out|mav|E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out/DET_s0/ckpt/checkpoint_epoch_5.pth"
echo "$CELLS" | while IFS='|' read N S T CK; do
  export SX_SRC=$S SX_TGT=$T; SPLIT=$([ $T = mav ] && echo real_test || echo antifs_test)
  [ -f $XD/preds/${N}_test.pkl ] || $SLOT light $PY $INF --ckpt $CK --splits test --out $XW/preds/${N}_recs.pkl --pred-test $XW/preds/${N}_test.pkl > $XD/logs/${N}_infer.log 2>&1 || { say "!! infer $N"; continue; }
  [ -f $XD/preds/${N}_test.pkl.score_$SPLIT.json ] || $PY $X/srcx/lib/score_x.py $XW/preds/${N}_test.pkl $SPLIT > $XD/logs/${N}_score.log 2>&1 || { say "!! score $N"; continue; }
  say "$N (cfg srcx/real_${S}2${T}.yaml): $(head -1 $XD/logs/${N}_score.log)"
done
say "A done"
