#!/bin/bash
# [review1113 rgbst_equal] GUPNet + "RGB self-training under equal conditions" on MAV6D (in -> mav), seed 0.
# = baseline pack recipe for the st/kd step (scripts/run.sh train(): official GUPNet, ImageNet DLA-34, 140 epochs x 3712 samples, sim_train
# labels + pseudo.pkl), pseudo.pkl = lib/make_gupnet_naive.py (same 5,227 frames / boxes as index_naive.pkl).
# Local deviation: batch 8 (lr scaled linearly by train_uav.py, as the local sim-only gupnet_s0), server main-table runs used batch 16.
# Starts only after run.sh (CenterNet rows) has finished -> at most one training of ours at a time. Resumable (official resume from last ckpt).
R=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home GUPNET_DLA34_WEIGHTS=E:/torch_home/hub/checkpoints/dla34-ba72cf86.pth
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
G=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet; O=D:/S2MUAV3D/rgbst_equal_models/gupnet_in_naivest_mav_s0
say() { echo "$(date '+%m-%d %H:%M:%S') [gupnet] $1" | tee -a $R/logs/run.log; }
until grep -q "ALL DONE" $R/logs/run.log 2>/dev/null; do sleep 600; done
[ -f $O/pseudo.pkl ] || $PY $R/lib/make_gupnet_naive.py $O/pseudo.pkl >> $R/logs/gupnet.log 2>&1 || { say "!! pseudo"; exit 2; }
CK=$O/checkpoints/checkpoint_final.pth; BS=8
while [ ! -f $CK ]; do
  RES=$(ls -v $O/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | tail -1); [ -n "$RES" ] && RES="--resume $RES"
  say "train GUPNet naive-ST batch $BS $RES (waiting for gpu slot)"
  $SLOT heavy bash $G/train.sh --labels D:/cmpfull/sim_train/labels.pkl $O/pseudo.pkl --out $O --seed 0 --epochs 140 --batch-size $BS $RES >> $O/train.log 2>&1
  [ -f $CK ] && break
  if tail -200 $O/train.log | grep -qi "out of memory" && [ $BS -gt 1 ]; then BS=$((BS / 2)); say "!! OOM -> batch $BS (RECIPE DEVIATION)"; continue; fi
  if tail -200 $O/train.log | grep -q "Unable to allocate\|MemoryError\|paging file"; then say "!! host RAM error -> resume in 2 min"; sleep 120; continue; fi
  say "!! GUPNet train failed, see $O/train.log"; exit 2
done
say "GUPNet trained"
P=$O/pred_real_test.pkl
[ -f $P ] || ( cd $G && $SLOT light $PY predict.py --ckpt $CK --split real_test --out $P --thresh 0 ) >> $R/logs/gupnet.log 2>&1 || { say "!! predict"; exit 2; }
cp $P $R/preds/gupnet_in_naivest_mav_s0_pred_real_test.pkl
[ -f $R/logs/gupnet_score.log ] || $PY $R/lib/score_gupnet.py $P real_test > $R/logs/gupnet_score.log 2>&1 || { say "!! score"; exit 2; }
say "GUPNet naive-ST MAV6D scored: see logs/gupnet_score.log"
