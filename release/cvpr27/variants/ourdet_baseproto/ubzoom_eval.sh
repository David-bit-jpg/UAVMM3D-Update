#!/bin/bash
# [11-17] Upper-bound row WITH view augmentation (ub_zoom_train.py, config <combo>_ubzoom.yaml, only out2antifs). Same as ub_eval.sh:
# Upper-bound row: real supervised training (no sim, no validation split -> real test never touched during training), epochs set so the
# iteration count matches the 5-epoch sim recipe (MAV6D 7 ep x 1,609 = 11,263 its vs 11,315; OutdoorUAV3D 8 ep x 412 = 3,296 vs 3,355).
C=$1; source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh; cd $H
N=${C}_ubzoom; case $C in in2mav) EP=7 ;; out2antifs) EP=8 ;; esac
CK=$MR/$N/DET_s0/ckpt/checkpoint_epoch_$EP.pth; BS=8
while [ ! -f $CK ]; do
  say "train $N batch $BS epochs $EP (waiting for gpu slot)"
  ( cd /e/Open3DUAVDet/tools; $SLOT light $PY $HW/ub_zoom_train.py --cfg_file cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/$N.yaml \
      --epochs $EP --batch_size $BS --workers 0 --extra_tag DET_s0 --fix_random_seed --seed 0 --skip_test_eval --max_ckpt_save_num 2 >> $H/logs/train_$N.log 2>&1 )
  [ -f $CK ] && break
  if tail -300 logs/train_$N.log | grep -qi "out of memory" && [ $BS -gt 1 ]; then BS=$((BS / 2)); say "!! OOM -> batch $BS, resume (RECIPE DEVIATION)"; continue; fi
  if tail -300 logs/train_$N.log | grep -q "Unable to allocate\|MemoryError\|Segmentation fault"; then say "!! host RAM error -> retry/resume in 2 min"; sleep 120; continue; fi
  say "!! train $N failed"; exit 2
done
say "trained $N"
[ -f preds/${N}_recs.pkl ] || $SLOT light $PY $HW/lib/ourdet_infer.py --ckpt $CK --splits test --out $HW/preds/${N}_recs.pkl --pred-test $HW/preds/${N}_test.pkl > logs/${N}_infer.log 2>&1 || { say "!! infer $N"; exit 2; }
[ -f preds/${N}_test.pkl.score_$SPLIT.json ] || $PY /e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py $HW/preds/${N}_test.pkl $SPLIT > logs/${N}_score.log 2>&1 || { say "!! score $N"; exit 2; }
say "$N: $(head -1 logs/${N}_score.log)"
