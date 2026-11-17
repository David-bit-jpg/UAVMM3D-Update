#!/bin/bash
# +PST / +ST row: our sim-only recipe (stage1_det.sh flags; workers 0 via the light slot) on sim + index_<L>.pkl, seed 0, last checkpoint
# (epoch 5) -> single model, no flip, predicted size on the target test set -> scored once.
C=$1; L=$2; source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh; cd $H
N=${C}_$L; CK=$MR/$N/DET_s0/ckpt/checkpoint_epoch_5.pth; BS=8
until [ -f labels/${C}_$L.json ]; do sleep 120; done
while [ ! -f $CK ]; do
  say "train $N batch $BS (waiting for gpu slot)"
  ( cd /e/Open3DUAVDet/tools; $SLOT light $PY $HW/ourdet_train.py --cfg_file cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/$N.yaml \
      --epochs 5 --batch_size $BS --workers 0 --extra_tag DET_s0 --fix_random_seed --seed 0 --val_split test --val_interval 10 --val_every 100 \
      --skip_test_eval --max_ckpt_save_num 2 >> $H/logs/train_$N.log 2>&1 )
  [ -f $CK ] && break
  if tail -300 logs/train_$N.log | grep -qi "out of memory" && [ $BS -gt 1 ]; then BS=$((BS / 2)); say "!! OOM -> batch $BS, resume (RECIPE DEVIATION)"; continue; fi
  if tail -300 logs/train_$N.log | grep -q "Unable to allocate\|MemoryError\|Segmentation fault"; then say "!! host RAM error -> retry/resume in 2 min"; sleep 120; continue; fi
  say "!! train $N failed"; exit 2
done
say "trained $N"
[ -f preds/${N}_recs.pkl ] || $SLOT light $PY $HW/lib/ourdet_infer.py --ckpt $CK --splits test --out $HW/preds/${N}_recs.pkl --pred-test $HW/preds/${N}_test.pkl > logs/${N}_infer.log 2>&1 || { say "!! infer $N"; exit 2; }
[ -f preds/${N}_test.pkl.score_$SPLIT.json ] || $PY /e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py $HW/preds/${N}_test.pkl $SPLIT > logs/${N}_score.log 2>&1 || { say "!! score $N"; exit 2; }
say "$N: $(head -1 logs/${N}_score.log)"
