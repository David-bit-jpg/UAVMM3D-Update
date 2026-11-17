#!/bin/bash
# [review1113 rgbst_equal] "RGB self-training under equal conditions" = CenterNet +ST row of ourdet_baseproto, pseudo labels replaced by
# PST step-1 pseudo boxes WITHOUT teacher correction (3-detector ensemble + flip, conf >= 0.5, size s_bar, own depth and rotation;
# lib/make_naive_index.py -> index_naive.pkl). Training/eval = ourdet_baseproto/train_eval.sh unchanged (ImageNet init, sim + pseudo, 5 epochs,
# batch 8, seed 0, last ckpt; single model, no flip, predicted size; score_x.py once on test). One training at a time (sequential).
# Resumable: each step skipped once its output exists; train.py resumes from its last epoch ckpt.
# Uses wrapper entry points (rgbst_train.py, lib/rgbst_det_infer.py) so gpu_slot.sh treats this as a normal (non-urgent) job: it waits for
# gproto/.prio_hold to disappear.
R=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal; RW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal
export OD_EXPROOT=D:/S2MUAV3D/rgbst_equal_models
for C in ${COMBOS:-in2mav out2antifs}; do
  source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh
  export OD_EXPROOT=D:/S2MUAV3D/rgbst_equal_models
  say() { echo "$(date '+%m-%d %H:%M:%S') [$C] $1" | tee -a $R/logs/run.log; }
  cd $R; N=${C}_naivest; CK=$OD_EXPROOT/models/uavdet_3d/camnorm/rgbst_equal/$N/DET_s0/ckpt/checkpoint_epoch_5.pth; BS=8
  [ -f labels/${C}_naive.json ] || { say "!! labels missing"; exit 2; }
  while [ ! -f $CK ]; do
    say "train $N batch $BS (waiting for gpu slot)"
    ( cd /e/Open3DUAVDet/tools; $SLOT light $PY $RW/rgbst_train.py --cfg_file cfgs/models/uavdet_3d/camnorm/rgbst_equal/$N.yaml \
        --epochs 5 --batch_size $BS --workers 0 --extra_tag DET_s0 --fix_random_seed --seed 0 --val_split test --val_interval 10 --val_every 100 \
        --skip_test_eval --max_ckpt_save_num 2 >> $R/logs/train_$N.log 2>&1 )
    [ -f $CK ] && break
    if tail -300 logs/train_$N.log | grep -qi "out of memory" && [ $BS -gt 1 ]; then BS=$((BS / 2)); say "!! OOM -> batch $BS, resume (RECIPE DEVIATION)"; continue; fi
    if tail -300 logs/train_$N.log | grep -q "Unable to allocate\|MemoryError\|Segmentation fault\|paging file"; then say "!! host RAM error -> retry/resume in 2 min"; sleep 120; continue; fi
    say "!! train $N failed"; exit 2
  done
  say "trained $N ($CK)"
  [ -f preds/${N}_recs.pkl ] || $SLOT light $PY $RW/lib/rgbst_det_infer.py --ckpt $CK --splits test --out $RW/preds/${N}_recs.pkl --pred-test $RW/preds/${N}_test.pkl > logs/${N}_infer.log 2>&1 || { say "!! infer $N"; exit 2; }
  [ -f preds/${N}_test.pkl.score_$SPLIT.json ] || $PY /e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py $RW/preds/${N}_test.pkl $SPLIT > logs/${N}_score.log 2>&1 || { say "!! score $N"; exit 2; }
  say "$N: $(head -1 logs/${N}_score.log)"
done
say "ALL DONE"
