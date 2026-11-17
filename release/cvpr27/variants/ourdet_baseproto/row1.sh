#!/bin/bash
# row 1 = seed-0 sim-only detector alone (no ensemble, no flip, predicted size) on the target test set, scored once; the same pass on the
# real train frames feeds the +ST pseudo labels (index_st.pkl, top-1 box, top 80 % by score).
C=$1; source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh; cd $H
R=$HW/preds/${C}_sim_s0_recs.pkl; RT=$HW/preds/${C}_sim_s0_train_recs.pkl
[ -f preds/${C}_sim_s0_train_recs.pkl ] || $SLOT light $PY $HW/lib/ourdet_infer.py --ckpt $DET0 --splits train --data-path $TRAIN_COPY --out $RT > logs/${C}_row1_train_infer.log 2>&1 || { say "!! train infer"; exit 2; }
[ -f labels/${C}_st.json ] || $PY $HW/lib/make_labels.py st $C $RT >> logs/${C}_row1_train_infer.log 2>&1 || { say "!! st labels"; exit 2; }
say "st labels: $(cat labels/${C}_st.json | tr -d '\n ' | cut -c1-200)"
[ -f preds/${C}_sim_s0_recs.pkl ] || $SLOT light $PY $HW/lib/ourdet_infer.py --ckpt $DET0 --splits test --out $R --pred-test $HW/preds/${C}_sim_s0_test.pkl > logs/${C}_row1_infer.log 2>&1 || { say "!! test infer"; exit 2; }
[ -f preds/${C}_sim_s0_test.pkl.score_$SPLIT.json ] || $PY /e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py $HW/preds/${C}_sim_s0_test.pkl $SPLIT > logs/${C}_sim_score.log 2>&1 || { say "!! score"; exit 2; }
say "row1 sim-only s0: $(head -1 logs/${C}_sim_score.log)"
