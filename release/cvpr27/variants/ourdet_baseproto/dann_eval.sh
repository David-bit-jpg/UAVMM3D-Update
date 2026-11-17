#!/bin/bash
# CenterNet + DANN row: ourdet_dann_train.py (protocol of the GUPNet + DANN baseline) -> single model, no flip, predicted size -> scored once.
C=$1; source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh; cd $H
N=${C}_dann; O=D:/S2MUAV3D/ourdet_baseproto_models/dann/$N; CK=$O/ckpt/checkpoint_final.pth; BS=8
while [ ! -f $CK ]; do
  say "train $N batch $BS (waiting for gpu slot)"
  $SLOT light $PY $HW/ourdet_dann_train.py --combo $C --out $O --batch $BS >> logs/train_$N.log 2>&1
  [ -f $CK ] && break
  if tail -300 logs/train_$N.log | grep -qi "out of memory" && [ $BS -gt 1 ]; then BS=$((BS / 2)); say "!! OOM -> batch $BS, resume (RECIPE DEVIATION)"; continue; fi
  if tail -300 logs/train_$N.log | grep -q "Unable to allocate\|MemoryError\|Segmentation fault"; then say "!! host RAM error -> retry/resume in 2 min"; sleep 120; continue; fi
  say "!! train $N failed"; exit 2
done
say "trained $N"
[ -f preds/${N}_recs.pkl ] || $SLOT light $PY $HW/lib/ourdet_infer.py --ckpt $CK --splits test --out $HW/preds/${N}_recs.pkl --pred-test $HW/preds/${N}_test.pkl > logs/${N}_infer.log 2>&1 || { say "!! infer $N"; exit 2; }
[ -f preds/${N}_test.pkl.score_$SPLIT.json ] || $PY /e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py $HW/preds/${N}_test.pkl $SPLIT > logs/${N}_score.log 2>&1 || { say "!! score $N"; exit 2; }
say "$N: $(head -1 logs/${N}_score.log)"
