#!/bin/bash
# [review1113 stage1seeds] train the RE-SEEDED stage-1 (simulation-only) networks of a source. NOT RUN by its author.
#   usage: bash train_stage1.sh <src in|out> <sd 30|40> [item ...]      items (default = all, in this order):
#          det:0 det:1 det:2 rgb:0 rgb:1 irT:0 kdir:0 kdir:1 irlid:0 irlid:1 kps:0 lid:0 lid:1      (seed = sd + k)
#   DRY=1: print every item's exact command, check cfg / trainer / dependency files and that the new run dir does not exist yet; no GPU.
# Recipes, paths, tags: s1paths.py (= the reported networks' train-log args; cfgs verified unchanged by check_cfg_drift.py).
#   kdir waits for this seed set's irT (its IR teacher); kps waits for this seed set's irlid:0 (its init). Indoor det / irlid / lid then get the
#   chain's checkpoint format (release/in_sd<sd>/ramp_s*.pth model_state only; kzm_s*/kzl_s*.pth aux heads stripped by lidar/irlid/strip_aux.py).
# Writes ONLY: D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm/<cfg stem>/S1X_*_s<seed>/ (new train.py run dirs via s1_train.py; E: is
#   almost full), review1113/stage1seeds/{release,logs,stage1}/.
# Robustness: every training via gpu_slot.sh heavy (DataLoader workers 2 = the recipe; S1_WORKERS=0 -> light slot, logged as deviation);
#   train.py resumes from the newest checkpoint_epoch_*.pth by itself; CUDA OOM -> --batch_size halved and resumed (max 3; RECIPE DEVIATION in
#   logs/stage1_deviations.log); other failure -> one resume retry, then marker stage1/<src>_sd<sd>/<item>.failed; an item whose final
#   checkpoint exists is skipped (.done marker). LATEST_START=MMDDHHMM: do not START a training after that machine time (default 11231200).
#   Never kills processes.
SRC=$1; SD=$2; shift 2; ITEMS="$*"; DRY=${DRY:-0}
case "$SRC/$SD" in in/30|in/40|out/30|out/40) ;; *) echo "usage: train_stage1.sh <in|out> <30|40> [item...]"; exit 2 ;; esac
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=D:/Miniconda3/envs/city/python.exe
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; S1=$GP/review1113/stage1seeds; S1W=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds
SLOT="bash $GP/gpu_slot.sh"; WORKERS=${S1_WORKERS:-2}; KIND=heavy; [ "$WORKERS" = 0 ] && KIND=light
MK=$S1/stage1/${SRC}_sd$SD; mkdir -p $MK $S1/logs
[ -n "$ITEMS" ] || ITEMS=$($PY $S1W/s1paths.py items)
LOG=$S1/logs/stage1_${SRC}_sd$SD.log; DEV=$S1/logs/stage1_deviations.log
[ $DRY = 1 ] && LOG=/dev/null
say() { echo "$(date '+%m-%d %H:%M:%S') [stage1 $SRC sd$SD] $1" | tee -a $LOG; }
pp() { echo "$1" | sed -E 's|^([A-Za-z]):/|/\L\1/|'; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }

# indoor chain format (CPU): det -> model_state only (+epoch, src) = release_20260920; irlid/lid -> aux heads stripped = lidar/irlid/release
release() {
  [ -f "$(pp $REL)" ] && return 0
  if [ "$RELDET" = 1 ]; then
    $PY -c "
import os, sys, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False); os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
torch.save({'model_state': a['model_state'], 'epoch': a.get('epoch'), 'src': sys.argv[1]}, sys.argv[2]); print('release', sys.argv[2], len(a['model_state']), 'tensors')" "$CKPT" "$REL" >> $LOG 2>&1
  elif [ "$STRIP" = 1 ]; then
    $PY E:/Open3DUAVDet/output/camnorm/geo1004/lidar/irlid/strip_aux.py "$CKPT" "$REL" >> $LOG 2>&1
  fi
  [ "$REL" = "$CKPT" ] || [ -f "$(pp $REL)" ]
}

for IT in $ITEMS; do
  eval "$($PY $S1W/s1paths.py sh $SRC $SD $IT)" || { echo "bad item $IT"; exit 2; }
  NM=${IT/:/}
  if [ $DRY = 1 ]; then
    echo "  == $SRC sd$SD $IT  seed $SEED  tag $TAG  (recipe of $(basename $REFRUN))"
    echo "     cd E:/Open3DUAVDet/tools && $PY $S1W/s1_train.py $TRAINER --cfg_file $CFG $FLAGS"
    echo "     -> $CKPT   chain uses: $REL"
    [ -f /e/Open3DUAVDet/tools/$CFG ] && echo "     cfg ok" || { echo "     !! cfg missing /e/Open3DUAVDet/tools/$CFG"; exit 2; }
    [ -f "$(pp $TRAINER)" ] && echo "     trainer ok" || { echo "     !! trainer missing $TRAINER"; exit 2; }
    [ -n "$DEP" ] && echo "     depends on $DEP (trained earlier in this seed set; exists now: $([ -f "$(pp $DEP)" ] && echo yes || echo no))"
    [ -d "$(pp $RUNDIR)" ] && echo "     NOTE run dir exists already (resume / done): $RUNDIR" || echo "     new run dir (does not exist yet): $RUNDIR"
    continue
  fi
  if [ -f "$(pp $CKPT)" ]; then release || { say "!! release copy $IT failed"; touch $MK/$NM.failed; continue; }; touch $MK/$NM.done; say "skip $IT (exists: $CKPT)"; continue; fi
  LS=${LATEST_START:-11231200}; NOW=$(date '+%m%d%H%M')
  if [ $((10#$NOW)) -gt $((10#$LS)) ] && [ ! -d "$(pp $CKDIR)" ]; then say "SKIPPED $IT: machine time $NOW past latest start $LS"; touch $MK/$NM.skipped; continue; fi
  if [ -n "$DEP" ]; then n=0; dfail=0; until [ -f "$(pp $DEP)" ]; do
      if [ -f $MK/$DEPITEM.failed ] || [ -f $MK/$DEPITEM.skipped ]; then dfail=1; break; fi
      [ $n = 0 ] && say "$IT waiting for its dependency $DEPITEM ($DEP)"; n=1; sleep 300; done
    [ $dfail = 1 ] && { say "!! $IT not trained: dependency $DEPITEM failed/skipped"; touch $MK/$NM.failed; continue; }; fi
  rm -f $MK/$NM.failed $MK/$NM.skipped
  [ "$WORKERS" = 2 ] || echo "$(date '+%m-%d %H:%M:%S') $SRC sd$SD $IT: DataLoader workers $WORKERS instead of 2 (RECIPE DEVIATION: augmentation stream only)" >> $DEV
  BS=8; try=0; nf=0; lg=$S1/logs/train_${SRC}_sd${SD}_$NM.log
  while [ ! -f "$(pp $CKPT)" ]; do
    FL=$(echo "$FLAGS" | sed "s/--batch_size 8/--batch_size $BS/")
    say "train $IT (seed $SEED, tag $TAG, batch $BS, workers $WORKERS, slot $KIND; resumes from the newest epoch checkpoint if any)"
    ( cd /e/Open3DUAVDet/tools && $SLOT $KIND $PY $S1W/s1_train.py $TRAINER --cfg_file $CFG $FL ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f "$(pp $CKPT)" ] && break
    if isoom $lg.try && [ $try -lt 3 ] && [ $BS -gt 1 ]; then try=$((try + 1)); BS=$((BS / 2))
      echo "$(date '+%m-%d %H:%M:%S') $SRC sd$SD $IT OOM -> retry $try/3 with --batch_size $BS, resuming from the last epoch checkpoint (RECIPE DEVIATION)" | tee -a $DEV >> $LOG
    elif [ $nf -lt 1 ]; then nf=1; say "$IT failed (not OOM) -> one resume retry, see $lg"; sleep 60
    else say "!! $IT FAILED, see $lg"; touch $MK/$NM.failed; break; fi
  done; rm -f $lg.try
  [ -f "$(pp $CKPT)" ] || continue
  release || { say "!! release copy $IT failed"; touch $MK/$NM.failed; continue; }
  touch $MK/$NM.done; say "$IT done -> $REL"
done
[ $DRY = 1 ] && exit 0
say "end of item list ($ITEMS); failed: $(ls $MK/*.failed 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
