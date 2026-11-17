#!/bin/bash
# [review1113 seeds] ONE seed-shifted run of the clean chain (= srcx/run_x.sh <src> <tgt> with SX_TEACH=<teach>), generalising
# review1113/sup3/run_seed2.sh (which is in/mav/full only) to parameters. NOT RUN by its author; launched by run_all_seeds.sh.
#   usage: bash run_seed_chain.sh <teach full|none> <src in|out> <tgt mav|antifs> <shift>          (DRY=1: print the config and exit)
# Seed shift (as sup3): SX_SEEDSHIFT=<shift> is added to the seed of EVERY component run_x.sh trains:
#   depth refiner stages 1-3 (nominal 5, 6 -> 5+shift, 6+shift), self-trained detectors x3 (train.py --seed k+shift; init = the same fixed
#   sim-detector member), orientation students S6I (nominal 0, 1 -> +shift) -> everything downstream is recomputed.
# KEPT FIXED: the stage-1 networks of the source (sx.SOURCES[src]: sim-only detectors, IR+LiDAR / IR-distilled / LiDAR teachers, height net;
#   for teach=none the 2 plain-RGB orientation nets) are pretrained checkpoints that run_x.sh does not train. Their outputs are therefore
#   identical inputs and are COPIED (COPY_FIXED=1, default) from the teach-matched reported run srcx/<src>2<tgt>[_none]:
#   recs/det0.pkl, preds/srconly_{val,test}.pkl, ref/{sim_meta.pkl, sim_rgb.npy, sim_sil.npy, bank.npy, real_d0_meta.pkl, real_d0_rgb.npy},
#   wsim_cache.pkl. COPY_FIXED=0 recomputes them with the same code. Non-training data seeds stay at rng 0.
# Teach = none (as run_x.sh): teachers = 2 plain RGB nets (in: sim_full_r34_kp_w12_zoom/KZs0,1; out: sim_out_r34_kp_w12_zoom_out/OKZrgb_s0,1),
#   gate = agreement of the 2 seeds, students init from them, refiner without IR silhouettes (sx.NOIR), constant height aspect of s_bar.
# Writes ONLY: review1113/seeds/<tgt>_<teach>_s<shift>/ (lib, recs, ref, ou, preds, logs), D:/S2MUAV3D/srcx_work/<C>_sd<shift>/ (crops),
#   and new tagged files tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_<C>_sd<shift>.yaml, <target cache>/train/index_pseudo_srcx_<C>_sd<shift>.pkl,
#   output/models/uavdet_3d/camnorm/srcx/st_<C>_sd<shift>/DET1_s{0,1,2}/   (C = <src>2<tgt>[_none]). Nothing existing is overwritten.
# Robustness (= sup3): every training step resumes from its last checkpoint and is skipped once its output exists; CUDA OOM -> the batch is
#   halved and the step retried (max 4; logged to logs/retry.log; a halved TRAINING batch is flagged as a recipe deviation); DataLoader
#   workers 0 everywhere; inference via gpu_slot.sh light + infer_wrap.py (memory fraction 0.2); never kills processes.
#   SERIAL=1 runs the paired trainings (2 refiners, 3 detectors, 2 students) one after another instead of the run_x.sh overlap.
# Log: <run dir>/logs/run.log, last line "ALL DONE: <test metrics> | val <metrics> | ...".
TEACH=$1; SRC=$2; TGT=$3; SHIFT=$4; COPY_FIXED=${COPY_FIXED:-1}; SERIAL=${SERIAL:-0}; DRY=${DRY:-0}
case "$TEACH/$SRC/$TGT" in full/in/mav|none/in/mav|full/out/antifs|none/out/antifs) ;; *) echo "bad args: $*"; exit 2 ;; esac
[ "$SHIFT" -gt 0 ] 2>/dev/null || { echo "bad shift $SHIFT"; exit 2; }
RUN=${TGT}_${TEACH}_s$SHIFT; C=${SRC}2${TGT}; [ $TEACH = full ] || C=${C}_$TEACH; TAGC=${C}_sd$SHIFT
export SX_SEEDSHIFT=$SHIFT SX_TEACH=$TEACH SX_SRC=$SRC SX_TGT=$TGT PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
SD=$GPW/review1113/seeds; X=$GP/review1113/seeds/$RUN; XW=$SD/$RUN; L=$XW/lib; WK=/d/S2MUAV3D/srcx_work/$TAGC; WKW=D:/S2MUAV3D/srcx_work/$TAGC
F=$GP/srcx/$C                                               # teach-matched reported run: read only
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
IW="$PY $L/infer_wrap.py"
case $TGT in mav) REP="test 60.81 / val 59.71"; [ $TEACH = none ] && REP="test 57.38 / val 53.52" ;;
             antifs) REP="test 41.74 / val 34.98"; [ $TEACH = none ] && REP="test 38.92 / val 32.95" ;; esac

if [ $DRY = 1 ]; then   # config print only: builds the lib copy (idempotent, inside the run dir), no makedirs, no GPU, nothing else written
  $PY $SD/make_seeds_lib.py $TEACH $SRC $TGT $SHIFT || exit 2
  SX_NOMKDIR=1 CUDA_VISIBLE_DEVICES="" $PY -c "
import sys, os; sys.path.insert(0, '$L'); import sx; S = sx.S
print('  run %s  combo %s  seedshift %s  copy-fixed-from %s' % ('$RUN', sx.COMBO, os.environ['SX_SEEDSHIFT'], '$F'))
print('  O  =', sx.O); print('  W  =', sx.W); print('  REAL_CFG =', sx.REAL_CFG); print('  ST_CFG   =', sx.ST_CFG)
print('  pseudo index =', '%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG))
print('  det1 ckpts   =', '$M/srcx/%s/DET1_s{0,1,2}' % os.path.basename(sx.ST_CFG)[:-5])
print('  NOIR =', sx.NOIR, ' score split =', sx.T['score_split'])
print('  fixed sim detectors:', [os.path.exists(p) for p in S['det']], ' height net:', os.path.exists(S['kps']))
for p in (sx.O, sx.W, sx.ROOT + '/tools/' + sx.ST_CFG, '%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG)):
    print('  exists already?', os.path.exists(p), p)
assert sx.O.endswith('/seeds/$RUN') and sx.W.endswith('$TAGC') and sx.PSEUDO_TAG == 'srcx_$TAGC' and sx.ST_CFG.endswith('st_$TAGC.yaml')
" || exit 2
  for f in recs/det0.pkl preds/srconly_val.pkl preds/srconly_test.pkl ref/sim_meta.pkl ref/sim_rgb.npy ref/sim_sil.npy ref/bank.npy ref/real_d0_meta.pkl ref/real_d0_rgb.npy wsim_cache.pkl; do
    [ -f $F/$f ] || { echo "  MISSING fixed output $F/$f"; exit 2; }; done
  echo "  fixed stage-1 outputs present in $F (10 files); reported: $REP"; exit 0
fi

mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs $WK; cd $X
LOG=$X/logs/run.log; RLOG=$X/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }; die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
waitf() { local n=0; until [ -f "$1" ]; do [ $n = 0 ] && say "waiting for $1"; n=1; sleep 120; done; }
# oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries; per-attempt output checked, appended to <log>
oomrun() {
  local lg=$1; local -a vals=($2); shift 2; local try=0 kv k v
  while true; do
    if env "${vals[@]}" "$@" > $lg.try 2>&1; then cat $lg.try >> $lg; rm -f $lg.try; return 0; fi
    cat $lg.try >> $lg
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); local -a nv=()
      for kv in "${vals[@]}"; do k=${kv%%=*}; v=${kv#*=}; v=$((v / 2)); [ $v -lt 1 ] && v=1; nv+=("$k=$v"); done; vals=("${nv[@]}")
      echo "$(date '+%m-%d %H:%M:%S') OOM retry $try/4 with ${vals[*]} :: $*" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') FAILED (not OOM or 4 retries used): $*  see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done
}
# par <sleep> fn args...   run_x.sh overlap (background + stagger) unless SERIAL=1
par() { local s=$1; shift; if [ $SERIAL = 1 ]; then "$@"; else "$@" & sleep $s; fi; }
say "start $RUN (combo $C, SX_SEEDSHIFT=$SHIFT, COPY_FIXED=$COPY_FIXED, SERIAL=$SERIAL)"
$PY $SD/make_seeds_lib.py $TEACH $SRC $TGT $SHIFT >> $LOG 2>&1 || die "make_seeds_lib"
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s SPLIT=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.T['score_split']))
assert sx.O.endswith('/seeds/$RUN') and sx.W.endswith('$TAGC') and sx.PSEUDO_TAG == 'srcx_$TAGC', (sx.O, sx.W, sx.PSEUDO_TAG)
")" || die "sx config"
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg

# ---------------- fixed stage-1 outputs (pretrained source networks, not seeded by run_x.sh): copy or recompute
if [ $COPY_FIXED = 1 ]; then
  [ -f recs/det0.pkl ] || cp $F/recs/det0.pkl recs/ || die "copy det0"
  for sp in val test; do [ -f preds/srconly_$sp.pkl ] || cp $F/preds/srconly_$sp.pkl preds/ || die "copy srconly"; done
  for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy; do [ -f ref/$f ] || cp $F/ref/$f ref/ || die "copy ref/$f"; done
  [ -f wsim_cache.pkl ] || cp $F/wsim_cache.pkl . || die "copy wsim_cache"
  say "copied fixed stage-1 outputs from srcx/$C (det0, srconly preds, refiner sim samples / bank / real d0 crops, wsim cache)"
else
  for c in $DET0 $DET1 $DET2; do waitf $c; done
  [ -f recs/det0.pkl ] || oomrun logs/det0_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $DET0 $DET1 $DET2 --splits train val test --out $XW/recs/det0.pkl || die "det0 infer"
  for sp in val test; do [ -f preds/srconly_$sp.pkl ] || $PY $L/to_pred.py $XW/recs/det0.pkl $sp $XW/preds/srconly_$sp.pkl >> $LOG 2>&1; done
  [ -f ref/sim_meta.pkl ] || oomrun logs/ref_sim.log "DET_BS=8" $SLOT light $IW $L/refiner_x.py sim $DET0 || die "ref sim"
  [ -f ref/bank.npy ] || $PY $L/refiner_x.py bank $XW/recs/det0.pkl >> $LOG 2>&1 || die "ref bank"
  [ -f ref/real_d0_meta.pkl ] || $PY $L/refiner_x.py real $XW/recs/det0.pkl d0 >> $LOG 2>&1 || die "ref real"
fi

# ---------------- Module 4 stages 1-2 (nominal seeds 5, 6 -> +SHIFT), resumable, OOM -> halve batch
ref12() { local j=$1
  [ -f ref/ref_st1_s$j.pt ] || oomrun logs/ref_st1_s$j.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $j
  [ -f ref/ref_st1_s$j.pt ] && { [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $j; }; }
for j in 5 6; do par 30 ref12 $j; done; wait
for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done
say "refiners stage 1-2 done"

# ---------------- Module 3: refiner stage-2 corrections on det0 train -> pseudo-labels (tag *_sd<shift>) -> self-trained detectors x3 (seed k+SHIFT)
for j in 5 6; do [ -f ref/st2_s${j}_det0_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det0.pkl train $XW/ref/st2_s${j}_det0_train.pkl || die "inf st2 det0 $j"; done
[ -f ref/inf_st2_det0_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det0_train.pkl $XW/ref/st2_s5_det0_train.pkl $XW/ref/st2_s6_det0_train.pkl >> $LOG 2>&1
[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
cat pseudo_$TAG.json >> $LOG
STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
DETS=($DET0 $DET1 $DET2)
train_det1() {   # $1 = member k; train.py resumes from its latest epoch checkpoint by itself; OOM -> halve --batch_size (recipe deviation)
  local k=$1 bs=8 try=0 lg=$X/logs/train_det1_s$1.log
  while [ ! -f $(stck $k) ]; do
    ( cd /e/Open3DUAVDet/tools; $SLOT light $PY train.py --cfg_file $ST_CFG --batch_size $bs --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
        --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $((k + SHIFT)) --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $(stck $k) ] && break
    if isoom $lg.try && [ $try -lt 4 ] && [ $bs -gt 1 ]; then try=$((try + 1)); bs=$((bs / 2))
      echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k OOM -> retry $try/4 with --batch_size $bs, resuming from the last epoch checkpoint (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k FAILED (not OOM or retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for k in 0 1 2; do par 100 train_det1 $k; done; wait
for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
[ -f recs/det1.pkl ] || oomrun logs/det1_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl || die "det1 infer"
grep -E "s_bar|frames" logs/det1_infer.log | tail -4 | tee -a $LOG
say "detectors done"

# ---------------- Module 5: orientation crops on the final detector, real background bank, teachers (fixed checkpoints; BN on real train crops <= 80/seq)
for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WK/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
[ -f $WK/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
case $TEACH in
  full) TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; INIT=($IRL0 $IRL1) ;;
  none) if [ $SRC = in ]; then RG=$M/sim_full_r34_kp_w12_zoom; RG0=$RG/KZs0/ckpt/checkpoint_epoch_30.pth; RG1=$RG/KZs1/ckpt/checkpoint_epoch_30.pth
        else RG=$M/sim_out_r34_kp_w12_zoom_out; RG0=$RG/OKZrgb_s0/ckpt/checkpoint_epoch_30.pth; RG1=$RG/OKZrgb_s1/ckpt/checkpoint_epoch_30.pth; fi
        TL="rgb0 $RG0 rgb1 $RG1"; INIT=($RG0 $RG1) ;;
esac
set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); waitf $2; [ -f ou/T_$1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 || die "adabn $1"; shift 2; done
T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
[ -f ou/read_tr1_teach.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 || die "read teachers"
q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
if [ $TEACH = full ]; then GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
else GRP="{\"T\": [$(q ${TN[0]}), $(q ${TN[1]})], \"IR\": [$(q ${TN[0]})], \"LiD\": [$(q ${TN[1]})]}"; fi   # single modality: gate = agreement of its 2 seeds
[ -f $WK/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
grep "pseudo pseudo_tr1" $LOG | tail -1
# students S6I nominal seeds 0, 1 (effective +SHIFT; init = same-index teacher member; gated pseudo-rotations + mirror loss 12, 2250 steps)
train_student() {   # $1 = seed; resume from runs/S6I_s<seed>/resume.pth; OOM -> halve --bs-src/--bs-tgt (recipe deviation)
  local s=$1 bs=8 bt=4 try=0 lg=$X/logs/ou_train_s$1.log
  while [ ! -f ou/runs/S6I_s$s/ckpt_last.pth ]; do
    OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $XW/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f ou/runs/S6I_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for s in 0 1; do par 100 train_student $s; done; wait
for s in 0 1; do [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || die "student s$s"
  [ -f ou/S_ab$s.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 || die "student adabn $s"; done
for n in tr1 val1 te1; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth || die "read $n"; done
if [ $TEACH = full ]; then   # height network (fixed checkpoint, kps seed 0): BN on train crops (<= 80/seq), readings
  waitf $KPS
  [ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
else   # no height network (it is IR+LiDAR trained): every crop gets the constant aspect h / mean(l, w) of s_bar (as run_x.sh, crops from THIS run's W)
  [ -f kps_te1.pkl ] || $PY -c "
import pickle, numpy as np
s = np.asarray(pickle.load(open('$XW/recs/det1.pkl', 'rb'))['size'], float); r = float(s[2] / ((s[0] + s[1]) / 2))
for n in ('tr1', 'val1', 'te1'):
    I = pickle.load(open('$WKW/oc_%s.pkl' % n, 'rb'))['info']
    pickle.dump({(f['seq'], f['frame']): r for f in I}, open('$XW/kps_%s.pkl' % n, 'wb'))
print('constant height aspect', r)" >> $LOG 2>&1 || die "const kps"
fi
say "orientation branch done"

# ---------------- Module 4 stage 3 with fused pseudo-targets; w by inverse variance on SIM validation
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1
[ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
WW=$($PY -c "import json;print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
ref3() { local j=$1; [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j; }
for j in 5 6; do par 30 ref3 $j; done; wait
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1
done

# ---------------- Module 6: constants (unlabeled target images) + single-frame deployment; scoring (val = sanity; test = once)
[ -f deploy_pcmd.json ] || $PY $L/chain_x.py deploy $WW pcmd >> $LOG 2>&1 || die deploy
for p in srconly_val pcmd_fused_val pcmd_nofuse_val; do [ -f preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/$p.pkl val >> $LOG 2>&1; done
for p in srconly_test pcmd_fused_test; do [ -f preds/$p.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/$p.pkl $SPLIT >> $LOG 2>&1 || die "score $p"; done
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
fm = lambda d: 'UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f' % (d['UDS'], d['AP'], d['ATE'], d['ASE'], d['AOE'], d['mDR'])
t = f('$XW/preds/pcmd_fused_test.pkl.score_$SPLIT.json'); v = f('$XW/preds/pcmd_fused_val.pkl.score_val.json')
print('test %s | val %s | $RUN w %.4f | reported (shift 0): $REP' % (fm(t), fm(v), $WW))") || die "summary"
say "ALL DONE: $SUM"
