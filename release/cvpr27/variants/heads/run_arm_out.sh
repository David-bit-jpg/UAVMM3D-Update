#!/bin/bash
# [11-17] COPY of run_arm.sh with ORI (H4/H5) + HGT (H6) ported to antifs (outdoor source). run_arm.sh itself is left byte-identical
#   because a live instance (H1 mav 10) reads it incrementally. Only the 2 marked lines differ; MAV6D paths are unchanged.
# [review1113 heads] ONE per-head ablation run of the clean PST chain (srcx/run_x.sh, SX_TEACH=full). NOT RUN by its author (queued).
#   usage: bash run_arm.sh <arm H1|H2|H3|H4|H5|H6|H1c|H7> <tgt mav|antifs> <shift 0|10|20>
#   env:   DEPS="H4/mav_s0 ..."   wait until each listed run dir has .done / .failed / .skipped (sequencing of the queue lines)
#          HEADS_NO_WAIT=1        skip the paper-critical start gate (plumbing only)     DRY=1   build the lib + config print, nothing else
#          LATEST_START=MMDDHHMM  do not START after this machine time (default: 11221800 for the full-chain arms, 11230800 for the others)
# Arms (one factor each vs the same-seed reported run; details in HEADS_PLAN.md and make_heads_lib.py):
#   FULL H1 refiner silhouette loss off (composites on) | H2 sil_feats not fed to the distance head | H3 no silhouette composites
#        -> the whole downstream chain is recomputed (= review1113/seeds/run_seed_chain.sh, teach full): refiner st1-2 -> pseudo-labels ->
#           3 self-trained detectors -> det1 -> orientation crops / teachers / students -> w (sim val) -> stage-3 targets -> stage 3 -> deploy.
#           Fixed stage-1 outputs (pretrained source nets, not trained by run_x.sh) are copied from srcx/<C>: det0, refiner sim samples,
#           real bank, real d0 crops, wsim cache (the large .npy as NTFS hard links, never written).
#   ORI  H4 LiDAR nets (kzl) -> same-seed plain KZ nets in the teacher group | H5 IR+LiDAR nets (kzm) -> same-seed KZ-LiD nets (= kzl, also
#           as the student init) -> teacher readings, gated pseudo-orientations, students S6I, AdaBN, mirror readings, deployment at the
#           reference run's FROZEN w with its frozen stage-3 depth students (= the review1113/initabl protocol).
#   HGT  H6 height ratio from the height net's box-corner keypoints (kpsbox_x.py) instead of its LiDAR structure points -> deployment.
#   DEP  H1c deploy-only diagnostic: reference run, one colour class.        BH  H7 box-hull silhouette (review1113/boxhull) at shift 10 / 20.
# Real labels: only TEST labels, only via score_x.py (once per run, skipped if the score file exists) and heads_boot.py. No val scoring.
# Robustness (= seeds/run_seed_chain.sh): waits for the paper-critical runs (initabl/.done and cmpfull gupnet_uda_mtstrict_s0/.done, poll 10 min);
#   every GPU call through gpu_slot.sh light (inference via infer_wrap.py, memory fraction 0.2); CUDA OOM -> batch halved and retried (max 4,
#   logs/retry.log, a halved TRAINING batch = RECIPE DEVIATION); trainings resume; every step skipped once its output exists; workers 0;
#   never kills processes. Markers: <run>/.done (success), .failed, .skipped (deadline). Log <run>/logs/run.log + heads/logs/heads.log.
ARM=$1; TGT=$2; SHIFT=$3; DRY=${DRY:-0}
case "$ARM" in H1|H2|H3) MODE=FULL ;; H4|H5) MODE=ORI ;; H6) MODE=HGT ;; H1c) MODE=DEP ;; H7) MODE=BH ;; *) echo "bad arm $ARM"; exit 2 ;; esac
case "$TGT" in mav) SRC=in; SPLIT=real_test ;; antifs) SRC=out; SPLIT=antifs_test ;; *) echo "bad tgt $TGT"; exit 2 ;; esac
case "$SHIFT" in 0|10|20) ;; *) echo "bad shift $SHIFT"; exit 2 ;; esac
[ $MODE = FULL ] || [ $MODE = ORI ] || [ $MODE = HGT ] || [ $TGT = mav ] || { echo "$ARM is prepared for MAV6D only"; exit 2; }   # [11-17] ORI + HGT ported to antifs
[ $MODE = BH ] && [ $SHIFT = 0 ] && { echo "H7 shift 0 = the existing review1113/boxhull/in2mav run"; exit 2; }
C=${SRC}2${TGT}; RUN=${TGT}_s$SHIFT; PFX=$(echo $ARM | tr 'A-Z' 'a-z')
export SX_SEEDSHIFT=$SHIFT SX_TEACH=full SX_SRC=$SRC SX_TGT=$TGT PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset OU_INIT OU_BANK OU_CROPS REF_TAG
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
HD=$GP/review1113/heads; HDW=$GPW/review1113/heads
X=$HD/$ARM/$RUN; XW=$HDW/$ARM/$RUN; L=$XW/lib
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm; IW="$PY $L/infer_wrap.py"
F=$GP/srcx/$C; FW=$GPW/srcx/$C                                   # reported run (shift 0): fixed stage-1 outputs
case "$TGT/$SHIFT" in
  mav/0)     R=$GP/srcx/in2mav;                 RWK=/d/S2MUAV3D/srcx_work/in2mav ;;
  mav/10)    R=$GP/review1113/sup3/run;         RWK=/d/S2MUAV3D/srcx_work/in2mav_seed2 ;;
  mav/20)    R=$GP/review1113/seeds/mav_full_s20; RWK=/d/S2MUAV3D/srcx_work/in2mav_sd20 ;;
  antifs/0)  R=$GP/srcx/out2antifs;             RWK=/d/S2MUAV3D/srcx_work/out2antifs ;;
  antifs/10) R=$GP/review1113/seeds/antifs_full_s10; RWK=/d/S2MUAV3D/srcx_work/out2antifs_sd10 ;;
  antifs/20) R=$GP/review1113/seeds/antifs_full_s20; RWK=/d/S2MUAV3D/srcx_work/out2antifs_sd20 ;;
esac
wp() { echo "$1" | sed -E 's|^/([a-z])/|\U\1:/|'; }               # /e/x -> E:/x (for Python)
RW=$(wp $R)
mkdir -p $HD/logs $X/logs; GLOG=$HD/logs/heads.log; LOG=$X/logs/run.log; RLOG=$X/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') [$ARM $RUN] $1" | tee -a $LOG >> $GLOG; }
die() { say "!! $1"; touch $X/.failed; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
waitf() { local n=0; until [ -f "$1" ]; do [ $n = 0 ] && say "waiting for $1"; n=1; sleep 120; done; }
# oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries (= seeds/run_seed_chain.sh)
oomrun() {
  local lg=$1; local -a vals=($2); shift 2; local try=0 kv k v
  while true; do
    if env "${vals[@]}" "$@" > $lg.try 2>&1; then cat $lg.try >> $lg; rm -f $lg.try; return 0; fi
    cat $lg.try >> $lg
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); local -a nv=()
      for kv in "${vals[@]}"; do k=${kv%%=*}; v=${kv#*=}; v=$((v / 2)); [ $v -lt 1 ] && v=1; nv+=("$k=$v"); done; vals=("${nv[@]}")
      echo "$(date '+%m-%d %H:%M:%S') OOM retry $try/4 with ${vals[*]} :: $* (a halved TRAINING batch = RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') FAILED (not OOM or 4 retries used): $*  see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done
}
lnk() { [ -f "$2" ] && return 0; $PY -c "import os, sys; os.link(sys.argv[1], sys.argv[2])" "$(wp $1)" "$(wp $2)" 2>/dev/null || cp "$1" "$2"; }   # hard link (same volume), else copy
cpf() { [ -f "$2" ] || cp "$1" "$2" || die "copy $1"; }

# ---------------- dry config print
if [ $DRY = 1 ]; then
  $PY $HDW/make_heads_lib.py $ARM $TGT $SHIFT || exit 2
  SX_NOMKDIR=1 CUDA_VISIBLE_DEVICES="" $PY -c "
import sys; sys.path.insert(0, '$L'); import sx
print('  %s %s shift %s mode %s | O %s | W %s | ONECLASS %s' % ('$ARM', '$TGT', '$SHIFT', '$MODE', sx.O, sx.W, sx.ONECLASS))
print('  REAL_CFG %s | ST_CFG %s | PSEUDO_TAG %s | reference run %s' % (sx.REAL_CFG, sx.ST_CFG, sx.PSEUDO_TAG, '$RW'))
assert sx.O == '$XW', sx.O" || exit 2
  [ -f $R/preds/pcmd_fused_test.pkl ] && echo "  reference prediction present: $R/preds/pcmd_fused_test.pkl" || echo "  (reference prediction not yet present)"
  exit 0
fi

[ -f $X/.done ] && { say "already done: $(grep 'ALL DONE' $LOG | tail -1 | cut -c1-400)"; exit 0; }
# ---------------- start gates: paper-critical runs first, then the predecessor runs of this queue
if [ "${HEADS_NO_WAIT:-0}" != 1 ]; then
  n=0; until [ -f $GP/review1113/initabl/.done ] && [ -f /d/cmpfull_runs/gupnet_uda_mtstrict_s0/.done ]; do
    [ $n = 0 ] && say "waiting for review1113/initabl/.done and D:/cmpfull_runs/gupnet_uda_mtstrict_s0/.done (poll 10 min)"; n=1; sleep 600; done
fi
for d in ${DEPS:-}; do n=0; until [ -f $HD/$d/.done ] || [ -f $HD/$d/.failed ] || [ -f $HD/$d/.skipped ]; do
  [ $n = 0 ] && say "waiting for predecessor $d"; n=1; sleep 600; done; done
case $MODE in FULL) LS=${LATEST_START:-11221800} ;; *) LS=${LATEST_START:-11230800} ;; esac
NOW=$(date '+%m%d%H%M'); if [ $((10#$NOW)) -gt $((10#$LS)) ] && ! ls $X/ref/*.pt > /dev/null 2>&1 && [ ! -d $X/ou/runs ]; then
  say "SKIPPED: machine time $NOW is past the latest start $LS (would not finish before the supplement deadline)"; touch $X/.skipped; exit 0; fi
rm -f $X/.failed $X/.skipped
[ -f $R/preds/pcmd_fused_test.pkl ] || die "reference run $R not finished"
say "start (mode $MODE, SX_SEEDSHIFT=$SHIFT, reference run $RW)"
$PY $HDW/make_heads_lib.py $ARM $TGT $SHIFT >> $LOG 2>&1 || die make_heads_lib
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
assert sx.O == '$XW' and sx.TEACH == 'full', (sx.O, sx.TEACH)
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s WK=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.W))
")" || die "sx config"
WKP=$(echo $WK | sed -E 's|^([A-Z]):/|/\L\1/|')                        # D:/x -> /d/x
mkdir -p $X/recs $X/ref $X/ou $X/preds $WKP; cd $X
say "lib ok: W $WK, pseudo tag $TAG"

# ================================================================== FULL: H1 H2 H3 (refiner changed -> whole downstream chain)
if [ $MODE = FULL ]; then
  $PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg
  cpf $F/recs/det0.pkl recs/det0.pkl
  for f in sim_meta.pkl real_d0_meta.pkl; do cpf $F/ref/$f ref/$f; done
  for f in sim_rgb.npy sim_sil.npy bank.npy real_d0_rgb.npy; do lnk $F/ref/$f ref/$f; [ -f ref/$f ] || die "link ref/$f"; done
  cpf $F/wsim_cache.pkl wsim_cache.pkl
  say "fixed stage-1 outputs from srcx/$C (det0, refiner sim samples / bank / real d0 crops [hard links], wsim cache)"
  ref12() { local j=$1
    [ -f ref/ref_st1_s$j.pt ] || oomrun logs/ref_st1_s$j.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $j
    [ -f ref/ref_st1_s$j.pt ] && { [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $j; }; }
  for j in 5 6; do ref12 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done
  say "refiners stage 1-2 done"
  for j in 5 6; do [ -f ref/st2_s${j}_det0_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det0.pkl train $XW/ref/st2_s${j}_det0_train.pkl || die "inf st2 det0 $j"; done
  [ -f ref/inf_st2_det0_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det0_train.pkl $XW/ref/st2_s5_det0_train.pkl $XW/ref/st2_s6_det0_train.pkl >> $LOG 2>&1 || die "avg st2 det0"
  [ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
  say "pseudo-labels: $(tr -d '\n ' < pseudo_$TAG.json)"
  STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
  DETS=($DET0 $DET1 $DET2)
  train_det1() {   # = run_seed_chain.sh: train.py resumes from its latest epoch checkpoint; OOM -> halve --batch_size (recipe deviation)
    local k=$1 bs=8 try=0 lg=$X/logs/train_det1_s$1.log
    while [ ! -f $(stck $k) ]; do
      ( cd /e/Open3DUAVDet/tools; $SLOT light $PY train.py --cfg_file $ST_CFG --batch_size $bs --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
          --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $((k + SHIFT)) --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} ) > $lg.try 2>&1
      cat $lg.try >> $lg
      [ -f $(stck $k) ] && break
      if isoom $lg.try && [ $try -lt 4 ] && [ $bs -gt 1 ]; then try=$((try + 1)); bs=$((bs / 2))
        echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k OOM -> retry $try/4 with --batch_size $bs (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
      else echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k FAILED, see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
    done; rm -f $lg.try
  }
  for k in 0 1 2; do train_det1 $k & sleep 100; done; wait
  for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
  [ -f recs/det1.pkl ] || oomrun logs/det1_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl || die "det1 infer"
  say "detectors done"
  for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WKP/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
  [ -f $WKP/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
  TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; INIT=($IRL0 $IRL1)
  set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); waitf $2; [ -f ou/T_$1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 || die "adabn $1"; shift 2; done
  T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
  [ -f ou/read_tr1_teach.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 || die "read teachers"
  q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
  GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
  [ -f $WKP/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
  say "$(grep 'pseudo pseudo_tr1' $LOG | tail -1)"
fi

# ================================================================== ORI: H4 H5 (teacher group changed; students + deployment)
if [ $MODE = ORI ]; then
  for f in recs/det1.pkl ref/inf_fin_det1_train.pkl ref/inf_fin_det1_val.pkl ref/inf_fin_det1_test.pkl kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do cpf $R/$f $f; done
  cpf $R/w_sim.json w_frozen.json
  for f in oc_tr1.npy oc_tr1.pkl oc_val1.npy oc_val1.pkl oc_te1.npy oc_te1.pkl ou_bank.npy; do lnk $RWK/$f $WKP/$f; [ -f $WKP/$f ] || die "link $f"; done
  say "reference inputs: det1, frozen stage-3 depth students, height readings, w; crops / bank hard-linked from $RWK"
  RG0=$M/sim_full_r34_kp_w12_zoom/KZs0/ckpt/checkpoint_epoch_30.pth; RG1=$M/sim_full_r34_kp_w12_zoom/KZs1/ckpt/checkpoint_epoch_30.pth
  # [11-17] outdoor source: the same-seed plain RGB KZ nets = srcx/run_x.sh SX_TEACH=none teachers (OKZrgb_s0/1, out kp_cfg, epoch 30)
  [ $SRC = out ] && { RG0=$M/sim_out_r34_kp_w12_zoom_out/OKZrgb_s0/ckpt/checkpoint_epoch_30.pth; RG1=$M/sim_out_r34_kp_w12_zoom_out/OKZrgb_s1/ckpt/checkpoint_epoch_30.pth; }
  if [ $ARM = H4 ]; then   # LiDAR nets kzl_s0/1 -> same-seed plain KZ nets (KZ recipe, 30 epochs, no auxiliary heads); student init unchanged (kzm)
    for f in $RG0 $RG1; do [ -f $f ] || die "missing $f"; done
    [ -f ou/T_kz0.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $RG0 tr1 $XW/ou/T_kz0.pth 80 || die "adabn kz0"
    [ -f ou/T_kz1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $RG1 tr1 $XW/ou/T_kz1.pth 80 || die "adabn kz1"
    [ -f ou/read_tr1_new.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_new.pkl $XW/ou/T_kz0.pth,$XW/ou/T_kz1.pth || die "read new teachers"
    SPEC="irlid0=ref:irlid0,irlid1=ref:irlid1,kdir0=ref:kdir0,kdir1=ref:kdir1,kz0=new:kz0,kz1=new:kz1"; NEW=$XW/ou/read_tr1_new.pkl
    TS=(irlid0 irlid1 kdir0 kdir1 kz0 kz1); LIDG=(kz0 kz1); INIT=($IRL0 $IRL1)
  else                     # H5: IR+LiDAR nets kzm_s0/1 -> same-seed KZ-LiD nets (kzl_s0/1 = KZ-IRLiD recipe without the IR target), teacher AND student init
    SPEC="irl2lid0=ref:lid0,irl2lid1=ref:lid1,kdir0=ref:kdir0,kdir1=ref:kdir1,lid0=ref:lid0,lid1=ref:lid1"; NEW=-
    TS=(irl2lid0 irl2lid1 kdir0 kdir1 lid0 lid1); LIDG=(lid0 lid1); INIT=($LID0 $LID1)
  fi
  [ -f ou/read_tr1_teach.pkl ] || $PY $HDW/merge_reads.py $RW/ou/read_tr1_teach.pkl $NEW $XW/ou/read_tr1_teach.pkl $XW/ou "$SPEC" >> $LOG 2>&1 || die "merge readings"
  q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
  GRP="{\"T\": [$(q ${TS[0]}), $(q ${TS[1]}), $(q ${TS[2]}), $(q ${TS[3]}), $(q ${TS[4]}), $(q ${TS[5]})], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q ${LIDG[0]}), $(q ${LIDG[1]})]}"
  echo "teacher groups: $GRP" >> $LOG
  [ -f $WKP/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
  say "$(grep 'pseudo pseudo_tr1' $LOG | tail -1) | reference: $(grep -h 'pseudo pseudo_tr1' $R/logs/*.log 2>/dev/null | tail -1)"
fi

# ---------------- students S6I (FULL and ORI): nominal seeds 0, 1 (+SHIFT), gated pseudo-rotations + mirror loss 12, 2250 steps
if [ $MODE = FULL ] || [ $MODE = ORI ]; then
  train_student() {   # $1 = seed; resume from runs/S6I_s<seed>/resume.pth; OOM -> halve --bs-src/--bs-tgt (recipe deviation)
    local s=$1 bs=8 bt=4 try=0 lg=$X/logs/ou_train_s$1.log
    while [ ! -f ou/runs/S6I_s$s/ckpt_last.pth ]; do
      OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $XW/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
      cat $lg.try >> $lg
      [ -f ou/runs/S6I_s$s/ckpt_last.pth ] && break
      if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
        echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
      else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED, see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
    done; rm -f $lg.try
  }
  for s in 0 1; do train_student $s & sleep 100; done; wait
  for s in 0 1; do [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || die "student s$s"
    grep -h -m1 "INFO  init " logs/ou_train_s$s.log | sed 's/ | steps.*//' >> $LOG
    [ -f ou/S_ab$s.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 || die "student adabn $s"; done
  [ $MODE = FULL ] && NS="tr1 val1 te1" || NS="val1 te1"   # ORI: train readings only feed the (frozen) stage-3 targets
  for n in $NS; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth || die "read $n"; done
  say "students done"
fi

# ---------------- FULL: height readings, w, stage 3
if [ $MODE = FULL ]; then
  waitf $KPS
  [ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
  for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
  [ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1 || die "avg st2 det1"
  [ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
  WW=$($PY -c "import json; print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW"
  [ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
  ref3() { local j=$1; [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j; }
  for j in 5 6; do ref3 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
  for sp in train val test; do
    for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
    [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1 || die "avg fin $sp"
  done
fi

# ================================================================== HGT: H6 (height ratio from box corners) / DEP: H1c (one colour class)
if [ $MODE = HGT ] || [ $MODE = DEP ]; then
  for f in recs/det1.pkl ref/inf_fin_det1_train.pkl ref/inf_fin_det1_val.pkl ref/inf_fin_det1_test.pkl ou/read_val1_stu.pkl ou/read_te1_stu.pkl; do cpf $R/$f $f; done
  cpf $R/w_sim.json w_frozen.json
  if [ $MODE = HGT ]; then
    for f in oc_tr1.npy oc_tr1.pkl oc_val1.npy oc_val1.pkl oc_te1.npy oc_te1.pkl; do lnk $RWK/$f $WKP/$f; [ -f $WKP/$f ] || die "link $f"; done
    waitf $KPS
    [ -f kps_te1.pkl ] || oomrun logs/kpsbox.log "ORI_BS=16" $SLOT light $IW $L/kpsbox_x.py $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kpsbox
    grep "kpsbox" logs/kpsbox.log | tail -3 | while read -r l; do say "$l"; done
    $PY -c "
import pickle, numpy as np
from scipy.stats import spearmanr
for n in ('tr1', 'val1', 'te1'):
    a = pickle.load(open('$XW/kps_%s.pkl' % n, 'rb')); b = pickle.load(open('$RW/kps_%s.pkl' % n, 'rb')); k = [x for x in a if x in b]
    va, vb = np.array([a[x] for x in k]), np.array([b[x] for x in k])
    print('height ratio %s: box corners median %.3f (IQR %.3f) | LiDAR structure points median %.3f (IQR %.3f) | Spearman %.2f, n %d' % (
        n, np.median(va), np.subtract(*np.percentile(va, [75, 25])), np.median(vb), np.subtract(*np.percentile(vb, [75, 25])), spearmanr(va, vb)[0], len(k)))
" > logs/kps_compare.log 2>&1; while read -r l; do say "$l"; done < logs/kps_compare.log
  else
    for f in kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do cpf $R/$f $f; done
    mkdir -p work; for f in oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do cpf $RWK/$f work/$f; done   # chain_x.zL_table reads the crop info pkl only
  fi
fi

# ================================================================== BH: H7 (box-hull silhouette target, refiner only, seed shift 10 / 20)
if [ $MODE = BH ]; then
  BH=$GP/review1113/boxhull/in2mav
  [ -f $BH/ref/sim_sil.npy ] && [ -f $BH/ref/align_hull.json ] || die "boxhull/in2mav hull array missing"
  grep -q '"mode": "ALIGNED' $BH/ref/align_hull.json || die "boxhull hull array not aligned to the clean sample list"
  cmp -s $BH/ref/sim_meta.pkl $R/ref/sim_meta.pkl || die "reference sim samples differ from the boxhull run's"
  for f in recs/det0.pkl recs/det1.pkl ou/read_tr1_stu.pkl ou/read_val1_stu.pkl ou/read_te1_stu.pkl kps_tr1.pkl kps_val1.pkl kps_te1.pkl ref/sim_meta.pkl ref/real_d0_meta.pkl; do mkdir -p $(dirname $f); cpf $R/$f $f; done
  for f in sim_rgb.npy bank.npy real_d0_rgb.npy; do lnk $R/ref/$f ref/$f; [ -f ref/$f ] || die "link ref/$f"; done
  cpf $BH/ref/sim_sil.npy ref/sim_sil.npy      # box hull (filled convex hull of the projected GT box, same crops) instead of the IR silhouette
  cpf $R/w_sim.json w_frozen.json
  mkdir -p work; for f in oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do cpf $RWK/$f work/$f; done
  [ -f $(echo $REAL | sed -E 's|^([A-Z]):/|/\L\1/|')/train/index_pseudo_$TAG.pkl ] || die "reference pseudo-label index $TAG missing"
  WW=$($PY -c "import json; print(json.load(open('$XW/w_frozen.json'))['w'])"); say "box hull at seed shift $SHIFT; frozen w of the reference run = $WW"
  bh12() { local j=$1
    [ -f ref/ref_st1_s$j.pt ] || oomrun logs/ref_st1_s$j.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $j
    [ -f ref/ref_st1_s$j.pt ] && { [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $j; }; }
  for j in 5 6; do bh12 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done
  for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
  [ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1 || die "avg st2"
  [ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW > logs/targets.log 2>&1 || die targets
  bh3() { local j=$1; [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j; }
  for j in 5 6; do bh3 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "refiner st3 s$j"; done
  for sp in train val test; do
    for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
    [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1 || die "avg fin $sp"
  done
fi

# ================================================================== deployment, ONE test scoring, paired bootstrap vs the same-seed reported run
[ -n "${WW:-}" ] || WW=$($PY -c "import json; print(json.load(open('$XW/w_frozen.json'))['w'])") || die "w"
[ -f deploy_$PFX.json ] || $PY $L/chain_x.py deploy $WW $PFX > logs/deploy.log 2>&1 || die deploy
grep -E '"c_k"|"r_k"|"col_centres"' -A3 deploy_$PFX.json | tr -d ' \n' | cut -c1-300 | while read -r l; do say "deploy constants: $l"; done
[ -f preds/${PFX}_fused_test.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/${PFX}_fused_test.pkl $SPLIT > logs/score.log 2>&1 || die "score test"
$PY $HDW/heads_boot.py $TGT $XW/boot_test.json "$ARM $RUN vs PST $RUN" $XW/preds/${PFX}_fused_test.pkl $RW/preds/pcmd_fused_test.pkl > logs/boot.log 2>&1 || die "bootstrap"
ONE=$(tail -1 logs/boot.log)
# pooled over every finished seed pair of this arm/target (+ the existing boxhull shift-0 run for H7)
PAIRS=""
for s in 0 10 20; do d=$HD/$ARM/${TGT}_s$s; dw=$HDW/$ARM/${TGT}_s$s
  case "$TGT/$s" in mav/0) rr=$GPW/srcx/in2mav ;; mav/10) rr=$GPW/review1113/sup3/run ;; mav/20) rr=$GPW/review1113/seeds/mav_full_s20 ;;
                    antifs/0) rr=$GPW/srcx/out2antifs ;; antifs/10) rr=$GPW/review1113/seeds/antifs_full_s10 ;; antifs/20) rr=$GPW/review1113/seeds/antifs_full_s20 ;; esac
  if [ $s = $SHIFT ] || [ -f $d/.done ]; then PAIRS="$PAIRS $dw/preds/${PFX}_fused_test.pkl $rr/preds/pcmd_fused_test.pkl"
  elif [ $ARM = H7 ] && [ $s = 0 ] && [ -f $GP/review1113/boxhull/in2mav/preds/boxhull_fused_test.pkl.score_real_test.json ]; then
    PAIRS="$PAIRS $GPW/review1113/boxhull/in2mav/preds/boxhull_fused_test.pkl $rr/preds/pcmd_fused_test.pkl"; fi
done
NP=$(echo $PAIRS | wc -w)
if [ $NP -gt 2 ]; then $PY $HDW/heads_boot.py $TGT $HDW/$ARM/pooled_$TGT.json "$ARM pooled ($((NP / 2)) seed pairs)" $PAIRS > $X/logs/boot_pooled.log 2>&1 && POOL=" || $(tail -1 $X/logs/boot_pooled.log)"; fi
say "ALL DONE: $ONE${POOL:-}"
touch $X/.done
