#!/bin/bash
# [review1113 hparam_1117] lambda sensitivity: ONE whole-chain PST run with the Module-3 pseudo-box lambda changed (make_pseudo_x.py LAM;
# deployed 0.75). usage: bash lam_chain.sh <lam tag 050|100|...> <tgt mav|antifs> <shift 0|10|20>
# lambda changes the step-1 pseudo boxes -> everything downstream is recomputed (= review1113/heads/run_arm.sh FULL mode):
#   pseudo-labels -> 3 self-trained detectors (seed k+shift, init = the fixed sim detector) -> det1 -> orientation crops / bank, 6 teachers
#   (fixed checkpoints, AdaBN on the new crops), pseudo-orientations + 10 deg gate -> students S6I x2 -> height readings -> w (sim val) ->
#   stage-3 targets -> stage-3 refiners x2 -> deployment -> val score, then test scored ONCE.
# NOT recomputed (they do not depend on lambda; copied / hard-linked from the same-shift reference PST run): fixed stage-1 outputs (det0,
#   refiner sim samples, real bank, real d0 crops, wsim cache) AND the stage-1/2 refiners (seeds 5,6 +shift) with their det0-train
#   corrections, which are the inputs of make_pseudo_x.py.
# Trainings strictly one at a time (detectors, students, stage-3 refiners serial). GPU via gpu_slot.sh light; inference via infer_wrap.py
#   (memory fraction 0.2); CUDA OOM -> batch halved + resume (max 4; logs/retry.log; a halved TRAINING batch = RECIPE DEVIATION);
#   workers 0; every step skipped once its output exists (re-launch resumes); never kills processes. Markers .done / .failed.
LT=$1; TGT=$2; SHIFT=$3
case "$TGT" in mav) SRC=in; SPLIT=real_test ;; antifs) SRC=out; SPLIT=antifs_test ;; *) echo "bad tgt"; exit 2 ;; esac
case "$SHIFT" in 0|10|20) ;; *) echo "bad shift"; exit 2 ;; esac
C=${SRC}2${TGT}; RUN=${TGT}_L${LT}_s$SHIFT
export SX_SEEDSHIFT=$SHIFT SX_TEACH=full SX_SRC=$SRC SX_TGT=$TGT PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset OU_INIT OU_BANK OU_CROPS REF_TAG HP_PSEUDO HP_GAMMA HP_IN HP_STU
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
HP=$GP/review1113/hparam_1117; HPW=$GPW/review1113/hparam_1117
X=$HP/lam/$RUN; XW=$HPW/lam/$RUN; L=$XW/lib
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm; IW="$PY $L/infer_wrap.py"
F=$GP/srcx/$C
case "$TGT/$SHIFT" in
  mav/0) R=$GP/srcx/in2mav ;; mav/10) R=$GP/review1113/sup3/run ;; mav/20) R=$GP/review1113/seeds/mav_full_s20 ;;
  antifs/0) R=$GP/srcx/out2antifs ;; antifs/10) R=$GP/review1113/seeds/antifs_full_s10 ;; antifs/20) R=$GP/review1113/seeds/antifs_full_s20 ;; esac
wp() { echo "$1" | sed -E 's|^/([a-z])/|\U\1:/|'; }
mkdir -p $HP/logs $X/logs; GLOG=$HP/logs/hparam.log; LOG=$X/logs/run.log; RLOG=$X/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') [lam $RUN] $1" | tee -a $LOG >> $GLOG; }
die() { say "!! $1"; touch $X/.failed; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
waitf() { local n=0; until [ -f "$1" ]; do [ $n = 0 ] && say "waiting for $1"; n=1; sleep 120; done; }
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
lnk() { [ -f "$2" ] && return 0; $PY -c "import os, sys; os.link(sys.argv[1], sys.argv[2])" "$(wp $1)" "$(wp $2)" 2>/dev/null || cp "$1" "$2"; }
cpf() { [ -f "$2" ] || cp "$1" "$2" || die "copy $1"; }

[ -f $X/.done ] && { say "already done: $(grep 'ALL DONE' $LOG | tail -1)"; exit 0; }
rm -f $X/.failed
[ -f $R/preds/pcmd_fused_test.pkl ] || die "reference run $R not finished"
$PY $HPW/make_lam_lib.py $LT $TGT $SHIFT >> $LOG 2>&1 || die make_lam_lib
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
assert sx.O == '$XW' and sx.TEACH == 'full', (sx.O, sx.TEACH)
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s WK=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.W))
")" || die "sx config"
grep "LAM, CONF" $L/make_pseudo_x.py >> $LOG
WKP=$(echo $WK | sed -E 's|^([A-Z]):/|/\L\1/|')
mkdir -p $X/recs $X/ref $X/ou $X/preds $WKP; cd $X
say "start (SX_SEEDSHIFT=$SHIFT, reference $(wp $R), W $WK, tag $TAG)"
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg
cpf $F/recs/det0.pkl recs/det0.pkl
for sp in val test; do cpf $F/preds/srconly_$sp.pkl preds/srconly_$sp.pkl; done
for f in sim_meta.pkl real_d0_meta.pkl; do cpf $F/ref/$f ref/$f; done
for f in sim_rgb.npy sim_sil.npy bank.npy real_d0_rgb.npy; do lnk $F/ref/$f ref/$f; [ -f ref/$f ] || die "link ref/$f"; done
cpf $F/wsim_cache.pkl wsim_cache.pkl
# refiners stage 1-2 and their det0-train corrections do not depend on lambda: same-shift reference run
for j in 5 6; do cpf $R/ref/ref_st1_s$j.pt ref/ref_st1_s$j.pt; cpf $R/ref/ref_st2_s$j.pt ref/ref_st2_s$j.pt; cpf $R/ref/st2_s${j}_det0_train.pkl ref/st2_s${j}_det0_train.pkl; done
cpf $R/ref/inf_st2_det0_train.pkl ref/inf_st2_det0_train.pkl
cmp -s recs/det0.pkl $R/recs/det0.pkl || die "det0 of the reference run differs from srcx/$C"
say "fixed inputs copied (det0, sim samples, bank, d0 crops, wsim cache from srcx/$C; refiners st1/st2 + det0 corrections from the reference run)"

[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
say "pseudo-labels: $(tr -d '\n ' < pseudo_$TAG.json) | reference: $(tr -d '\n ' < $(ls $R/pseudo_*.json | head -1))"
STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
DETS=($DET0 $DET1 $DET2)
train_det1() {
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
for k in 0 1 2; do train_det1 $k || die "self-trained detector s$k"; done
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
train_student() {
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
for s in 0 1; do train_student $s || die "student s$s"; done
for s in 0 1; do [ -f ou/S_ab$s.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 || die "student adabn $s"; done
for n in tr1 val1 te1; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth || die "read $n"; done
say "students done"
waitf $KPS
[ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1 || die "avg st2 det1"
[ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
WW=$($PY -c "import json; print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1 || die "avg fin $sp"
done
[ -f deploy_hp.json ] || $PY $L/chain_x.py deploy $WW hp > logs/deploy.log 2>&1 || die deploy
[ -f preds/hp_fused_val.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/hp_fused_val.pkl val >> logs/score.log 2>&1 || die "score val"
[ -f preds/hp_fused_test.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/hp_fused_test.pkl $SPLIT >> logs/score.log 2>&1 || die "score test"
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
fm = lambda d: 'UDS %.2f AP %.2f ATE %.3f ASE %.3f AOE %.2f mDR %.2f' % (d['UDS'], d['AP'], d['ATE'], d['ASE'], d['AOE'], d['mDR'])
print('test %s | val %s | w %.4f' % (fm(f('$XW/preds/hp_fused_test.pkl.score_$SPLIT.json')), fm(f('$XW/preds/hp_fused_val.pkl.score_val.json')), $WW))")
say "ALL DONE: $SUM"; touch $X/.done
