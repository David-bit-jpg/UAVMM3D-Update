#!/bin/bash
# [review1113 hparam_1117] orientation-gate / mirror-loss-weight sensitivity on ONE existing PST run (= review1113/sup2 nogate and
# review1113/initabl protocol): only the orientation students are retrained; AdaBN; mirror-view readings on val/test; deployment at the run's
# FROZEN w with its frozen stage-3 depth students (inf_fin_det1_*), det1 and height readings; val score, then test scored ONCE.
#   usage: bash stu_arm.sh <arm> <run key>
#   arms:  g05 / g20  IR-vs-LiDAR agreement gate 5 / 20 deg (deployed 10): gate flag recomputed from the run's own pseudo_tr1.npz (d_il <= T),
#                     same pseudo-rotations, --tgt gate
#          gall       no gate (--tgt all) = sup2 protocol (reported for mav_s0 by review1113/sup2/nogate; used here for the other runs)
#          m06 / m24  mirror-consistency loss weight 6 / 24 (deployed 12), gate 10
#          rep        deployed values (gate 10, mirror 12) retrained: retraining-noise reference for this protocol
#   run keys: mav_s0 mav_s10 mav_s20 anti_s0 anti_s10 anti_s20 (see make_hp_lib.py)
# Student recipe otherwise = run_x.sh S6I (init = same-seed IR+LiDAR member, bs 8+4, 2250 steps, background swap 0.5), nominal seeds 0, 1
#   (+ the run's SX_SEEDSHIFT). The two students train ONE AFTER ANOTHER (at most one training of ours at a time).
# Robustness: gpu_slot.sh light; students resume (resume.pth every 250 steps), OOM -> halve --bs-src/--bs-tgt (max 4, RECIPE DEVIATION in
#   logs/retry.log); inference via infer_wrap.py (memory fraction 0.2), OOM -> ORI_BS halved; workers 0; every step skipped once its output
#   exists; never kills processes. Writes only hparam_1117/stu/<arm>/<run>/. Markers .done / .failed.
ARM=$1; RK=$2
case $ARM in g05) GT=5; TGTM=gate; MIR=12 ;; g20) GT=20; TGTM=gate; MIR=12 ;; gall) GT=; TGTM=all; MIR=12 ;;
             m06) GT=; TGTM=gate; MIR=6 ;; m24) GT=; TGTM=gate; MIR=24 ;; rep) GT=; TGTM=gate; MIR=12 ;; *) echo "bad arm $ARM"; exit 2 ;; esac
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
case $RK in mav_s0) B=srcx/in2mav; WK=D:/S2MUAV3D/srcx_work/in2mav; SS= ;; mav_s10) B=review1113/sup3/run; WK=D:/S2MUAV3D/srcx_work/in2mav_seed2; SS=10 ;;
            mav_s20) B=review1113/seeds/mav_full_s20; WK=D:/S2MUAV3D/srcx_work/in2mav_sd20; SS=20 ;;
            anti_s0) B=srcx/out2antifs; WK=D:/S2MUAV3D/srcx_work/out2antifs; SS= ;; anti_s10) B=review1113/seeds/antifs_full_s10; WK=D:/S2MUAV3D/srcx_work/out2antifs_sd10; SS=10 ;;
            anti_s20) B=review1113/seeds/antifs_full_s20; WK=D:/S2MUAV3D/srcx_work/out2antifs_sd20; SS=20 ;; *) echo "bad run $RK"; exit 2 ;; esac
case $RK in mav*) export SX_SRC=in SX_TGT=mav; SPLIT=real_test ;; anti*) export SX_SRC=out SX_TGT=antifs; SPLIT=antifs_test ;; esac
export SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset OU_INIT OU_BANK OU_CROPS HP_PSEUDO HP_GAMMA HP_STU
if [ -n "$SS" ]; then export SX_SEEDSHIFT=$SS; else unset SX_SEEDSHIFT; fi
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"
R=$GP/$B; RW=$GPW/$B; WKP=$(echo $WK | sed -E 's|^([A-Z]):/|/\L\1/|')
HP=$GP/review1113/hparam_1117; HPW=$GPW/review1113/hparam_1117
A=$HP/stu/$ARM/$RK; AW=$HPW/stu/$ARM/$RK; L=$AW/lib
mkdir -p $HP/logs $A/ou/runs $A/preds $A/logs; cd $A
GLOG=$HP/logs/hparam.log; LOG=$A/logs/run.log; RLOG=$A/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') [stu $ARM $RK] $1" | tee -a $LOG >> $GLOG; }
die() { say "!! $1"; touch $A/.failed; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
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
[ -f $A/.done ] && { say "already done: $(grep 'ALL DONE' $LOG | tail -1)"; exit 0; }
rm -f $A/.failed
[ -f $R/preds/pcmd_fused_test.pkl ] && [ -f $R/w_sim.json ] || die "base run $R not finished"
for f in pseudo_tr1.npz oc_tr1.npy oc_tr1.pkl oc_val1.npy oc_val1.pkl oc_te1.npy oc_te1.pkl ou_bank.npy; do [ -f $WKP/$f ] || die "missing $WK/$f"; done
$PY $HPW/make_hp_lib.py $RK $AW >> $LOG 2>&1 || die make_hp_lib
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx
assert sx.O == '$AW' and sx.W == '$WK' and sx.TEACH == 'full', (sx.O, sx.W)
print('IRL0=%s IRL1=%s' % tuple(sx.S['irlid']))")" || die "sx config"
INIT=($IRL0 $IRL1)
WRUN=$($PY -c "import json; print(json.load(open('$RW/w_sim.json'))['w'])") || die "w"
PSEUDO_ARG=""
if [ -n "$GT" ]; then
  [ -f $A/pseudo_tr1_g$GT.npz ] || $PY -c "
import numpy as np
P = dict(np.load('$WK/pseudo_tr1.npz', allow_pickle=False)); g0 = P['gate'].copy()
assert np.array_equal(g0, P['d_il'] <= 10.0), 'stored gate is not d_il <= 10'
P['gate'] = P['d_il'] <= float($GT)
np.savez('$AW/pseudo_tr1_g$GT.npz', **P)
print('gate %s deg: %d / %d crops pass (deployed 10 deg: %d)' % ('$GT', P['gate'].sum(), len(P['gate']), g0.sum()))" >> $LOG 2>&1 || die "gate npz"
  export HP_PSEUDO=$AW/pseudo_tr1_g$GT.npz
fi
say "start: base $RW (w $WRUN, SX_SEEDSHIFT ${SS:-0}) | --tgt $TGTM --mir $MIR ${GT:+gate $GT deg} $(grep 'crops pass' $LOG | tail -1)"
train_student() {
  local s=$1 bs=8 bt=4 try=0 lg=$A/logs/ou_train_s$1.log
  while [ ! -f $A/ou/runs/S6I_s$s/ckpt_last.pth ]; do
    OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt $TGTM --mir $MIR --pseudo pseudo_tr1 --runs-dir $AW/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $A/ou/runs/S6I_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for s in 0 1; do train_student $s || die "student s$s"; done
for s in 0 1; do grep -h "目标域：" $A/ou/runs/S6I_s$s/train_*.log 2>/dev/null | tail -1 | sed "s/^/[s$s] /" >> $LOG; done
say "students done ($(grep -h '目标域：' $A/ou/runs/S6I_s0/train_*.log 2>/dev/null | tail -1 | sed 's/.*目标域：//'))"
for s in 0 1; do [ -f $A/ou/S_ab$s.pth ] || oomrun $A/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $AW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $AW/ou/S_ab$s.pth 0 || die "adabn s$s"; done
for n in val1 te1; do [ -f $A/ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun $A/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read $n $AW/ou/read_${n}_stu.pkl $AW/ou/S_ab0.pth,$AW/ou/S_ab1.pth || die "read $n"; done
[ -f $A/deploy_hp.json ] || HP_IN=$RW $PY $L/chain_x.py deploy $WRUN hp > $A/logs/deploy.log 2>&1 || die deploy
[ -f $A/preds/hp_fused_val.pkl.score_val.json ] || $PY $L/score_x.py $AW/preds/hp_fused_val.pkl val >> $A/logs/score.log 2>&1 || die "score val"
[ -f $A/preds/hp_fused_test.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $AW/preds/hp_fused_test.pkl $SPLIT >> $A/logs/score.log 2>&1 || die "score test"
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
fm = lambda d: 'UDS %.2f AP %.2f ATE %.3f ASE %.3f AOE %.2f mDR %.2f' % (d['UDS'], d['AP'], d['ATE'], d['ASE'], d['AOE'], d['mDR'])
print('test %s | val %s | base test %s' % (fm(f('$AW/preds/hp_fused_test.pkl.score_$SPLIT.json')), fm(f('$AW/preds/hp_fused_val.pkl.score_val.json')), fm(f('$RW/preds/pcmd_fused_test.pkl.score_$SPLIT.json'))))")
say "ALL DONE: $SUM"; touch $A/.done
