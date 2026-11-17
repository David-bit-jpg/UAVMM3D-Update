#!/bin/bash
# [sup2] NOT RUN by its author. "10-degree check off": retrain the clean MAV6D chain's orientation students on ALL frames that have a
# pseudo-orientation (12,866 / 12,866 train crops; --tgt all) instead of only the IR-vs-LiDAR-gated ones (gate == True, 4,301 crops; --tgt gate).
# Everything else = run_x.sh / nomir_commands.sh: S6I recipe, pseudo_tr1 pseudo-rotations (the same file, the gate flag is simply ignored),
# MIRROR LOSS KEPT (--mir 12), init = same-seed IR+LiDAR member, background swap 0.5, 2,250 steps, bs 8 sim + 4 real, mem 0.3, bag 0.8,
# seeds 0 and 1; then AdaBN (all train crops), 2-student readings with the test-time mirror view, deploy with the frozen omega, score val + test once.
#   Default (FULL=0, as nomir_commands.sh): depth refiners, c_k, r_k, s_bar frozen (copied); only rotations and the corner-PnP distance of the
#     test-time fusion change (b_k re-estimated from unlabeled val images by the deploy rule).  ~35-40 min wall.
#   FULL=1: also re-reads train crops and retrains the step-3 refiners (seeds 5, 6) on fused targets from the new students (omega frozen). ~80 min.
# Writes ONLY under review1113/sup2/nogate/ (lib copies from make_nogate_lib.py redirect sx.O/sx.W to $CL_AO). Read only: srcx/in2mav/{recs,ref,kps_*},
#   D:/S2MUAV3D/srcx_work/in2mav/{pseudo_tr1.npz,oc_*,ou_bank.npy} (crop arrays read in place via OU_CROPS / OU_BANK / _wf()).
# Robustness: every training step resumes (students: <run>/resume.pth every 250 steps; refiners: ref/resume_st3_s<j>.pt) and is skipped if its
#   output exists; on CUDA OOM the batch is halved and the step retried (max 4 retries), every retry logged to logs/retry.log (a halved
#   student batch is a recipe deviation and is flagged as such). All loaders use workers 0. Inference runs through gpu_slot.sh light and
#   infer_wrap.py (torch memory fraction 0.2). Never kills processes.
#   launch:  bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate_commands.sh
FULL=${FULL:-0}; MIRVIEW=${MIRVIEW:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; S2=$GP/review1113/sup2; L=$S2/nogate/lib
X=$GP/srcx/in2mav; XW=D:/S2MUAV3D/srcx_work/in2mav
export CL_AO=$S2/nogate; A=$CL_AO
export OU_CROPS=$XW/oc_tr1.npy OU_BANK=$XW/ou_bank.npy          # read in place (5.7 GB / 2.1 GB, not copied)
W_FROZEN=0.190282282581008                                       # = srcx/in2mav/w_sim.json (deployed omega)
TAG=NOGATE
mkdir -p $A/recs $A/ref $A/ou/runs $A/preds $A/logs $A/work
LOG=$A/logs/nogate.log; RLOG=$A/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
# oomrun <log> "<VAR=start ...>" cmd...   (inference / BN steps) every attempt logged separately; on OOM all VARs halve (min 1), max 4 retries
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

# 0) lib copies + read-only inputs (~20 MB; FULL adds ~1.2 GB)
$PY $S2/make_nogate_lib.py >> $LOG 2>&1 || die "make_nogate_lib"
[ -f $A/recs/det1.pkl ] || cp $X/recs/det1.pkl $A/recs/
for f in kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do [ -f $A/$f ] || cp $X/$f $A/; done
for f in pseudo_tr1.npz oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do [ -f $A/work/$f ] || cp $XW/$f $A/work/; done
if [ $FULL = 0 ]; then for sp in train val test; do [ -f $A/ref/inf_fin_det1_$sp.pkl ] || cp $X/ref/inf_fin_det1_$sp.pkl $A/ref/; done
else [ -f $A/recs/det0.pkl ] || cp $X/recs/det0.pkl $A/recs/
  for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st2_s5.pt ref_st2_s6.pt inf_st2_det1_train.pkl; do [ -f $A/ref/$f ] || cp $X/ref/$f $A/ref/; done
fi
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx
print('IRL0=%s IRL1=%s' % tuple(sx.S['irlid']))")" || die "sx"
INIT=($IRL0 $IRL1)
say "start nogate FULL=$FULL MIRVIEW=$MIRVIEW (students on all pseudo-oriented frames, --tgt all, --mir 12)"
$PY -c "import numpy as np; P = np.load('$A/work/pseudo_tr1.npz'); print('pseudo_tr1: %d crops with a pseudo-orientation, gate passes %d -> students now use all %d (bag 0.8)' % (len(P['idx']), P['gate'].sum(), len(P['idx'])))" | tee -a $LOG

# 1) students WITHOUT the gate, seeds 0 and 1 concurrently (as run_x.sh; ~25 min wall). Resume from <run>/resume.pth; OOM -> halve batch.
train_student() {   # $1 = seed
  local s=$1 bs=8 bt=4 try=0 lg=$A/logs/ou_train_s$1.log
  while [ ! -f $A/ou/runs/${TAG}_s$s/ckpt_last.pth ]; do
    OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag $TAG --tgt all --mir 12 --pseudo pseudo_tr1 \
      --runs-dir $A/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $A/ou/runs/${TAG}_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for s in 0 1; do ( train_student $s ) & sleep 100; done; wait
for s in 0 1; do [ -f $A/ou/runs/${TAG}_s$s/ckpt_last.pth ] || die "student s$s missing"; done
grep -h "目标域：" $A/ou/runs/${TAG}_s0/train_*.log | tail -1 | tee -a $LOG
say "students done"

# 2) AdaBN on all train crops (cap 0), as run_x.sh (~3 min)
for s in 0 1; do [ -f $A/ou/S_ab$s.pth ] || oomrun $A/logs/adabn_s$s.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $A/ou/runs/${TAG}_s$s/ckpt_last.pth tr1 $A/ou/S_ab$s.pth 0 || die "adabn s$s"; done

# 3) readings of the 2-student ensemble (+ test-time mirror view unless MIRVIEW=0): val1, te1 (~4.5 min); tr1 only for FULL (+6.6 min)
SETS="val1 te1"; [ $FULL = 1 ] && SETS="tr1 val1 te1"
for n in $SETS; do [ -f $A/ou/read_${n}_stu.pkl ] || ORI_MIRROR=$MIRVIEW oomrun $A/logs/read_$n.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read $n $A/ou/read_${n}_stu.pkl $A/ou/S_ab0.pth,$A/ou/S_ab1.pth || die "read $n"; done
say "readings done"

# 4) FULL only: step-3 targets from the new corner distances (omega frozen), refiners s5 || s6 (2,000 steps, resumable), inference, seed mean
if [ $FULL = 1 ]; then
  [ -f $A/ref/dp_fused.npy ] || $PY $L/chain_x.py targets $W_FROZEN > $A/logs/targets.log 2>&1 || die targets
  for j in 5 6; do ( [ -f $A/ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$A/ref/dp_fused.npy oomrun $A/logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
  for j in 5 6; do [ -f $A/ref/ref_st3_s$j.pt ] || die "ref st3 s$j missing"; done
  for sp in train val test; do
    for j in 5 6; do [ -f $A/ref/fin_s${j}_$sp.pkl ] || oomrun $A/logs/infer.log "REF_BS=128" $SLOT light $PY $L/infer_wrap.py $L/refiner_x.py infer $A/ref/ref_st3_s$j.pt $A/recs/det1.pkl $sp $A/ref/fin_s${j}_$sp.pkl || die "infer $j $sp"; done
    $PY $GP/srcx/lib/avg_inf.py $A/ref/inf_fin_det1_$sp.pkl $A/ref/fin_s5_$sp.pkl $A/ref/fin_s6_$sp.pkl >> $A/logs/infer.log 2>&1
  done
fi

# 5) deployment (CPU, frozen omega; also writes the nofuse arm) and scoring: val sanity, test ONCE
$PY $L/chain_x.py deploy $W_FROZEN nogate > $A/logs/deploy.log 2>&1 || die deploy
for p in nogate_fused_val nogate_nofuse_val; do [ -f $A/preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $A/preds/$p.pkl val >> $A/logs/score.log 2>&1; done
for p in nogate_fused_test nogate_nofuse_test; do [ -f $A/preds/$p.pkl.score_real_test.json ] || $PY $L/score_x.py $A/preds/$p.pkl real_test >> $A/logs/score.log 2>&1 || die "score $p"; done
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
t = f('$A/preds/nogate_fused_test.pkl.score_real_test.json'); v = f('$A/preds/nogate_fused_val.pkl.score_val.json')
print('test UDS %.2f AP %.2f ATE %.3f AOE %.2f ASE %.3f mDR %.2f | val UDS %.2f | vs PST test 60.81 / val 59.71' % (t['UDS'], t['AP'], t['ATE'], t['AOE'], t['ASE'], t['mDR'], v['UDS']))")
say "ALL DONE: $SUM"
# paired bootstrap vs PST afterwards: add ('PST', 'nogate') to review1113/clean3/boot12.py ARMS with $A/preds/nogate_fused_%s.pkl
