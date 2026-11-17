#!/bin/bash
# [clean3 item 4] NOT RUN. Retrain the clean MAV6D chain's orientation students WITHOUT the mirror loss (--mir 0; everything else = run_x.sh:
# S6I recipe, gated pseudo-rotations pseudo_tr1, init = same-seed IR+LiDAR member, background swap 0.5, 2,250 steps, bs 8 sim + 4 real, mem 0.3),
# seeds 0 and 1, then AdaBN, readings (test-time mirror VIEW kept, as deployed; MIRVIEW=0 drops it too), deploy with the frozen omega and score
# val + test once.
#   Default (FULL=0): depth refiners, c_k, r_k, s_bar stay frozen (copied); only rotations and the corner-PnP distance used by the test-time
#     fusion change (b_k is re-estimated from unlabeled val images by the deploy rule, as in the clean chain).   ~35 min wall.
#   FULL=1: additionally re-reads train crops and retrains the step-3 depth refiners (seeds 5, 6) on fused targets built from the new students'
#     corner distances (omega kept frozen at 0.190: w_sim would need the sim-val student readings, see CLEAN3_REPORT), then re-infers.  ~76 min wall.
# Writes ONLY under review1113/clean3/nomir/ (lib copies from make_nomir_lib.py redirect sx.O/sx.W to $CL_AO). Deployed files are read only:
#   srcx/in2mav/{recs,ref,kps_*}, D:/S2MUAV3D/srcx_work/in2mav/{oc_*.npy,ou_bank.npy} (crop arrays read in place via OU_CROPS / OU_BANK / _wf()).
# OOM handling: any step whose log shows a CUDA out-of-memory error is retried with its batch halved (students: --bs-src/--bs-tgt 8/4 -> 4/2 -> 2/1;
#   AdaBN/readings: ORI_BS 16 -> 8 -> ... -> 1; refiner inference REF_BS 128 -> ... -> 1; refiner training REF_TRAIN_BS 64 -> ... -> 8);
#   every halving is appended to logs/oom.log (a halved student batch is a recipe deviation and must be reported). All loaders use workers 0.
set -e
FULL=${FULL:-0}; MIRVIEW=${MIRVIEW:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; C3=$GP/review1113/clean3; L=$C3/nomir/lib
X=$GP/srcx/in2mav; XW=D:/S2MUAV3D/srcx_work/in2mav
export CL_AO=$C3/nomir; A=$CL_AO
export OU_CROPS=$XW/oc_tr1.npy OU_BANK=$XW/ou_bank.npy          # read in place (5.7 GB / 2.1 GB, not copied)
W_FROZEN=0.190282282581008                                       # = srcx/in2mav/w_sim.json (deployed omega)
mkdir -p $A/recs $A/ref $A/ou/runs $A/preds $A/logs $A/work
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $A/logs/nomir.log; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }

# 0) lib copies + read-only inputs (~20 MB; FULL adds ~1.2 GB)
$PY $C3/make_nomir_lib.py >> $A/logs/nomir.log 2>&1
cp $X/recs/det1.pkl $A/recs/; cp $X/kps_tr1.pkl $X/kps_val1.pkl $X/kps_te1.pkl $A/
cp $XW/pseudo_tr1.npz $XW/oc_tr1.pkl $XW/oc_val1.pkl $XW/oc_te1.pkl $A/work/
if [ $FULL = 0 ]; then cp $X/ref/inf_fin_det1_train.pkl $X/ref/inf_fin_det1_val.pkl $X/ref/inf_fin_det1_test.pkl $A/ref/
else cp $X/recs/det0.pkl $A/recs/
  for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st2_s5.pt ref_st2_s6.pt inf_st2_det1_train.pkl; do cp $X/ref/$f $A/ref/; done
fi
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx
print('IRL0=%s IRL1=%s' % tuple(sx.S['irlid']))")"
INIT=($IRL0 $IRL1)
say "start nomir FULL=$FULL MIRVIEW=$MIRVIEW"

# 1) students without the mirror loss, seeds 0 and 1 concurrently (as run_x.sh; ~25 min wall measured)
train_student() {   # $1 = seed
  local s=$1 bs=8 bt=4 lg=$A/logs/ou_train_s$1.log
  while true; do
    rm -rf $A/ou/runs/NOMIR_s$s
    if OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag NOMIR --tgt gate --mir 0 --pseudo pseudo_tr1 \
         --runs-dir $A/ou/runs --bs-src $bs --bs-tgt $bt > $lg 2>&1 && [ -f $A/ou/runs/NOMIR_s$s/ckpt_last.pth ]; then return 0; fi
    if isoom $lg && [ $bt -gt 1 ]; then mv $lg $lg.oom_bs${bs}_${bt}; bs=$((bs / 2)); bt=$((bt / 2))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry with --bs-src $bs --bs-tgt $bt (RECIPE DEVIATION)" >> $A/logs/oom.log
    else echo "student s$s failed (not OOM or batch already 1), see $lg" >> $A/logs/oom.log; return 1; fi
  done
}
for s in 0 1; do ( train_student $s ) & sleep 100; done; wait
for s in 0 1; do [ -f $A/ou/runs/NOMIR_s$s/ckpt_last.pth ] || { say "!! student s$s missing"; exit 2; }; done
say "students done"

# generic OOM-halving wrapper for inference / refiner steps:  oomrun <batch env var> <start> <min> <log> <cmd...>
oomrun() {
  local var=$1 b=$2 bmin=$3 lg=$4; shift 4
  while true; do
    if env $var=$b "$@" >> $lg 2>&1; then return 0; fi
    if isoom $lg && [ $b -gt $bmin ]; then b=$((b / 2)); echo "$(date '+%m-%d %H:%M:%S') OOM in: $* -> retry with $var=$b" >> $A/logs/oom.log
    else echo "failed (not OOM or batch at minimum): $*" >> $A/logs/oom.log; return 1; fi
  done
}

# 2) AdaBN on all train crops (cap 0), as run_x.sh (~3 min)
for s in 0 1; do oomrun ORI_BS 16 1 $A/logs/adabn_s$s.log $SLOT light $PY $L/ori_x.py adabn $A/ou/runs/NOMIR_s$s/ckpt_last.pth tr1 $A/ou/S_ab$s.pth 0; done

# 3) readings of the 2-student ensemble (+ test-time mirror view unless MIRVIEW=0): val1, te1 (~4.5 min); tr1 only for FULL (+6.6 min)
SETS="val1 te1"; [ $FULL = 1 ] && SETS="tr1 val1 te1"
for n in $SETS; do ORI_MIRROR=$MIRVIEW oomrun ORI_BS 16 1 $A/logs/read_$n.log $SLOT light $PY $L/ori_x.py read $n $A/ou/read_${n}_stu.pkl $A/ou/S_ab0.pth,$A/ou/S_ab1.pth; done
say "readings done"

# 4) FULL only: step-3 targets from the new corner distances (omega frozen), refiners s5 || s6 (2,000 steps), inference train/val/test, seed mean
if [ $FULL = 1 ]; then
  $PY $L/chain_x.py targets $W_FROZEN > $A/logs/targets.log 2>&1
  for j in 5 6; do ( REF_REAL=d0 REF_DP=$A/ref/dp_fused.npy oomrun REF_TRAIN_BS 64 8 $A/logs/ref_st3_s$j.log $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
  for j in 5 6; do [ -f $A/ref/ref_st3_s$j.pt ] || { say "!! ref st3 s$j missing"; exit 2; }; done
  for sp in train val test; do
    for j in 5 6; do oomrun REF_BS 128 1 $A/logs/infer.log $SLOT light $PY $L/refiner_x.py infer $A/ref/ref_st3_s$j.pt $A/recs/det1.pkl $sp $A/ref/fin_s${j}_$sp.pkl; done
    $PY $GP/srcx/lib/avg_inf.py $A/ref/inf_fin_det1_$sp.pkl $A/ref/fin_s5_$sp.pkl $A/ref/fin_s6_$sp.pkl >> $A/logs/infer.log 2>&1
  done
fi

# 5) deployment (CPU, frozen omega; also writes the nofuse arm) and scoring: val sanity, test ONCE
$PY $L/chain_x.py deploy $W_FROZEN nomir > $A/logs/deploy.log 2>&1
for p in nomir_fused_val nomir_nofuse_val; do $PY $L/score_x.py $A/preds/$p.pkl val >> $A/logs/score.log 2>&1; done
for p in nomir_fused_test nomir_nofuse_test; do $PY $L/score_x.py $A/preds/$p.pkl real_test >> $A/logs/score.log 2>&1; done
say "ALL DONE: $(grep -h -o '"UDS": [0-9.]*' $A/preds/nomir_fused_test.pkl.score_real_test.json | head -1) (test)  vs PST 60.81"
# paired bootstrap vs PST afterwards: add ('PST', 'nomir') to review1113/clean3/boot12.py ARMS with $A/preds/nomir_fused_%s.pkl
