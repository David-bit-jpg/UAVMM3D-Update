#!/bin/bash
# [review1113 boxhull] NOT RUN by its author. Rendered-truth control of the depth refiner (Module 4): the IR-silhouette target of the simulated
# samples is replaced by the FILLED CONVEX HULL OF THE PROJECTED 3-D GT BOX on the same simulated crop (same samples, same crop window).
# Everything else = the clean PST chain's refiner (srcx/lib/refiner_x.py stages 1-3, seeds 5 and 6, same recipe / steps / batches / losses),
# followed by the clean chain's deploy + score (as review1113/clean2/cost/w0_commands.sh).
#
# usage:  bash boxhull_commands.sh                 (both targets, MAV6D first)      TARGETS="mav" bash boxhull_commands.sh   (one target)
#
# What is re-run per target (combo C = in2mav | out2antifs; PST = SX_TEACH=full), outputs ONLY under review1113/boxhull/$C:
#   a) `refiner_x.py sim` (patched copy): re-runs the sim-only detector s0 on source-sim train to rebuild the deployment crop geometry,
#      and builds the box hull with that geometry (regen_sim_*); align_hull.py checks the regenerated samples reproduce the clean list
#      (same frames, |dw| < 0.5 px, |d delta| < 0.01) and swaps ONLY the silhouette array (else FALLBACK to the regenerated set, logged).
#      Consequences of the swap inside the unchanged code: the sim silhouette BCE+Dice target = hull; the cross-domain composites use the
#      hull as alpha (pasting the whole sim hull region, background inside the hull included); stage-2/3 real-crop silhouette targets are the
#      EMA teacher's own predictions (unchanged rule) and so follow the hull-trained network; colour classes (silhouette brightness) too.
#   b) refiner stages 1, 2 (seeds 5, 6), stage-2 inference on det1 train (colour classes for the stage-3 targets),
#      chain_x targets at the clean chain's FROZEN w (srcx/$C/w_sim.json), stage 3 (seeds 5, 6), inference train/val/test, seed average.
#   c) chain_x deploy at the frozen w (prefix boxhull) -> preds/boxhull_{fused,nofuse}_{val,test}.pkl; score val, then test once.
# Held FIXED (copied, read only): detectors det0/det1 and Module-3 pseudo-labels (built from the clean stage-2 refiner on det0; re-training
#   the self-trained detectors is out of scope), real background bank, real d0 crops, orientation-student readings, height readings, w.
#
# Robustness: CUDA OOM -> the step's batch env halves (max 4 retries, logged to logs/retry.log; a halved TRAINING batch = recipe deviation);
#   training resumes from its last state (stage 1 per epoch, stages 2/3 every 250 steps; sup_patch); every step is skipped once its output
#   exists; workers = 0 everywhere (refiner has no DataLoader workers; the sim loader is workers=0); GPU steps go through gproto/gpu_slot.sh.
# Start gate: waits until review1113/seeds/logs/run_all_seeds.log contains the wrapper's final line "run_all_seeds end" (written by
#   run_all_seeds.sh after all 7 seed runs, success or not), so it never overlaps the seed chains. NO_WAIT=1 skips the gate.
# Estimated wall time (from the clean chains' logs, GPU alone): MAV6D ~1 h 55 min (sim 13 min, stage 1 ~27, stage 2 ~15, st2 infer ~8,
#   stage 3 ~16, final infer ~18, copies/CPU ~10), AntiUAV3D ~1 h 20 min (sim 4, stage 1 ~13, stage 2 ~20, st2 infer ~3, stage 3 ~18,
#   infer ~11, CPU ~5)  ->  ~3 h 15 min for both, ~4-5 h with gpu_slot queueing.
set -u
BH=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull; BHW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; L=$BHW/lib
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; IW="$PY $L/infer_wrap.py"
export SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
SEEDLOG=$GP/review1113/seeds/logs/run_all_seeds.log
TARGETS=${TARGETS:-"mav antifs"}
mkdir -p $BH/logs; GLOG=$BH/logs/boxhull.log
gsay() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $GLOG; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }

# ---------------- start gate: the seed chains must be finished
if [ "${NO_WAIT:-0}" != 1 ]; then
  n=0; until grep -q "run_all_seeds end" "$SEEDLOG" 2>/dev/null; do [ $n = 0 ] && gsay "waiting for 'run_all_seeds end' in $SEEDLOG"; n=1; sleep 300; done
  gsay "seed chains finished: $(grep 'run_all_seeds end' $SEEDLOG | tail -1)"
fi
$PY $BHW/make_boxhull_lib.py >> $GLOG 2>&1 || { gsay "!! make_boxhull_lib failed"; exit 2; }

run_target() {   # $1 = mav | antifs
  local TGT=$1 SRC; case $TGT in mav) SRC=in ;; antifs) SRC=out ;; *) gsay "!! unknown target $TGT"; return 2 ;; esac
  local C=${SRC}2${TGT}; export SX_SRC=$SRC SX_TGT=$TGT
  local X=$BH/$C XW=$BHW/$C F=$GP/srcx/$C FW=/d/S2MUAV3D/srcx_work/$C
  mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs $X/work; cd $X || return 2
  local LOG=$X/logs/run.log RLOG=$X/logs/retry.log
  say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG >> $GLOG; }; die() { say "!! $1"; return 2; }
  # oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries (= seeds/run_seed_chain.sh)
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
  say "start boxhull $C"
  [ -f $F/preds/pcmd_fused_test.pkl ] || { die "clean chain $C not finished"; return 2; }
  # sx config check: outputs must land under review1113/boxhull/$C
  $PY -c "import sys; sys.path.insert(0, '$L'); import sx; assert sx.O == '$XW' and sx.W == '$XW/work' and sx.TEACH == 'full', (sx.O, sx.W); print('sx ok', sx.O, sx.REAL, sx.PSEUDO_TAG)" >> $LOG 2>&1 || { die "sx config"; return 2; }
  local DET0; DET0=$($PY -c "import sys; sys.path.insert(0, '$L'); import sx; print(sx.S['det'][0])") || { die "det0 path"; return 2; }
  local WW; WW=$($PY -c "import json; print(json.load(open('$(echo $F | sed 's|^/e/|E:/|')/w_sim.json'))['w'])") || { die "w_sim"; return 2; }
  say "frozen w (clean chain sim-val inverse variance) = $WW; sim detector s0 = $DET0"

  # ---- 0) read-only inputs of the clean chain (copies)
  local f
  for f in det0.pkl det1.pkl; do [ -f recs/$f ] || cp $F/recs/$f recs/ || { die "copy recs/$f"; return 2; }; done
  for f in sim_meta.pkl sim_rgb.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy; do [ -f ref/$f ] || [ -f ref/clean_$f ] || cp $F/ref/$f ref/ || { die "copy ref/$f"; return 2; }; done
  [ -f ref/clean_sim_sil.npy ] || cp $F/ref/sim_sil.npy ref/clean_sim_sil.npy || { die "copy clean sim_sil"; return 2; }   # IR silhouettes, for the alignment check only
  for f in read_tr1_stu.pkl read_val1_stu.pkl read_te1_stu.pkl; do [ -f ou/$f ] || cp $F/ou/$f ou/ || { die "copy ou/$f"; return 2; }; done
  for f in kps_tr1.pkl kps_val1.pkl kps_te1.pkl w_sim.json; do [ -f $f ] || cp $F/$f . || { die "copy $f"; return 2; }; done
  for f in oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do [ -f work/$f ] || cp $FW/$f work/ || { die "copy work/$f"; return 2; }; done   # chain_x.zL_table reads OO.W/oc_*.pkl (pkl only)

  # ---- a) sim samples with the box hull (GPU: sim-only detector s0 on source-sim train), then align to the clean sample list
  [ -f ref/regen_sim_meta.pkl ] || oomrun logs/ref_sim.log "REF_SIM_BS=8" $SLOT light $IW $L/refiner_x.py sim $DET0 || { die "ref sim (box hull)"; return 2; }
  [ -f ref/sim_sil.npy ] || $PY $BHW/align_hull.py $XW >> $LOG 2>&1 || { die "align_hull"; return 2; }
  grep -o '"mode": "[A-Z]*' ref/align_hull.json | sed 's/"mode": "/sample alignment: /' | while read -r l; do say "$l"; done

  # ---- b1) stages 1 -> 2, seeds 5 and 6 in parallel (as run_x.sh)
  local j
  st12() { [ -f ref/ref_st1_s$1.pt ] || oomrun logs/ref_st1_s$1.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $1 || return 1
           [ -f ref/ref_st2_s$1.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$1.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $1; }
  for j in 5 6; do st12 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || { die "refiner st2 s$j"; return 2; }; done
  # ---- b2) stage-2 inference on det1 train (colour classes for the stage-3 targets), seed average
  for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || { die "inf st2 det1 $j"; return 2; }; done
  [ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1 || { die "avg st2"; return 2; }
  # ---- b3) fused stage-3 pseudo-targets at the frozen w (CPU), stage 3 seeds 5 and 6 in parallel
  [ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW > logs/targets.log 2>&1 || { die "targets"; return 2; }
  st3() { [ -f ref/ref_st3_s$1.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$1.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $1; }
  for j in 5 6; do st3 $j & sleep 30; done; wait
  for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || { die "refiner st3 s$j"; return 2; }; done
  # ---- b4) final inference (det1 crops) train / val / test, seed average
  local sp
  for sp in train val test; do
    for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || { die "inf fin $j $sp"; return 2; }; done
    [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1 || { die "avg fin $sp"; return 2; }
  done
  # ---- c) deployment at the frozen w (writes fused + nofuse), score: val (sanity), test once
  [ -f deploy_boxhull.json ] || $PY $L/chain_x.py deploy $WW boxhull > logs/deploy.log 2>&1 || { die deploy; return 2; }
  local SPLIT; SPLIT=$($PY -c "import sys; sys.path.insert(0, '$L'); import sx; print(sx.T['score_split'])")
  local p
  for p in boxhull_fused_val boxhull_nofuse_val; do [ -f preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/$p.pkl val >> $LOG 2>&1; done
  for p in boxhull_fused_test boxhull_nofuse_test; do [ -f preds/$p.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/$p.pkl $SPLIT >> $LOG 2>&1 || { die "score $p"; return 2; }; done
  local r; r=$($PY -c "
import json; a = json.load(open('$XW/preds/boxhull_fused_test.pkl.score_$SPLIT.json'))['all']; b = json.load(open('$F/preds/pcmd_fused_test.pkl.score_$SPLIT.json'))['all']
print('test boxhull UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f | clean PST UDS %.2f AP %.2f ATE %.3f' % (a['UDS'], a['AP'], a['ATE'], a['ASE'], a['AOE'], a['mDR'], b['UDS'], b['AP'], b['ATE']))")
  say "ALL DONE: $C $r"
}

fails=()
for t in $TARGETS; do
  if run_target $t; then :; else fails+=($t); gsay "!! boxhull $t failed (see $BH/*2$t/logs/run.log)"; fi
done
gsay "boxhull end; failed: ${fails[*]:-none}"
[ ${#fails[@]} = 0 ]
