#!/bin/bash
# [review1113 initabl] Supplement ablation (low priority): does initialising the step-2 orientation students from the IR+LiDAR teacher weights
# help, versus an RGB-only initialisation, everything else fixed?
#   arm (a) = the existing PST students of each run (init = same-seed IR+LiDAR KZ teacher, lidar/irlid/release/kzm_s{0,1}.pth). NOT retrained:
#             their scores are read from the run's preds/pcmd_fused_{val,test}.pkl.score_*.json.
#   arm (b) = the same 2 students per run retrained with OU_INIT = the same-seed RGB-only KZ crop keypoint net trained in simulation with the
#             same KZ recipe (sim_full_r34_kp_w12_zoom, 30 epochs, no IR/LiDAR auxiliary heads):
#               nominal seed 0 -> output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom/KZs0/ckpt/checkpoint_epoch_30.pth
#               nominal seed 1 -> .../KZs1/ckpt/checkpoint_epoch_30.pth
#             (= the student init of the SX_TEACH=none run srcx/in2mav_none, see its logs/ou_train_s*.log "init ..."; kzm_s* are the KZ-IRLiD
#             twins of these nets: cfg sim_full_r34_kp_w12_zoom_irlid, 30 epochs, lidar/irlid/PREREG_irlid.md).
#             Everything else identical: same run, same seeds (nominal 0/1 + that run's SX_SEEDSHIFT), same gated pseudo-orientations
#             (the run's pseudo_tr1.npz), crops, background bank, composites, mirror loss 12, 2250 steps, LR 2e-4, BN re-estimation (AdaBN on
#             real train crops), mirror-view readings, the run's height readings, stage-3 depth students (NOT retrained: inf_fin_det1_*),
#             detector det1, and the run's FROZEN w (w_sim.json; s0 0.1903, s10 0.1731, s20 0.2050). Then the run's normal deployment
#             (chain_x deploy) and scoring. Only the student readings (rotation + corner-PnP depth used in the deployment fusion) change.
#   runs: s0 = srcx/in2mav (reported), s10 = review1113/sup3/run, s20 = review1113/seeds/mav_full_s20.
# Real labels: read only by score_x.py. Val scored for all runs first; test scored ONCE at the very end (after all 3 runs' val scores exist).
# Writes ONLY under review1113/initabl/ (lib copies from make_initabl_lib.py; per run <run>/{ou,preds,recs,ref,logs}). The run dirs and their
# crop dirs D:/S2MUAV3D/srcx_work/{in2mav,in2mav_seed2,in2mav_sd20} are read only (no ori_x crops/bank/pseudo call).
# Robustness: waits at start until the paper-critical GUPNet-UDA runs are done (polls every 10 min); every GPU step goes through
#   gpu_slot.sh light; students resume from resume.pth (every 250 steps) and on CUDA OOM halve --bs-src/--bs-tgt (max 4; logged in
#   logs/retry.log as RECIPE DEVIATION); inference via infer_wrap.py (memory fraction 0.2), OOM -> ORI_BS halved; DataLoader workers 0
#   (ou_train_x / ori_x use num_workers=0); every step skipped once its output exists; per-run done marker <run>/.done, global .done.
#   Never kills processes.
# GPU time (from the runs' logs): 2 students concurrently ~21-24 min, student AdaBN x2 + val/test readings ~8 min, deploy/score CPU ~2 min
#   -> ~35 min per run, ~1 h 45 min for 3 runs alone on the GPU, ~2-2.5 h with gpu_slot queueing.
# Final line in logs/run.log: "ALL DONE: ..." with per-run and mean (b - a) AOE / UDS, val and test.
#   launch:  bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/initabl_commands.sh     (RUNS="s0 s10 s20" default)
RUNS=${RUNS:-"s0 s10 s20"}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset SX_SEEDSHIFT OU_INIT OU_BANK OU_CROPS
PY=D:/Miniconda3/envs/city/python.exe
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
IA=$GP/review1113/initabl; IAW=$GPW/review1113/initabl
SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
RGB0=$M/sim_full_r34_kp_w12_zoom/KZs0/ckpt/checkpoint_epoch_30.pth; RGB1=$M/sim_full_r34_kp_w12_zoom/KZs1/ckpt/checkpoint_epoch_30.pth
mkdir -p $IA/logs
LOG=$IA/logs/run.log; RLOG=$IA/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }; die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
oomrun() {   # oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries
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

if [ -f $IA/.done ]; then say "already done (remove $IAW/.done to recompute the summary)"; tail -1 $LOG; exit 0; fi
# ---------------- never compete with the paper-critical GUPNet-UDA runs
n=0; until [ -f /d/cmpfull_runs/gupnet_uda_mt_s0/.done ] && [ -f /d/cmpfull_runs/gupnet_uda_dann_s0/.done ]; do
  [ $n = 0 ] && say "waiting for D:/cmpfull_runs/gupnet_uda_{mt,dann}_s0/.done (poll every 10 min)"; n=1; sleep 600; done
say "start initabl (student init: IR+LiDAR teacher vs RGB-only KZ net), runs: $RUNS"
for f in $RGB0 $RGB1; do [ -f $f ] || die "RGB-only init missing: $f"; done

one_run() {   # $1 = s0 | s10 | s20 : arm (b) students -> AdaBN -> mirror readings (val/test) -> deploy at the frozen w -> val score
  local run=$1 X XW WK SHIFT A AW L
  case $run in s0)  X=$GP/srcx/in2mav; XW=$GPW/srcx/in2mav; WK=D:/S2MUAV3D/srcx_work/in2mav; SHIFT= ;;
               s10) X=$GP/review1113/sup3/run; XW=$GPW/review1113/sup3/run; WK=D:/S2MUAV3D/srcx_work/in2mav_seed2; SHIFT=10 ;;
               s20) X=$GP/review1113/seeds/mav_full_s20; XW=$GPW/review1113/seeds/mav_full_s20; WK=D:/S2MUAV3D/srcx_work/in2mav_sd20; SHIFT=20 ;;
               *) say "bad run $run"; return 2 ;; esac
  if [ -n "$SHIFT" ]; then export SX_SEEDSHIFT=$SHIFT; else unset SX_SEEDSHIFT; fi
  A=$IA/$run; AW=$IAW/$run; L=$AW/lib
  [ -f $A/.done ] && { say "$run already done"; return 0; }
  mkdir -p $A/recs $A/ref $A/ou/runs $A/preds $A/logs; cd $A
  local RL=$A/logs/run.log
  rsay() { echo "$(date '+%m-%d %H:%M:%S') [$run] $1" | tee -a $RL >> $LOG; }
  rsay "start (baseline run $XW, crops $WK, SX_SEEDSHIFT=${SHIFT:-0})"
  $PY $IAW/make_initabl_lib.py $run >> $RL 2>&1 || { rsay "!! make_initabl_lib"; return 2; }
  $PY -c "
import sys; sys.path.insert(0, '$L'); import sx
assert sx.O == '$AW' and sx.W == '$WK' and (sx.SRC, sx.TGT, sx.TEACH) == ('in', 'mav', 'full'), (sx.O, sx.W)
print('[initabl] O', sx.O, '| W (read only)', sx.W)" >> $RL 2>&1 || { rsay "!! sx config"; return 2; }
  # 0) fixed inputs of the run (copies; read only originals)
  local f
  for f in recs/det1.pkl ref/inf_fin_det1_train.pkl ref/inf_fin_det1_val.pkl ref/inf_fin_det1_test.pkl kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do
    [ -f $A/$f ] || cp $X/$f $A/$f || { rsay "!! copy $f"; return 2; }; done
  [ -f $A/w_frozen.json ] || cp $X/w_sim.json $A/w_frozen.json || { rsay "!! copy w_sim"; return 2; }
  for f in pseudo_tr1.npz oc_tr1.npy oc_tr1.pkl oc_val1.npy oc_val1.pkl oc_te1.npy oc_te1.pkl ou_bank.npy; do
    [ -f $WK/$f ] || { rsay "!! missing crop-dir input $WK/$f"; return 2; }; done
  local WRUN; WRUN=$($PY -c "import json; print(json.load(open('$AW/w_frozen.json'))['w'])") || return 2
  # 1) arm (b) students: nominal seeds 0, 1 (effective +SHIFT), OU_INIT = same-seed RGB-only KZ net; recipe = run_x.sh line for S6I
  local INIT=($RGB0 $RGB1)
  train_student() {   # $1 = seed; resume from runs/S6I_s<seed>/resume.pth; OOM -> halve --bs-src/--bs-tgt (recipe deviation)
    local s=$1 bs=8 bt=4 try=0 lg=$A/logs/ou_train_s$1.log
    while [ ! -f $A/ou/runs/S6I_s$s/ckpt_last.pth ]; do
      OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $AW/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
      cat $lg.try >> $lg
      [ -f $A/ou/runs/S6I_s$s/ckpt_last.pth ] && break
      if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
        echo "$(date '+%m-%d %H:%M:%S') [$run] student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
      else echo "$(date '+%m-%d %H:%M:%S') [$run] student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
    done; rm -f $lg.try
  }
  for s in 0 1; do train_student $s & sleep 100; done; wait     # 2 concurrent trainings, as run_x.sh
  for s in 0 1; do [ -f $A/ou/runs/S6I_s$s/ckpt_last.pth ] || { rsay "!! student s$s missing"; return 2; }
    grep -h -m1 "INFO  init " $A/logs/ou_train_s$s.log | sed 's/ | steps.*//' | sed "s/^/[$run s$s] /" >> $RL; done
  grep -q "sim_full_r34_kp_w12_zoom/KZs0/" $A/logs/ou_train_s0.log && grep -q "sim_full_r34_kp_w12_zoom/KZs1/" $A/logs/ou_train_s1.log || { rsay "!! students not initialised from the RGB-only nets"; return 2; }
  rsay "students done"
  # 2) BN re-estimation on real train crops (cap 0 = all, as run_x.sh), mirror-view readings on val/test (train readings are only used by
  #    the stage-3 targets, which are not recomputed)
  for s in 0 1; do [ -f $A/ou/S_ab$s.pth ] || oomrun $A/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $AW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $AW/ou/S_ab$s.pth 0 || { rsay "!! student adabn $s"; return 2; }; done
  for n in val1 te1; do [ -f $A/ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun $A/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read $n $AW/ou/read_${n}_stu.pkl $AW/ou/S_ab0.pth,$AW/ou/S_ab1.pth || { rsay "!! read $n"; return 2; }; done
  # 3) deployment (CPU) at the run's frozen w; val score (sanity / comparison); test is scored at the end of the whole script
  [ -f $A/deploy_initb.json ] || $PY $L/chain_x.py deploy $WRUN initb > $A/logs/deploy.log 2>&1 || { rsay "!! deploy"; return 2; }
  [ -f $A/preds/initb_fused_val.pkl.score_val.json ] || $PY $L/score_x.py $AW/preds/initb_fused_val.pkl val >> $A/logs/score.log 2>&1 || { rsay "!! score val"; return 2; }
  rsay "val done (w $WRUN)"; touch $A/.done
}

for r in $RUNS; do ( one_run $r ) || say "!! run $r failed (re-launch resumes)"; done
for r in $RUNS; do [ -f $IA/$r/.done ] || die "run $r not done -> test NOT scored (re-launch resumes)"; done
# ---------------- test: scored once, after every run's val score exists
for r in $RUNS; do
  case $r in s0) SS= ;; s10) SS=10 ;; s20) SS=20 ;; esac
  if [ -n "$SS" ]; then export SX_SEEDSHIFT=$SS; else unset SX_SEEDSHIFT; fi
  [ -f $IA/$r/preds/initb_fused_test.pkl.score_real_test.json ] || $PY $IAW/$r/lib/score_x.py $IAW/$r/preds/initb_fused_test.pkl real_test >> $IA/$r/logs/score.log 2>&1 || die "score test $r"
done
SUM=$($PY -c "
import json, os
f = lambda p: json.load(open(p))['all']
G = '$GPW'; IA = '$IAW'
base = {'s0': G + '/srcx/in2mav', 's10': G + '/review1113/sup3/run', 's20': G + '/review1113/seeds/mav_full_s20'}
out = []
for sp, tag in (('val', 'val'), ('test', 'real_test')):
    rows = []
    for r in '$RUNS'.split():
        a = f('%s/preds/pcmd_fused_%s.pkl.score_%s.json' % (base[r], sp, tag)); b = f('%s/%s/preds/initb_fused_%s.pkl.score_%s.json' % (IA, r, sp, tag))
        rows.append((r, a, b))
    dA = [b['AOE'] - a['AOE'] for _, a, b in rows]; dU = [b['UDS'] - a['UDS'] for _, a, b in rows]
    out.append('%s: mean(b-a) AOE %+.2f UDS %+.2f over %d runs [%s]' % (sp, sum(dA) / len(dA), sum(dU) / len(dU), len(rows),
               '; '.join('%s AOE %.2f->%.2f UDS %.2f->%.2f' % (r, a['AOE'], b['AOE'], a['UDS'], b['UDS']) for r, a, b in rows)))
    json.dump([dict(run=r, split=sp, a=a, b=b) for r, a, b in rows], open('%s/summary_%s.json' % (IA, sp), 'w'), indent=1, default=float)
print(' || '.join(out) + ' || a = IR+LiDAR-teacher init (PST), b = RGB-only KZ init; AOE deg (lower better), UDS higher better')") || die summary
say "ALL DONE: $SUM"; touch $IA/.done
