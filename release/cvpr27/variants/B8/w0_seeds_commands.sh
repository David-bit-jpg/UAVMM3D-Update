#!/bin/bash
# [review1113 B8] NOT RUN by its author. IR-only fusion control on PST seed runs 2 and 3, so that the fused-vs-IR-only gain (+1.44 test UDS on the
# reported run: PST 60.81 vs clean2/w0 59.37) rests on 3 runs.
# Recipe = review1113/clean2/cost/w0_commands.sh, applied to each seed run:
#   stage-3 depth students retrained with omega = 0 (targets = IR depth only: delta = log(z_pl / z_det0), no corner term), from that run's own
#   stage-2 weights, 2,000 steps, REF_REAL = d0, seeds as in that run (nominal 5, 6 + SX_SEEDSHIFT -> 15/16 resp. 25/26); everything else of the
#   run is copied unchanged (det0/det1, refiner sim samples, bank, real d0 crops, student readings, height readings, crop index pkls).
#   Then inference (train/val/test, seed average), deployment with THAT run's frozen omega (w_sim.json: s10 0.1731, s20 0.2050; = what the
#   reported w0 run did with 0.190), scoring val (sanity) and test (once). Compare against the seed run's own pcmd_fused (PST) numbers.
#     run s10 = review1113/sup3/run            (PST test 59.28 / val 59.54)
#     run s20 = review1113/seeds/mav_full_s20  (PST test 62.70 / val 59.87)
# Writes ONLY under review1113/B8/<run>/ (lib copy from make_w0seeds_lib.py redirects sx.O/W to $CL_AO; ~1.3 GB copied inputs per run).
#   The seed runs and their crop dirs are read only.
# Robustness: stage-3 training resumes (ref/resume_st3_s<j>.pt every 250 steps, from the seed lib's sup_patch) and every step is skipped once its
#   output exists; on CUDA OOM the batch is halved and the step retried (max 4; REF_TRAIN_BS 64 -> ..., inference REF_BS 128 -> ...), each retry in
#   logs/retry.log (a halved training batch is a recipe deviation). Workers 0 (refiner has no DataLoader). Inference via gpu_slot.sh light +
#   infer_wrap.py (memory fraction 0.2). Never kills processes.
# Time: per run st3 2 refiners concurrently ~16 min + inference 6 x ~3 min ~18 min + deploy/score ~2 min = ~36 min; both runs ~1 h 15 min alone
#   on the GPU, ~1.5-2 h with gpu_slot queueing.
#   launch:  bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/w0_seeds_commands.sh          (RUNS="s10 s20" default)
RUNS=${RUNS:-"s10 s20"}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=D:/Miniconda3/envs/city/python.exe; GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
B8=$GP/review1113/B8; mkdir -p $B8/logs
MLOG=$B8/logs/w0_seeds.log
msay() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $MLOG; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }

one_run() {   # $1 = s10 | s20
  local run=$1 X WSEED SHIFT
  case $run in s10) X=$GP/review1113/sup3/run; WSEED=D:/S2MUAV3D/srcx_work/in2mav_seed2; SHIFT=10 ;;
               s20) X=$GP/review1113/seeds/mav_full_s20; WSEED=D:/S2MUAV3D/srcx_work/in2mav_sd20; SHIFT=20 ;;
               *) msay "bad run $run"; return 2 ;; esac
  export SX_SEEDSHIFT=$SHIFT CL_AO=$B8/$run; local A=$CL_AO L=$B8/$run/lib
  mkdir -p $A/recs $A/ref $A/ou $A/preds $A/logs $A/work
  LOG=$A/logs/w0.log; RLOG=$A/logs/retry.log
  say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
  oomrun() {   # oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries
    local lg=$1; local -a vals=($2); shift 2; local try=0 kv k v
    while true; do
      if env "${vals[@]}" "$@" > $lg.try 2>&1; then cat $lg.try >> $lg; rm -f $lg.try; return 0; fi
      cat $lg.try >> $lg
      if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); local -a nv=()
        for kv in "${vals[@]}"; do k=${kv%%=*}; v=${kv#*=}; v=$((v / 2)); [ $v -lt 1 ] && v=1; nv+=("$k=$v"); done; vals=("${nv[@]}")
        echo "$(date '+%m-%d %H:%M:%S') OOM retry $try/4 with ${vals[*]} :: $* $( [[ "$*" == *" train "* ]] && echo '(RECIPE DEVIATION: halved training batch)')" | tee -a $RLOG >> $LOG
      else echo "$(date '+%m-%d %H:%M:%S') FAILED (not OOM or 4 retries used): $*  see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
    done
  }
  say "start $run (seed run $X, SX_SEEDSHIFT=$SHIFT): stage-3 retrain with omega = 0"
  $PY $B8/make_w0seeds_lib.py $run >> $LOG 2>&1 || { say "!! make_w0seeds_lib"; return 2; }
  $PY -c "
import sys; sys.path.insert(0, '$L'); import sx
assert sx.O == '$A' and sx.W == '$A/work' and sx.W_SEED == '$WSEED', (sx.O, sx.W)
print('[B8] O', sx.O, '| pseudo index', sx.PSEUDO_TAG)" >> $LOG 2>&1 || { say "!! sx config"; return 2; }
  # 0) read-only inputs of the seed run (copies)
  for f in recs/det0.pkl recs/det1.pkl ref/sim_meta.pkl ref/sim_rgb.npy ref/sim_sil.npy ref/bank.npy ref/real_d0_meta.pkl ref/real_d0_rgb.npy \
           ref/ref_st2_s5.pt ref/ref_st2_s6.pt ref/inf_st2_det1_train.pkl ou/read_tr1_stu.pkl ou/read_val1_stu.pkl ou/read_te1_stu.pkl \
           kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do [ -f $A/$f ] || cp $X/$f $A/$f || { say "!! copy $f"; return 2; }; done
  for n in tr1 val1 te1; do [ -f $A/work/oc_$n.pkl ] || cp $WSEED/oc_$n.pkl $A/work/ || { say "!! copy oc_$n"; return 2; }; done
  local WRUN; WRUN=$($PY -c "import json; print(json.load(open('$X/w_sim.json'))['w'])")
  # 1) omega = 0 targets (CPU, < 1 min)
  [ -f $A/ref/dp_fused.npy ] || $PY $L/chain_x.py targets 0 > $A/logs/targets_w0.log 2>&1 || { say "!! targets"; return 2; }
  # 2) stage-3 training, nominal seeds 5 and 6 (effective +SHIFT) in parallel, 2,000 steps, resumable
  for j in 5 6; do ( [ -f $A/ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$A/ref/dp_fused.npy oomrun $A/logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
  for j in 5 6; do [ -f $A/ref/ref_st3_s$j.pt ] || { say "!! ref st3 s$j missing"; return 2; }; done
  # 3) inference of both seeds on det1 crops (train for c_k, val for b'_k, test), seed average
  for sp in train val test; do
    for j in 5 6; do [ -f $A/ref/fin_s${j}_$sp.pkl ] || oomrun $A/logs/infer.log "REF_BS=128" $SLOT light $PY $L/infer_wrap.py $L/refiner_x.py infer $A/ref/ref_st3_s$j.pt $A/recs/det1.pkl $sp $A/ref/fin_s${j}_$sp.pkl || { say "!! infer $j $sp"; return 2; }; done
    [ -f $A/ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $A/ref/inf_fin_det1_$sp.pkl $A/ref/fin_s5_$sp.pkl $A/ref/fin_s6_$sp.pkl >> $A/logs/infer.log 2>&1
  done
  # 4) deployment (CPU) at the run's frozen omega (also writes the nofuse arm), scoring: val sanity, test once
  [ -f $A/deploy_w0.json ] || $PY $L/chain_x.py deploy $WRUN w0 > $A/logs/deploy.log 2>&1 || { say "!! deploy"; return 2; }
  for p in w0_fused_val w0_nofuse_val; do [ -f $A/preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $A/preds/$p.pkl val >> $A/logs/score.log 2>&1; done
  for p in w0_fused_test w0_nofuse_test; do [ -f $A/preds/$p.pkl.score_real_test.json ] || $PY $L/score_x.py $A/preds/$p.pkl real_test >> $A/logs/score.log 2>&1 || { say "!! score $p"; return 2; }; done
  local S; S=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
t, v = f('$A/preds/w0_fused_test.pkl.score_real_test.json'), f('$A/preds/w0_fused_val.pkl.score_val.json')
pt, pv = f('$X/preds/pcmd_fused_test.pkl.score_real_test.json'), f('$X/preds/pcmd_fused_val.pkl.score_val.json')
print('$run IR-only(w0) test UDS %.2f AP %.2f ATE %.3f | val UDS %.2f || PST test %.2f val %.2f || fused - IR-only: test %+.2f val %+.2f (w %.4f)' % (
    t['UDS'], t['AP'], t['ATE'], v['UDS'], pt['UDS'], pv['UDS'], pt['UDS'] - t['UDS'], pv['UDS'] - v['UDS'], $WRUN))")
  say "RUN DONE: $S"; echo "$S" >> $B8/logs/summary_$run.txt
}

msay "start B8 runs: $RUNS"
for r in $RUNS; do ( one_run $r ) || msay "!! run $r failed (re-launch resumes)"; done
SUM=$($PY -c "
import json, os
f = lambda p: json.load(open(p))['all']['UDS']
G = '$GP'
rows = [('s0 (reported)', G + '/srcx/in2mav/preds/pcmd_fused_%s.pkl.score_%s.json', G + '/review1113/clean2/w0/preds/w0_fused_%s.pkl.score_%s.json'),
        ('s10', G + '/review1113/sup3/run/preds/pcmd_fused_%s.pkl.score_%s.json', G + '/review1113/B8/s10/preds/w0_fused_%s.pkl.score_%s.json'),
        ('s20', G + '/review1113/seeds/mav_full_s20/preds/pcmd_fused_%s.pkl.score_%s.json', G + '/review1113/B8/s20/preds/w0_fused_%s.pkl.score_%s.json')]
out = {}
for sp, tag in (('test', 'real_test'), ('val', 'val')):
    d = []
    for n, a, b in rows:
        if os.path.exists(a % (sp, tag)) and os.path.exists(b % (sp, tag)):
            d.append((n, f(a % (sp, tag)), f(b % (sp, tag))))
    out[sp] = '%s mean fused-IRonly %+.2f over %d runs [%s]' % (sp, sum(x[1] - x[2] for x in d) / max(len(d), 1), len(d),
              ', '.join('%s %.2f vs %.2f (%+.2f)' % (n, p, w, p - w) for n, p, w in d))
print(out['test'] + ' | ' + out['val'])")
msay "ALL DONE: $SUM"
