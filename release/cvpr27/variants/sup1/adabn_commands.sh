#!/bin/bash
# [sup1] NOT RUN by its author. AdaBN baseline (inference only): the clean MAV6D chain's sim-only detector ensemble (3 seeds behind
# srcx/in2mav/preds/srconly_{val,test}.pkl = sx.S['det'] = output/release_20260920/ramp_s{0,1,2}.pth) with every BatchNorm layer's statistics
# re-estimated on the 12,866 UNLABELED MAV6D train frames (tools/eval_camnorm.adapt_bn: cumulative average, no gradient, weights untouched),
# then predicted on val + test exactly as the chain's srconly predictions (det_infer_x: ensemble + flip, MAX_OBJ 1, width-only geometric depth,
# s_bar self-calibrated on unlabeled val by the same — here adapted — ensemble) and scored with srcx/lib/score_x.py (mav rules): val, then test once.
# Compare with srconly (test UDS 49.51) and PST (test 60.81 / val 59.71).
# Writes ONLY under review1113/sup1/adabn/ (bn/, recs/, preds/, logs/). Reads srcx/lib (code), the 3 checkpoints, the MAV6D cache.
# Robustness: inference through gpu_slot.sh light, torch memory fraction 0.2 (inside det_infer_adabn.py), DataLoader workers 0; on CUDA OOM the
#   batch ($DET_BS 8 -> 4 -> 2 -> 1) is halved and the step retried (max 4 retries), each retry logged to logs/retry.log (a halved batch changes the
#   AdaBN variance estimate slightly: it is reported). Restart-safe: adapted BN states are cached in bn/bn_s<k>.pth, finished steps are skipped.
# Time: ~20-30 min (AdaBN pass 3 x 12,866 frames ~8 min; s_bar + val + test with ensemble + flip ~10-12 min; det0 infer of train+val+test took 25 min).
#   launch:  bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1/adabn_commands.sh
#   short smoke test of the AdaBN path (uses the GPU via gpu_slot, ~2 min, writes only bn/*.smoke):  SMOKE=5 bash .../sup1/adabn_commands.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; S1=$GP/review1113/sup1; A=$S1/adabn; SL=$GP/srcx/lib
mkdir -p $A/bn $A/recs $A/preds $A/logs
LOG=$A/logs/adabn.log; RLOG=$A/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
oomrun() {   # oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries, each attempt logged
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
if [ -n "$SMOKE" ]; then
  $SLOT light $PY $S1/det_infer_adabn.py --out $A/recs/smoke_unused.pkl --bn-dir $A/bn --smoke $SMOKE 2>&1 | tail -5; exit $?
fi
say "start AdaBN baseline (sim-only detector ensemble, BN on 12,866 unlabeled MAV6D train frames)"

# 1) AdaBN + ensemble/flip decoding of val and test (one GPU step; BN states cached, so a retry after OOM resumes from the finished members)
[ -f $A/recs/det0_adabn.pkl ] || oomrun $A/logs/det0_adabn_infer.log "DET_BS=8" $SLOT light $PY $S1/det_infer_adabn.py --out $A/recs/det0_adabn.pkl --bn-dir $A/bn --splits val test --interval 1 || die "adabn infer"
grep -hE "AdaBN|s_bar|frames, conf" $A/logs/det0_adabn_infer.log | tail -6 | tee -a $LOG
echo "reference (no AdaBN, srcx/in2mav/logs/det0_infer.log):" | tee -a $LOG; grep -hE "s_bar|^val|^test" $GP/srcx/in2mav/logs/det0_infer.log | tee -a $LOG

# 2) prediction pickles (= run_x.sh to_pred.py) and scoring: val sanity, then test ONCE
for sp in val test; do [ -f $A/preds/adabn_$sp.pkl ] || $PY $SL/to_pred.py $A/recs/det0_adabn.pkl $sp $A/preds/adabn_$sp.pkl >> $LOG 2>&1 || die "to_pred $sp"; done
[ -f $A/preds/adabn_val.pkl.score_val.json ] || $PY $SL/score_x.py $A/preds/adabn_val.pkl val >> $LOG 2>&1 || die "score val"
[ -f $A/preds/adabn_test.pkl.score_real_test.json ] || $PY $SL/score_x.py $A/preds/adabn_test.pkl real_test >> $LOG 2>&1 || die "score test"
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
t = f('$A/preds/adabn_test.pkl.score_real_test.json'); v = f('$A/preds/adabn_val.pkl.score_val.json')
s = f('$GP/srcx/in2mav/preds/srconly_test.pkl.score_real_test.json')
print('AdaBN test UDS %.2f AP %.2f ATE %.3f AOE %.2f ASE %.3f mDR %.2f | val UDS %.2f | srconly test UDS %.2f, PST test 60.81 / val 59.71' % (
    t['UDS'], t['AP'], t['ATE'], t['AOE'], t['ASE'], t['mDR'], v['UDS'], s['UDS']))")
say "ALL DONE: $SUM"
