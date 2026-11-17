#!/bin/bash
# [cleanfix B5] Background-reliance test (X_suppl.tex:68: 7.89 / 8.61 / 16.13 / 18.65 deg) redone with the CLEAN banks.
#   = wf20/design/ctx_check.py (552 sim TEST crops, airframe pixels untouched; variants O / I (own inpainted scene) / S_v (other sim scene) /
#     B_ir, B_v (real single-frame background) / B_sf), via the generated copy ctx_check_clean.py (make_ctx_clean.py lists every change):
#     real bank -> clean orientation bank D:/S2MUAV3D/srcx_work/in2mav/ou_bank.npy; B_sf -> clean refiner bank srcx/in2mav/ref/bank.npy;
#     nets = the 5 RGB-only simulation KZ nets KZs0..4 (byte-identical to the dev ones); dev AdaBN family dropped.
#   Paper mapping: O -> 7.89, I -> 8.61, S_v -> 16.13, B_v -> 18.65 (KZ single-net 5-seed mean of geo_mean). O / I / S_v use no real data and
#   identical nets/crops -> expected to reproduce the dev values exactly; only B_ir / B_v / B_sf change.
# Inference only, no real label read. GPU ~3-4 min (dev run: 10 nets 227 s; here 5 nets). Output: ctx_check_test.json + logs/run.log,
#   done marker .done. Resume: per-network kp_KZ_s<k>.npz cache. OOM -> CC_BS halved (16 -> 8 -> ... -> 1). Workers 0 (no DataLoader iteration).
#   Waits at start for D:/cmpfull_runs/gupnet_uda_mt_s0/.done and review1113/initabl/.done (poll 10 min). Never kills processes.
#   launch: bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B5_bgrel/B5_bgrel.sh
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; B=$GP/review1113/cleanfix/B5_bgrel; BW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B5_bgrel
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset CC_SMOKE
mkdir -p $B/logs; LOG=$B/logs/run.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $B/.done ] && { say "already done"; exit 0; }
n=0; until [ -f /d/cmpfull_runs/gupnet_uda_mt_s0/.done ] && [ -f $GP/review1113/initabl/.done ]; do
  [ $n = 0 ] && say "waiting for gupnet_uda_mt_s0/.done and initabl/.done (poll 10 min)"; n=1; sleep 600; done
say "start B5 background reliance (clean banks)"
[ -f $B/ctx_check_clean.py ] || $PY $BW/make_ctx_clean.py >> $LOG 2>&1 || { say "!! make_ctx_clean"; exit 2; }
bs=16
while true; do
  if CC_BS=$bs $SLOT light $PY $BW/ctx_check_clean.py > $B/logs/ctx.try 2>&1; then cat $B/logs/ctx.try >> $B/logs/ctx.log; break; fi
  cat $B/logs/ctx.try >> $B/logs/ctx.log
  if grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" $B/logs/ctx.try && [ $bs -gt 1 ]; then
    bs=$((bs / 2)); say "OOM -> retry with CC_BS=$bs (inference batch only; results unaffected)"
  else say "!! ctx_check_clean failed (not OOM or batch 1), see logs/ctx.log"; exit 2; fi
done
rm -f $B/logs/ctx.try
[ -f $B/ctx_check_test.json ] || { say "!! output json missing"; exit 2; }
SUM=$($PY -c "
import json; r = json.load(open('$BW/ctx_check_test.json'))['mean_KZ']
print(' | '.join('%s %.2f' % (k, r[k]['geo_mean']) for k in ('O', 'I', 'S_v', 'B_ir', 'B_v', 'B_sf')) + '  (dev: O 7.89, I 8.61, S_v 16.13, B_ir 18.61, B_v 18.65, B_sf 14.38)')")
say "ALL DONE: $SUM"; touch $B/.done
