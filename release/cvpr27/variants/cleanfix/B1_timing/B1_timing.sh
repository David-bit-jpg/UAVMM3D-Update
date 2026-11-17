#!/bin/bash
# [cleanfix B1] timing of the CLEAN deployed networks (srcx/in2mav) -> replaces paper 4_experiments.tex:39 "316 / 133 ms ... development networks".
# Method = review1113/M12/timing_m12.py (M12_REPORT.md l.49-66): modes full / single / det, batch 1, 30 warm-up + 300 timed MAV6D test frames,
# cuda.synchronize per stage, memory fraction 0.3, peak = max_memory_allocated. Only the networks and code paths are the clean chain's
# (timing_clean.py header). AntiUAV3D was not timed in M12 either (same architectures / 512x288 input) -> not timed here.
# Inference only, no labels. Robustness: waits for GUPNet-UDA-MT + initabl done markers (poll 10 min); GPU via gpu_slot.sh light;
#   batch is 1 by definition, so OOM handling = retry the mode up to 3 times with MEMFRAC 0.3 -> 0.4 -> 0.5 (logged in retry.log);
#   DataLoader workers 0; each mode skipped once out/timing_<mode>.json exists; done marker .done. Never kills processes.
# Caveat to report with the numbers: the GPU is shared; nvidia-smi load at the end of each mode is stored in the json (upper bound, as M12).
# GPU estimate: ~1.5 min per 330 frames for full (~0.33 s/frame), less for single/det -> ~5 min total plus model loading (~3 min).
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; D=$GP/review1113/cleanfix/B1_timing; DW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B1_timing
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; unset SMOKE
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"
LOG=$D/run.log; say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $D/.done ] && { say "already done"; exit 0; }
n=0; until [ -f /d/cmpfull_runs/gupnet_uda_mt_s0/.done ] && [ -f $GP/review1113/initabl/.done ]; do
  [ $n = 0 ] && say "waiting for gupnet_uda_mt_s0/.done and review1113/initabl/.done (poll 10 min)"; n=1; sleep 600; done
say "start B1"
$PY $DW/make_lib.py >> $LOG 2>&1 || { say "!! make_lib"; exit 2; }
for mode in full single det; do
  try=0; mf=0.3
  until [ -f $D/out/timing_$mode.json ]; do
    MEMFRAC=$mf $SLOT light $PY $DW/timing_clean.py $mode 300 30 > $D/t.try 2>&1; rc=$?; grep -v " INFO " $D/t.try >> $LOG
    [ -f $D/out/timing_$mode.json ] && break
    if grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" $D/t.try && [ $try -lt 2 ]; then
      try=$((try + 1)); MFS=(0.3 0.4 0.5); mf=${MFS[$try]}; say "OOM in $mode -> retry $try with MEMFRAC $mf" | tee -a $D/retry.log
    else say "!! timing $mode failed rc=$rc"; rm -f $D/t.try; exit 2; fi
  done
done; rm -f $D/t.try
SUM=$($PY -c "
import json
f = lambda m: json.load(open('$DW/out/timing_%s.json' % m))
o = []
for m in ('full', 'single', 'det'):
    r = f(m); o.append('%s %.1f / %.1f ms (p90 %.0f), peak %.0f MB, resident %.0f MB, GPU load at end: %s' % (m, r['total_gpu_chain']['mean'], r['total_gpu_chain']['median'], r['total_gpu_chain']['p90'], r['peak_mem_MB_inference'], r['weights_resident_MB'], r['nvidia_smi_used_MB_at_end']))
print(' || '.join(o))")
say "ALL DONE: $SUM"; touch $D/.done
