#!/bin/bash
# [review1113/cleanfix] shared plumbing for B3 / B4 / B6 (sourced, not run). Same robustness pattern as review1113/initabl + clean3/nomir:
#   - waits at start until the paper-critical GUPNet-UDA run AND the initabl ablation are done (poll every 10 min)
#   - every GPU step through gpu_slot.sh light; DataLoader workers 0 (ou_train_x / ori_x already use num_workers=0)
#   - students resume from <run>/resume.pth (every 250 steps); CUDA OOM -> --bs-src/--bs-tgt halved (max 4 retries, RECIPE DEVIATION in logs/oom.log)
#   - inference via infer_wrap.py (memory fraction 0.2); OOM -> ORI_BS halved down to 1
#   - every step skipped once its output exists; arm done marker <arm>/.done
#   - never kills processes. Real labels: read only by score_x.py (test, once per arm) [B6 additionally reads TRAIN GT to build the oracle labels].
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
unset SX_SEEDSHIFT OU_INIT
PY=D:/Miniconda3/envs/city/python.exe
GPB=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
SLOT="bash $GPB/gpu_slot.sh"
CF=$GP/review1113/cleanfix
X=$GP/srcx/in2mav; XW=D:/S2MUAV3D/srcx_work/in2mav            # deployed clean MAV6D run (READ ONLY)
W_FROZEN=0.190282282581008                                       # = srcx/in2mav/w_sim.json (deployed omega)
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }

wait_prereq() {   # $1 = log file
  local n=0
  until [ -f /d/cmpfull_runs/gupnet_uda_mt_s0/.done ] && [ -f $GPB/review1113/initabl/.done ]; do
    [ $n = 0 ] && echo "$(date '+%m-%d %H:%M:%S') waiting for D:/cmpfull_runs/gupnet_uda_mt_s0/.done and review1113/initabl/.done (poll every 10 min)" | tee -a $1
    n=1; sleep 600; done
}

# setup_arm <arm dir>: lib copy (O/W -> arm), read-only inputs of the deployed run (FULL=0: refiners, c_k, r_k, s_bar, height readings frozen)
setup_arm() {
  local A=$1
  mkdir -p $A/recs $A/ref $A/ou/runs $A/preds $A/logs $A/work
  $PY $CF/make_cf_lib.py $A/lib >> $A/logs/arm.log 2>&1 || return 1
  local f
  for f in recs/det1.pkl kps_tr1.pkl kps_val1.pkl kps_te1.pkl ref/inf_fin_det1_train.pkl ref/inf_fin_det1_val.pkl ref/inf_fin_det1_test.pkl; do
    [ -f $A/$f ] || cp $X/$f $A/$f || return 1; done
  for f in oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do [ -f $A/work/$f ] || cp $XW/$f $A/work/ || return 1; done
  CL_AO=$A $PY -c "
import sys; sys.path.insert(0, '$A/lib'); import sx
assert sx.O == '$A' and sx.W == '$A/work' and sx.W_ORIG == '$XW' and (sx.SRC, sx.TGT, sx.TEACH) == ('in', 'mav', 'full'), (sx.O, sx.W)
print('[cleanfix] O', sx.O, '| W', sx.W, '| crop arrays fall back to', sx.W_ORIG)" >> $A/logs/arm.log 2>&1
}

# oomrun <arm> <log> <start ORI_BS> cmd...   (inference; ORI_BS halved on OOM down to 1)
oomrun() {
  local A=$1 lg=$2 b=$3; shift 3
  while true; do
    if env ORI_BS=$b "$@" >> $lg 2>&1; then return 0; fi
    if isoom $lg && [ $b -gt 1 ]; then b=$((b / 2)); echo "$(date '+%m-%d %H:%M:%S') OOM in: $* -> retry with ORI_BS=$b" >> $A/logs/oom.log
    else echo "$(date '+%m-%d %H:%M:%S') failed (not OOM or batch at 1): $*  see $lg" >> $A/logs/oom.log; return 1; fi
  done
}

# train_student <arm> <seed> <tag> <extra ou_train args...>   init = same-seed IR+LiDAR member (sx default), recipe of run_x.sh S6I otherwise
train_student() {
  local A=$1 s=$2 tag=$3; shift 3
  local bs=8 bt=4 try=0 lg=$A/logs/ou_train_s$s.log
  while [ ! -f $A/ou/runs/${tag}_s$s/ckpt_last.pth ]; do
    CL_AO=$A OU_CROPS=$XW/oc_tr1.npy OU_BANK=$XW/ou_bank.npy $SLOT light $PY $A/lib/ou_train_x.py --seed $s --tag $tag --tgt gate --mir 12 \
        --runs-dir $A/ou/runs --bs-src $bs --bs-tgt $bt "$@" > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $A/ou/runs/${tag}_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" >> $A/logs/oom.log
    else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED (not OOM or 4 retries used), see $lg" >> $A/logs/oom.log; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}

# students_arm <arm> <tag> <pseudo name in arm/work> <extra args...> : 2 students (concurrent, as run_x.sh) -> AdaBN (all train crops)
#   -> mirror-view readings on val1 (only for b_k, label-free calibration images) and te1 -> deploy at frozen omega -> score fused TEST once
students_arm() {
  local A=$1 tag=$2 pse=$3; shift 3
  [ -f $A/.done ] && { echo "$(date '+%m-%d %H:%M:%S') $A already done"; return 0; }
  local L=$A/lib s n
  for s in 0 1; do ( train_student $A $s $tag --pseudo $pse "$@" ) & sleep 100; done; wait
  for s in 0 1; do [ -f $A/ou/runs/${tag}_s$s/ckpt_last.pth ] || { echo "!! student s$s missing ($A)"; return 2; }; done
  for s in 0 1; do [ -f $A/ou/S_ab$s.pth ] || CL_AO=$A oomrun $A $A/logs/ori_infer.log 16 $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $A/ou/runs/${tag}_s$s/ckpt_last.pth tr1 $A/ou/S_ab$s.pth 0 || return 2; done
  for n in val1 te1; do [ -f $A/ou/read_${n}_stu.pkl ] || CL_AO=$A ORI_MIRROR=1 oomrun $A $A/logs/ori_infer.log 16 $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read $n $A/ou/read_${n}_stu.pkl $A/ou/S_ab0.pth,$A/ou/S_ab1.pth || return 2; done
  deploy_score $A $tag
}

# deploy_score <arm> <prefix>: chain_x deploy (CPU, frozen omega; reads ou/read_{val1,te1}_stu.pkl) -> score <prefix>_fused_test ONCE
deploy_score() {
  local A=$1 p=$2
  [ -f $A/deploy_$p.json ] || CL_AO=$A $PY $A/lib/chain_x.py deploy $W_FROZEN $p > $A/logs/deploy.log 2>&1 || { echo "!! deploy $A"; return 2; }
  [ -f $A/preds/${p}_fused_test.pkl.score_real_test.json ] || CL_AO=$A $PY $A/lib/score_x.py $A/preds/${p}_fused_test.pkl real_test >> $A/logs/score.log 2>&1 || { echo "!! score $A"; return 2; }
  touch $A/.done
  echo "$(date '+%m-%d %H:%M:%S') $A: test $(grep -o '"AOE": [0-9.]*' $A/preds/${p}_fused_test.pkl.score_real_test.json | head -1) $(grep -o '"UDS": [0-9.]*' $A/preds/${p}_fused_test.pkl.score_real_test.json | head -1)  (PST reported: AOE 14.48 UDS 60.81)"
}
