# [review1113 stage1seeds] shared shell helpers (sourced by prep_shared.sh / run_group.sh). Same conventions as seeds/run_seed_chain.sh and
# sensorfree/run_sf.sh: oomrun halves batch env vars on CUDA OOM (max 4; a halved TRAINING batch = RECIPE DEVIATION), never kills anything.
PY=D:/Miniconda3/envs/city/python.exe
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
S1=$GP/review1113/stage1seeds; S1W=$GPW/review1113/stage1seeds
SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset OU_INIT OU_BANK OU_CROPS REF_TAG REF_REAL REF_DP
GLOG=$S1/logs/stage1seeds.log
wp() { echo "$1" | sed -E 's|^/([a-z])/|\U\1:/|'; }                 # /e/x -> E:/x
pp() { echo "$1" | sed -E 's|^([A-Za-z]):/|/\L\1/|'; }              # E:/x -> /e/x
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
# say/die/dev need LOG, RLOG, TAGL, MARK (set by the caller)
say() { echo "$(date '+%m-%d %H:%M:%S') [$TAGL] $1" | tee -a $LOG >> $GLOG; }
die() { say "!! $1"; touch $MARK.failed; exit 2; }
dev() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $RLOG >> $LOG; echo "$(date '+%m-%d %H:%M:%S') [$TAGL] $1" >> $GLOG; }
oomrun() {   # oomrun <log> "<VAR=start ...>" cmd...
  local lg=$1; local -a vals=($2); shift 2; local try=0 kv k v
  while true; do
    if env "${vals[@]}" "$@" > $lg.try 2>&1; then cat $lg.try >> $lg; rm -f $lg.try; return 0; fi
    cat $lg.try >> $lg
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); local -a nv=()
      for kv in "${vals[@]}"; do k=${kv%%=*}; v=${kv#*=}; v=$((v / 2)); [ $v -lt 1 ] && v=1; nv+=("$k=$v"); done; vals=("${nv[@]}")
      dev "OOM retry $try/4 with ${vals[*]} :: $* (a halved TRAINING batch = RECIPE DEVIATION)"
    else dev "FAILED (not OOM or 4 retries used): $*  see $lg"; rm -f $lg.try; return 1; fi
  done
}
waitf() { local n=0; until [ -f "$1" ]; do [ $n = 0 ] && say "waiting for $1"; n=1; sleep 120; done; }
# waitnet <src> <sd> <item>...   wait for the chain checkpoints of re-seeded stage-1 items; die if train_stage1.sh marked one failed/skipped
waitnet() { local src=$1 sd=$2 it nm n; shift 2
  for it in "$@"; do nm=${it/:/}; eval "$($PY $S1W/s1paths.py sh $src $sd $it | grep '^REL=')"; n=0
    until [ -f "$(pp $REL)" ]; do
      [ -f $S1/stage1/${src}_sd$sd/$nm.failed ] && die "stage-1 $src sd$sd $it failed (see logs/stage1_${src}_sd$sd.log)"
      [ -f $S1/stage1/${src}_sd$sd/$nm.skipped ] && die "stage-1 $src sd$sd $it skipped (latest start)"
      [ $n = 0 ] && say "waiting for stage-1 $src sd$sd $it -> $REL"; n=1; sleep 300; done
  done; }
# lnk <src> <dst>: hard link, else copy (both bash paths)
lnk() { [ -f "$2" ] && return 0; $PY -c "import os, sys; os.link(sys.argv[1], sys.argv[2])" "$(wp $1)" "$(wp $2)" 2>/dev/null || cp "$1" "$2"; }
cpf() { [ -f "$2" ] || cp "$1" "$2" || die "copy $1"; }
latest_start() {   # latest_start <default MMDDHHMM> <dir that exists once started>: skip a run that has not started yet after LATEST_START
  local LS=${LATEST_START:-$1} NOW=$(date '+%m%d%H%M')
  if [ $((10#$NOW)) -gt $((10#$LS)) ] && [ ! -d "$2" ]; then say "SKIPPED: machine time $NOW past latest start $LS"; touch $MARK.skipped; exit 0; fi; }
