#!/bin/bash
# [review1113 lidraw] Supplement control: why a LiDAR "body map" auxiliary target rather than raw LiDAR points? (PST, CVPR'27; low priority)
# All orientation teachers see RGB only; LiDAR is only an auxiliary TRAINING target in simulation. 3 arms x 3 seeds (same-numbered seeds paired):
#   (a) body map  = KZ-LiD (CenterDetLiD, cfg sim_full_r34_kp_w12_zoom_lid): s0/s1 = the chain's teachers lidar/irlid/release/kzl_s{0,1}.pth
#                   (= KZLs{0,1} epoch 30 with lid_head.* stripped; verified tensor-identical). s2 never existed -> trained here, identical recipe.
#   (b) raw points = same recipe, body-map head replaced by a 64-point set head + Chamfer to the tagged LiDAR returns (center_det_lidpts.py,
#                   cfgs/sim_full_r34_kp_w12_zoom_lidpts.yaml, weight 1.0 as the body map). Seeds 0, 1, 2.
#   (c) no LiDAR head = plain KZ (CenterDet, cfg sim_full_r34_kp_w12_zoom), existing KZs{0,1,2} epoch 30, reused.
# KZ-LiD recipe (geo1004/lidar/kzlid/run_kzlid_train.sh + KZLs0.stdout): tools/train.py via train_kz_lid.py, --batch_size 8 --workers 2
#   --fix_random_seed --seed k --cudnn_benchmark --use_amp --epochs 30 --max_ckpt_save_num 3 --logger_iter_interval 500 --val_split test
#   --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval (sim-test monitor only; the epoch-30 weights are used).
# Deviation (default): --workers $WORKERS = 0 (Windows spawn-loader deadlock rule) instead of 2 -> same data / augmentation distribution, different
#   random augmentation stream; ~2-3x slower. WORKERS=2 restores the exact recipe and then uses gpu_slot "heavy" (<2 spawn-loader trainings).
# Metric (teacher level, MAV6D TEST only; the former validation sequences are never read): each net -> the chain's teacher BN treatment
#   (ori_x adabn on the reported in2mav run's unlabeled real train crops oc_tr1, conf >= 0.5, <= 80 per sequence, seed 0; reproduces
#   srcx/in2mav/ou/T_lid0.pth bit-exactly) -> ori_x read on oc_te1 (4,800 test crops) -> single-member ori_x.ensemble (PnP) -> ori_eval.py:
#   mean symmetric AOE + median folded yaw, 3-seed mean +- sd, paired (b-a), (c-a) with a sequence bootstrap over the 10 test sequences.
#   Test labels are read ONLY by "ori_eval.py score", ONCE, after all 9 nets' predictions exist. No choice uses any real label.
# Writes only under review1113/lidraw/ (runs/, release/, eval/, logs/). Never modifies shared code / checkpoints / run dirs. Never kills processes.
# Robustness: waits for review1113/initabl/.done and every review1113/cleanfix/<task>/.done (task = subdir with a *.sh) or 12 h after start
#   (poll 10 min); one training at a time (PAR=1); every GPU step via gpu_slot.sh; training resumes from the newest checkpoint_epoch_*.pth
#   (tools/train.py built-in); CUDA OOM -> batch halved and resumed (logged as RECIPE DEVIATION in logs/retry.log); inference via infer_wrap.py
#   (memory fraction 0.2), OOM -> ORI_BS halved; every step skipped once its output exists; done marker review1113/lidraw/.done.
# GPU time: KZ-LiD-type training 30 epochs x 2,263 it; measured 7.6 min/epoch with workers 2 on an idle GPU (~4 h); workers 0 ~2-3x slower
#   (~8-12 h) - all much slower under CPU contention. 4 trainings (a s2, b s0-2) + 9 x (adabn + read) ~ 3 min each.
#   launch:  bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw/lidraw_commands.sh
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet SX_SRC=in SX_TGT=mav SX_TEACH=full SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LIDRAW_EXP=camnorm/geo1004/gproto/review1113/lidraw/runs
unset SX_SEEDSHIFT
WORKERS=${WORKERS:-0}; PAR=${PAR:-1}; SEEDS=${SEEDS:-"0 1 2"}
KIND=light; [ "$WORKERS" -gt 0 ] && KIND=heavy
PY=D:/Miniconda3/envs/city/python.exe
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
D=$GP/review1113/lidraw; DW=$GPW/review1113/lidraw; L=$DW/eval/lib
M=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm; MW=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
REL=E:/Open3DUAVDet/output/camnorm/geo1004/lidar/irlid/release
SLOT="bash $GP/gpu_slot.sh"
mkdir -p $D/logs $D/runs $D/release $D/eval
LOG=$D/logs/run.log; RLOG=$D/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }; die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
oomrun() {   # oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries
  local lg=$1; local -a vals=($2); shift 2; local try=0 kv k v
  while true; do
    if env "${vals[@]}" "$@" > $lg.try 2>&1; then cat $lg.try >> $lg; rm -f $lg.try; return 0; fi
    cat $lg.try >> $lg
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); local -a nv=()
      for kv in "${vals[@]}"; do k=${kv%%=*}; v=${kv#*=}; v=$((v / 2)); [ $v -lt 1 ] && v=1; nv+=("$k=$v"); done; vals=("${nv[@]}")
      echo "$(date '+%m-%d %H:%M:%S') OOM retry $try/4 with ${vals[*]} (RECIPE DEVIATION) :: $*" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') FAILED (not OOM or 4 retries used): $*  see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done
}

if [ -f $D/.done ]; then say "already done (remove $DW/.done to recompute)"; cat $DW/eval/RESULT.md 2>/dev/null; exit 0; fi
# ---------------- start gate: initabl done; cleanfix tasks done or 12 h
T0=$(date +%s); n=0
until [ -f $GP/review1113/initabl/.done ]; do [ $n = 0 ] && say "waiting for review1113/initabl/.done (poll 10 min)"; n=1; sleep 600; done
cf_pending() { local d p=""; for d in $GP/review1113/cleanfix/*/; do ls $d*.sh > /dev/null 2>&1 || continue; [ -f $d.done ] || p="$p $(basename $d)"; done; echo $p; }
n=0; while [ -n "$(cf_pending)" ] && [ $(( $(date +%s) - T0 )) -lt 43200 ]; do
  [ $n = 0 ] && say "waiting for cleanfix .done markers:$(cf_pending) (or 12 h after start; poll 10 min)"; n=1; sleep 600; done
say "start lidraw (WORKERS=$WORKERS slot=$KIND PAR=$PAR seeds: $SEEDS); cleanfix pending at start:$(cf_pending)"
$PY $DW/make_eval_lib.py >> $LOG 2>&1 || die "make_eval_lib"

# ---------------- trainings
train_one() {   # train_one <cfg (relative to tools/ or absolute)> <cfg stem> <extra_tag> <seed>
  local cfg=$1 stem=$2 tag=$3 s=$4 bs=8 try=0 nf=0
  local ck=$D/runs/$stem/$tag/ckpt lg=$D/logs/train_$tag.log
  while [ ! -f $ck/checkpoint_epoch_30.pth ]; do
    say "train $tag (cfg $stem, seed $s, batch $bs, workers $WORKERS; resumes from the newest epoch checkpoint if any)"
    ( cd /e/Open3DUAVDet/tools && $SLOT $KIND $PY $DW/lidraw_train.py --cfg_file $cfg --batch_size $bs --workers $WORKERS --fix_random_seed \
        --cudnn_benchmark --use_amp --epochs 30 --max_ckpt_save_num 3 --logger_iter_interval 500 --val_split test --val_interval 10 \
        --val_every 100 --val_match greedy --skip_test_eval --seed $s --extra_tag $tag ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $ck/checkpoint_epoch_30.pth ] && break
    if isoom $lg.try && [ $try -lt 3 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') $tag OOM -> retry $try/3 with --batch_size $bs, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    elif [ $nf -lt 1 ]; then nf=1; echo "$(date '+%m-%d %H:%M:%S') $tag failed (not OOM) -> one resume retry, see $lg" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') $tag FAILED, see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try; say "$tag epoch 30 done"
}
JOBS=()
for s in $SEEDS; do
  JOBS+=("$DW/cfgs/sim_full_r34_kp_w12_zoom_lidpts.yaml sim_full_r34_kp_w12_zoom_lidpts KZPs$s $s")                        # arm (b)
  [ -f $M/sim_full_r34_kp_w12_zoom_lid/KZLs$s/ckpt/checkpoint_epoch_30.pth ] || \
    JOBS+=("cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_lid.yaml sim_full_r34_kp_w12_zoom_lid KZLs$s $s")      # arm (a), missing seeds only
done
i=0; for j in "${JOBS[@]}"; do train_one $j & i=$((i + 1)); [ $((i % PAR)) = 0 ] && wait; [ $PAR -gt 1 ] && sleep 120; done; wait
# ---------------- checkpoints per arm (release format, aux heads stripped)
declare -A CK
for s in $SEEDS; do
  if [ -f $M/sim_full_r34_kp_w12_zoom_lid/KZLs$s/ckpt/checkpoint_epoch_30.pth ]; then
    CK[a_s$s]=$REL/kzl_s$s.pth; [ -f $REL/kzl_s$s.pth ] || CK[a_s$s]=$DW/release/a_s$s.pth
    [ -f ${CK[a_s$s]} ] || $PY $DW/strip_aux.py $MW/sim_full_r34_kp_w12_zoom_lid/KZLs$s/ckpt/checkpoint_epoch_30.pth ${CK[a_s$s]} >> $LOG 2>&1 || die "strip a_s$s"
  else
    [ -f $D/runs/sim_full_r34_kp_w12_zoom_lid/KZLs$s/ckpt/checkpoint_epoch_30.pth ] || die "arm a seed $s training missing (re-launch resumes)"
    CK[a_s$s]=$DW/release/a_s$s.pth
    [ -f ${CK[a_s$s]} ] || $PY $DW/strip_aux.py $DW/runs/sim_full_r34_kp_w12_zoom_lid/KZLs$s/ckpt/checkpoint_epoch_30.pth ${CK[a_s$s]} >> $LOG 2>&1 || die "strip a_s$s"
  fi
  [ -f $D/runs/sim_full_r34_kp_w12_zoom_lidpts/KZPs$s/ckpt/checkpoint_epoch_30.pth ] || die "arm b seed $s training missing (re-launch resumes)"
  CK[b_s$s]=$DW/release/b_s$s.pth
  [ -f ${CK[b_s$s]} ] || $PY $DW/strip_aux.py $DW/runs/sim_full_r34_kp_w12_zoom_lidpts/KZPs$s/ckpt/checkpoint_epoch_30.pth ${CK[b_s$s]} >> $LOG 2>&1 || die "strip b_s$s"
  CK[c_s$s]=$MW/sim_full_r34_kp_w12_zoom/KZs$s/ckpt/checkpoint_epoch_30.pth
  [ -f ${CK[c_s$s]} ] || die "arm c KZs$s missing"
done
# ---------------- teacher BN treatment (as run_x.sh: adabn tr1 cap 80) + test-crop readings + single-net predictions (no labels)
for s in $SEEDS; do for a in a b c; do nm=${a}_s$s; ck=${CK[$nm]}
  [ -f $D/eval/T_$nm.pth ] || oomrun $D/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $ck tr1 $DW/eval/T_$nm.pth 80 || die "adabn $nm"
  [ -f $D/eval/read_te1_$nm.pkl ] || oomrun $D/logs/ori_infer.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read te1 $DW/eval/read_te1_$nm.pkl $DW/eval/T_$nm.pth || die "read $nm"
  [ -f $D/eval/pred_$nm.npz ] || $PY $DW/ori_eval.py predict $DW/eval/read_te1_$nm.pkl $DW/eval/T_$nm.pth $DW/eval/pred_$nm.npz >> $LOG 2>&1 || die "predict $nm"
  say "$nm read (ckpt $ck)"
done; done
# ---------------- TEST scored once
$PY -c "
import json; S='$SEEDS'.split(); D='$DW/eval'
json.dump({a: {s: '%s/pred_%s_s%s.npz' % (D, a, s) for s in S} for a in ('a', 'b', 'c')}, open(D + '/arms.json', 'w'), indent=1)" || die arms
[ -f $D/eval/RESULT.json ] || $PY $DW/ori_eval.py score $DW/eval/arms.json $DW/eval/RESULT >> $LOG 2>&1 || die "score"
say "ALL DONE (a = body map, b = raw points, c = no LiDAR head; MAV6D test, teacher level): $(tr '\n' ' ' < $D/eval/RESULT.md | sed 's/  */ /g')"
touch $D/.done
