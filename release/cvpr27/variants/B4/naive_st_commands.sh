#!/bin/bash
# [review1113 B4] NOT RUN by its author. Naive self-training baseline on our detector (MAV6D, clean chain srcx/in2mav).
# = Step 1 (Module 3) of srcx/run_x.sh, except that the pseudo-labels are the sim-only detector ensemble's OWN confident boxes:
#   frames with conf >= 0.5 (same frames as PST's Module-3 labels), box = (t_det, s_bar, R_det), i.e. its own depth; NO IR-refiner correction
#   (make_pseudo_naive.py = make_pseudo_x.py with the delta / c_bar term removed). Self-training recipe unchanged: 3 detectors, seeds 0/1/2,
#   init = the 3 sim-only members, st_*.yaml recipe (3 epochs, LR 2e-4, rot weight 0, hflip only), batch 8, workers 0, AMP.
# Evaluation of the self-trained detector alone = build-up row (b) (recs/det1.pkl as is: own rotation, size s_bar). PST's row (b) = 53.38 test
#   / 51.29 val (review1113/clean2/CLEAN2_REPORT.md section 1).
# REST=1 (default): also runs the rest of the PST chain on top of the naive detector (orientation crops, teachers, pseudo-orientations, students
#   S6I, AdaBN, readings, height net, stage-2 refiner on det1, w on sim val, stage-3 depth students, deployment). Steps 2-3 are exactly PST's:
#   stage-3 targets use PST's IR-corrected Module-3 labels (index_pseudo_srcx_in2mav, read only) so that only step 1 differs. REST=0 stops after
#   row (b).
# KEPT FIXED / COPIED from srcx/in2mav (read only; same as the seed runs' COPY_FIXED, plus the stage-1/2 refiners because there is no seed shift,
#   i.e. retraining them would repeat the identical recipe): recs/det0.pkl, preds/srconly_*, ref/{sim_*,bank,real_d0_*}, wsim_cache.pkl,
#   ref/ref_st{1,2}_s{5,6}.pt (the latter only needed for REST=1).
# Writes ONLY: review1113/B4/run/ (+ lib), D:/S2MUAV3D/srcx_work/in2mav_naive/ (crops, ~9 GB, REST=1), new files
#   tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_in2mav_naive.yaml, E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_naive.pkl,
#   output/models/uavdet_3d/camnorm/srcx/st_in2mav_naive/DET1_s{0,1,2}/. Nothing of srcx/in2mav or other runs is written.
# Robustness (as sup3/seeds): every training step resumes (detectors: train.py epoch auto-resume; students: resume.pth every 250 steps;
#   refiner st3: every 250 steps) and is skipped once its output exists; on CUDA OOM the batch is halved and the step retried (max 4), each
#   retry in logs/retry.log (halved training batch = RECIPE DEVIATION). Workers 0 everywhere. Inference via gpu_slot.sh light + infer_wrap.py
#   (memory fraction 0.2). Never kills processes. Max 2 concurrent trainings (detectors: 2 then 1, students 2).
# Time (from srcx/in2mav run_x.log stage timings): naive pseudo < 1 min, 3 detectors ~25 min (2 + 1 instead of 3 concurrent), det1 infer ~17 min,
#   score 1 min -> row (b) after ~45 min. REST: crops + bank 13, teachers AdaBN + read 18, students 25, student AdaBN + readings + height 18,
#   st2 det1 infer + w 18, st3 16, final infer 18, deploy + score 1 -> +~2 h 10 min. Total ~3 h alone on the GPU, ~3.5-4 h with queueing.
#   launch:  bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/naive_st_commands.sh          (REST=0 for row (b) only)
REST=${REST:-1}
export SX_TEACH=full SX_SRC=in SX_TGT=mav PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
unset SX_SEEDSHIFT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
B4=$GPW/review1113/B4; L=$B4/lib; X=$GP/review1113/B4/run; XW=$B4/run; WK=/d/S2MUAV3D/srcx_work/in2mav_naive
F=$GP/srcx/in2mav                                            # reported PST run: read only
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
IW="$PY $L/infer_wrap.py"
mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs $WK; cd $X
LOG=$X/logs/run.log; RLOG=$X/logs/retry.log
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
fmt() { $PY -c "
import json, sys; d = json.load(open(sys.argv[1]))['all']
print('UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f' % (d['UDS'], d['AP'], d['ATE'], d['ASE'], d['AOE'], d['mDR']))" "$1"; }
say "start B4 naive self-training (REST=$REST)"
$PY $B4/make_naive_lib.py >> $LOG 2>&1 || die "make_naive_lib"
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s SPLIT=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.T['score_split']))
assert sx.O.endswith('review1113/B4/run') and sx.W.endswith('in2mav_naive') and sx.PSEUDO_TAG == 'srcx_in2mav_naive' and sx.PSEUDO_TAG_ST3 == 'srcx_in2mav'
")" || die "sx config"
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg

# ---------------- fixed inputs copied from srcx/in2mav
[ -f recs/det0.pkl ] || cp $F/recs/det0.pkl recs/ || die "copy det0"
for sp in val test; do [ -f preds/srconly_$sp.pkl ] || cp $F/preds/srconly_$sp.pkl preds/ || die "copy srconly"; done
for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st1_s5.pt ref_st1_s6.pt ref_st2_s5.pt ref_st2_s6.pt; do
  [ -f ref/$f ] || cp $F/ref/$f ref/ || die "copy ref/$f"; done
[ -f wsim_cache.pkl ] || cp $F/wsim_cache.pkl . || die "copy wsim_cache"

# ---------------- naive Module 3: own boxes (conf >= 0.5, own depth) -> self-trained detectors x3 (seeds 0, 1, 2)
[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_naive.py $XW/recs/det0.pkl none $TAG >> $LOG 2>&1 || die "naive pseudo"
cat pseudo_$TAG.json >> $LOG; echo >> $LOG
STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
DETS=($DET0 $DET1 $DET2)
train_det1() {   # $1 = member k; train.py resumes from its latest epoch checkpoint by itself; OOM -> halve --batch_size (recipe deviation)
  local k=$1 bs=8 try=0 lg=$X/logs/train_det1_s$1.log
  while [ ! -f $(stck $k) ]; do
    ( cd /e/Open3DUAVDet/tools; $SLOT light $PY train.py --cfg_file $ST_CFG --batch_size $bs --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
        --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $k --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $(stck $k) ] && break
    if isoom $lg.try && [ $try -lt 4 ] && [ $bs -gt 1 ]; then try=$((try + 1)); bs=$((bs / 2))
      echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k OOM -> retry $try/4 with --batch_size $bs, resuming from the last epoch checkpoint (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k FAILED (not OOM or retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
( train_det1 0 ) & sleep 100; ( train_det1 1 ) & wait      # at most 2 concurrent trainings
train_det1 2
for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
[ -f recs/det1.pkl ] || oomrun logs/det1_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl || die "det1 infer"
grep -E "s_bar|frames" logs/det1_infer.log | tail -4 | tee -a $LOG
for sp in val test; do [ -f preds/naive_det1_$sp.pkl ] || $PY $L/to_pred.py $XW/recs/det1.pkl $sp $XW/preds/naive_det1_$sp.pkl >> $LOG 2>&1 || die "to_pred $sp"; done
[ -f preds/naive_det1_val.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/naive_det1_val.pkl val >> $LOG 2>&1 || die "score row b val"
[ -f preds/naive_det1_test.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/naive_det1_test.pkl $SPLIT >> $LOG 2>&1 || die "score row b test"
RB="row (b) naive self-trained detector: test $(fmt $XW/preds/naive_det1_test.pkl.score_$SPLIT.json) | val $(fmt $XW/preds/naive_det1_val.pkl.score_val.json) || PST row (b): test 53.38 / val 51.29"
say "ROW_B DONE: $RB"
if [ $REST = 0 ]; then say "ALL DONE: $RB"; exit 0; fi

# ---------------- REST: PST steps 2-3 on top of the naive detector (= run_x.sh / seeds run_seed_chain.sh with shift 0)
for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WK/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
[ -f $WK/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; INIT=($IRL0 $IRL1)
set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); [ -f ou/T_$1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 || die "adabn $1"; shift 2; done
T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
[ -f ou/read_tr1_teach.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 || die "read teachers"
q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
[ -f $WK/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
train_student() {   # $1 = seed; resume from runs/S6I_s<seed>/resume.pth; OOM -> halve --bs-src/--bs-tgt (recipe deviation)
  local s=$1 bs=8 bt=4 try=0 lg=$X/logs/ou_train_s$1.log
  while [ ! -f ou/runs/S6I_s$s/ckpt_last.pth ]; do
    OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $XW/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f ou/runs/S6I_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for s in 0 1; do ( train_student $s ) & sleep 100; done; wait
for s in 0 1; do [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || die "student s$s"
  [ -f ou/S_ab$s.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 || die "student adabn $s"; done
for n in tr1 val1 te1; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth || die "read $n"; done
[ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
say "orientation branch done"
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1
[ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
WW=$($PY -c "import json;print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW   (PST: 0.1903)"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
for j in 5 6; do ( [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1
done
[ -f deploy_pcmd.json ] || $PY $L/chain_x.py deploy $WW pcmd >> $LOG 2>&1 || die deploy
for p in pcmd_fused_val pcmd_nofuse_val; do [ -f preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/$p.pkl val >> $LOG 2>&1; done
[ -f preds/pcmd_fused_test.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/pcmd_fused_test.pkl $SPLIT >> $LOG 2>&1 || die "score full chain"
say "ALL DONE: test row(b) $(fmt $XW/preds/naive_det1_test.pkl.score_$SPLIT.json) / full chain $(fmt $XW/preds/pcmd_fused_test.pkl.score_$SPLIT.json) | val row(b) $(fmt $XW/preds/naive_det1_val.pkl.score_val.json) / full chain $(fmt $XW/preds/pcmd_fused_val.pkl.score_val.json) || PST: row(b) 53.38/51.29, full 60.81/59.71 (test/val), naive w $WW"
