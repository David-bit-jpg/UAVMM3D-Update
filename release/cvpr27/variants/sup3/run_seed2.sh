#!/bin/bash
# [sup3] NOT RUN by its author. Second run of the clean MAV6D chain (= srcx/run_x.sh in mav, TEACH=full) with the seed of EVERY component that
# run_x.sh trains shifted by SX_SEEDSHIFT (default 10), to give a seed spread for the headline number (PST test UDS 60.81 / val 59.71):
#   refiner stages 1, 2, 3 (nominal seeds 5, 6 -> effective 15, 16; torch.manual_seed + numpy rng 1000+seed inside refiner_x.cmd_train)
#   self-trained detectors x3 (train.py --seed 0,1,2 -> 10,11,12; init = the same fixed sim detector member as before)
#   orientation students S6I (nominal 0, 1 -> effective 10, 11: set_random_seed, sim-crop dataset, loader generators, 80% target bag)
#   -> and therefore everything downstream: pseudo-labels, self-trained detector outputs, crops, teacher AdaBN + pseudo-orientations, student
#      AdaBN/readings, height-net readings, w (sim-val inverse variance), stage-3 targets, deployment constants.
# KEPT FIXED (documented choice): the stage-1 networks of the indoor source = pretrained checkpoints in sx.SOURCES['in'] (sim-only detectors
#   ramp_s0-2, IR+LiDAR / IR-distilled / LiDAR orientation teachers, height network). run_x.sh does not train them; retraining them would take
#   days. Hence their outputs are bit-for-bit inputs of this run and are COPIED from srcx/in2mav instead of recomputed (COPY_FIXED=1, saves ~45 min):
#   recs/det0.pkl + preds/srconly_* (sim-only ensemble on MAV6D), ref/sim_* (refiner sim samples from the sim detector), ref/bank.npy and
#   ref/real_d0_* (from det0), wsim_cache.pkl (sim detector on sim val). COPY_FIXED=0 recomputes them with the same code.
#   Data-construction seeds that are not training seeds also stay (refiner sim-sample selection, banks, BN subsets: rng 0).
# Writes ONLY: review1113/sup3/run/ (recs, ref, ou, preds, logs; ~1.6 GB), D:/S2MUAV3D/srcx_work/in2mav_seed2/ (crops ~9 GB),
#   new files tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_in2mav_seed2.yaml, E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_seed2.pkl
#   (the self-training dataloader reads its index from the cache dir), output/models/uavdet_3d/camnorm/srcx/st_in2mav_seed2/DET1_s{0,1,2}/.
#   Nothing of srcx/in2mav, srcx/in2mav_none or D:/S2MUAV3D/srcx_work/in2mav is written (code = copies in review1113/sup3/lib, make_seed_lib.py).
# Robustness: every training step resumes from its last checkpoint (refiner resume_st<k>_s<j>.pt per epoch / 250 steps; students resume.pth
#   per 250 steps; detectors: train.py's own epoch-checkpoint auto-resume) and is skipped once its output exists; on CUDA OOM the batch is halved
#   and the step retried (max 4 retries; refiner REF_ST1_BS 128 / REF_TRAIN_BS 64, students 8+4, detectors 8, inference DET_BS 8 / ORI_BS 16 /
#   REF_BS 128), each retry logged to logs/retry.log (a halved training batch is a recipe deviation and is flagged). DataLoader workers 0 everywhere.
#   Every inference step runs through gpu_slot.sh light + infer_wrap.py (torch memory fraction 0.2). Never kills processes.
# Time: in2mav took 3 h 29 min from the restart at 04:55 to ALL DONE 08:24 with det0 / sim samples already done (= what COPY_FIXED skips):
#   refiners st1+st2 36 min, st2 infer + pseudo 14, self-trained detectors 15, det1 infer 17, crops + bank 13, teachers AdaBN + read 18,
#   students 25, student AdaBN + readings + height 18, st2 det1 infer + w 18, st3 16, final infer 18, deploy + score 1  ->  ~3.5 h alone on the
#   GPU, ~4-5 h with gpu_slot queueing.
#   launch:  bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run_seed2.sh        (SX_SEEDSHIFT=10 default)
SHIFT=${SX_SEEDSHIFT:-10}; COPY_FIXED=${COPY_FIXED:-1}
export SX_SEEDSHIFT=$SHIFT SX_TEACH=full SX_SRC=in SX_TGT=mav PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
S3=$GPW/review1113/sup3; L=$S3/lib; X=$GP/review1113/sup3/run; XW=$S3/run; WK=/d/S2MUAV3D/srcx_work/in2mav_seed2
F=$GP/srcx/in2mav; FW=/d/S2MUAV3D/srcx_work/in2mav          # reported run: read only
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
IW="$PY $L/infer_wrap.py"
mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs $WK; cd $X
LOG=$X/logs/run_seed2.log; RLOG=$X/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }; die() { say "!! $1"; exit 2; }
isoom() { grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$1"; }
# oomrun <log> "<VAR=start ...>" cmd...   on OOM all VARs halve (min 1), max 4 retries; per-attempt output checked, appended to <log>
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
say "start in2mav seed run 2 (SX_SEEDSHIFT=$SHIFT, COPY_FIXED=$COPY_FIXED)"
$PY $S3/make_seed_lib.py >> $LOG 2>&1 || die "make_seed_lib"
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s SPLIT=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.T['score_split']))
assert sx.O.endswith('review1113/sup3/run') and sx.W.endswith('in2mav_seed2') and sx.PSEUDO_TAG.endswith('_seed2'), (sx.O, sx.W, sx.PSEUDO_TAG)
")" || die "sx config"
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg

# ---------------- fixed stage-1 outputs (indoor pretrained networks, not seeded by run_x.sh): copy or recompute
if [ $COPY_FIXED = 1 ]; then
  [ -f recs/det0.pkl ] || cp $F/recs/det0.pkl recs/
  for sp in val test; do [ -f preds/srconly_$sp.pkl ] || cp $F/preds/srconly_$sp.pkl preds/; done
  for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy; do [ -f ref/$f ] || cp $F/ref/$f ref/; done
  [ -f wsim_cache.pkl ] || cp $F/wsim_cache.pkl .
  say "copied fixed stage-1 outputs from srcx/in2mav (det0, srconly preds, refiner sim samples / bank / real d0 crops, wsim cache)"
else
  [ -f recs/det0.pkl ] || oomrun logs/det0_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $DET0 $DET1 $DET2 --splits train val test --out $XW/recs/det0.pkl || die "det0 infer"
  for sp in val test; do [ -f preds/srconly_$sp.pkl ] || $PY $L/to_pred.py $XW/recs/det0.pkl $sp $XW/preds/srconly_$sp.pkl >> $LOG 2>&1; done
  [ -f ref/sim_meta.pkl ] || oomrun logs/ref_sim.log "DET_BS=8" $SLOT light $IW $L/refiner_x.py sim $DET0 || die "ref sim"
  [ -f ref/bank.npy ] || $PY $L/refiner_x.py bank $XW/recs/det0.pkl >> $LOG 2>&1 || die "ref bank"
  [ -f ref/real_d0_meta.pkl ] || $PY $L/refiner_x.py real $XW/recs/det0.pkl d0 >> $LOG 2>&1 || die "ref real"
fi

# ---------------- Module 4 stages 1-2 (nominal seeds 5, 6 -> 5+SHIFT, 6+SHIFT), resumable, OOM -> halve batch
for j in 5 6; do ( [ -f ref/ref_st1_s$j.pt ] || oomrun logs/ref_st1_s$j.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $j
                  [ -f ref/ref_st1_s$j.pt ] && { [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $j; } ) & sleep 30; done; wait
for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done
say "refiners stage 1-2 done"

# ---------------- Module 3: refiner stage-2 corrections on det0 train -> pseudo-labels (tag *_seed2) -> self-trained detectors x3 (seed k+SHIFT)
for j in 5 6; do [ -f ref/st2_s${j}_det0_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det0.pkl train $XW/ref/st2_s${j}_det0_train.pkl || die "inf st2 det0 $j"; done
[ -f ref/inf_st2_det0_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det0_train.pkl $XW/ref/st2_s5_det0_train.pkl $XW/ref/st2_s6_det0_train.pkl >> $LOG 2>&1
[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
cat pseudo_$TAG.json >> $LOG
STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
DETS=($DET0 $DET1 $DET2)
train_det1() {   # $1 = member k; train.py resumes from its latest epoch checkpoint by itself; OOM -> halve --batch_size (recipe deviation)
  local k=$1 bs=8 try=0 lg=$X/logs/train_det1_s$1.log
  while [ ! -f $(stck $k) ]; do
    ( cd /e/Open3DUAVDet/tools; $SLOT light $PY train.py --cfg_file $ST_CFG --batch_size $bs --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
        --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $((k + SHIFT)) --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $(stck $k) ] && break
    if isoom $lg.try && [ $try -lt 4 ] && [ $bs -gt 1 ]; then try=$((try + 1)); bs=$((bs / 2))
      echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k OOM -> retry $try/4 with --batch_size $bs, resuming from the last epoch checkpoint (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') self-trained detector s$k FAILED (not OOM or retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for k in 0 1 2; do ( train_det1 $k ) & sleep 100; done; wait
for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
[ -f recs/det1.pkl ] || oomrun logs/det1_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl || die "det1 infer"
grep -E "s_bar|frames" logs/det1_infer.log | tail -4 | tee -a $LOG
say "detectors done"

# ---------------- Module 5: orientation crops on the final detector, real background bank, 6 teachers (fixed checkpoints; BN on real train crops <= 80/seq)
for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WK/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
[ -f $WK/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; INIT=($IRL0 $IRL1)
set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); [ -f ou/T_$1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 || die "adabn $1"; shift 2; done
T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
[ -f ou/read_tr1_teach.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 || die "read teachers"
q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
[ -f $WK/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
grep "pseudo pseudo_tr1" $LOG | tail -1
# students S6I nominal seeds 0, 1 (effective 0+SHIFT, 1+SHIFT; init = same-index IR+LiDAR member; gated pseudo-rotations + mirror loss 12, 2250 steps)
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
# height network (fixed checkpoint, kps seed 0): BN on train crops (<= 80/seq), readings
[ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
say "orientation branch done"

# ---------------- Module 4 stage 3 with fused pseudo-targets; w by inverse variance on SIM validation
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1
[ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
WW=$($PY -c "import json;print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW   (reported run: 0.1903)"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
for j in 5 6; do ( [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1
done

# ---------------- Module 6: constants + single-frame deployment; scoring (val = sanity only; test = once)
[ -f deploy_pcmd.json ] || $PY $L/chain_x.py deploy $WW pcmd >> $LOG 2>&1 || die deploy
for p in srconly_val pcmd_fused_val pcmd_nofuse_val; do [ -f preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/$p.pkl val >> $LOG 2>&1; done
for p in srconly_test pcmd_fused_test; do [ -f preds/$p.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/$p.pkl $SPLIT >> $LOG 2>&1 || die "score $p"; done
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
t = f('$XW/preds/pcmd_fused_test.pkl.score_$SPLIT.json'); v = f('$XW/preds/pcmd_fused_val.pkl.score_val.json')
print('seed run 2 (shift $SHIFT) test UDS %.2f AP %.2f ATE %.3f AOE %.2f ASE %.3f mDR %.2f | val UDS %.2f | w %.4f | run 1: test 60.81 / val 59.71' % (
    t['UDS'], t['AP'], t['ATE'], t['AOE'], t['ASE'], t['mDR'], v['UDS'], $WW))")
say "ALL DONE: $SUM"
