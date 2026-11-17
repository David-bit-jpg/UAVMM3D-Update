#!/bin/bash
# [review1113 B3] NOT RUN by its author. Transfer ROUTE at the same insertion point (central thesis): how teacher knowledge reaches the
# orientation students of the clean MAV6D chain (srcx/in2mav). Same frozen teachers, same real crops, same initialisation; only the route changes.
#   (i)   hard : hard pseudo-orientation labels (L_pl^ori) + mirror loss 12 = the deployed recipe.  NOT retrained: = srcx/in2mav (test 60.81 /
#                val 59.71; students S6I_s0/s1).
#   (ii)  feat : NO hard pseudo labels. Feature alignment + soft KD on the real crops: L_feat (1 - cos between the student's kp2d-head input
#                'features_2d' and the 6 teachers' features_2d, 5x5 cells around the teachers' mean-heatmap peak inside the 5x5 detection-cell window (deployment read rule); FEAT_TARGET=adapt (default):
#                through one 1x1 adapter per teacher, identity init, Adam 1e-3, averaged over teachers (FitNets; needed because the 6
#                independently trained teachers' features are near-orthogonal: pairwise cos 0.09-0.16, only kdir0/kdir1 0.62, so their plain
#                mean is not a coherent target); FEAT_TARGET=mean: literal 1 - cos(f_s, mean_t f_t), outputs under B3/mean/) + w_soft * (soft Bernoulli-KL on the mean
#                teacher heatmap + 12 * symmetric L1 to the teachers' kp2d at the peak). Simulated composite loss (L_src, background swap 0.5) and
#                mirror loss 12 (on the teachers' peak cell) stay.
#   (iii) both : hard pseudo labels + mirror loss (unchanged) + feature alignment w_feat * L_feat.
# Exact definitions: review1113/B3/make_route_lib.py docstring (code in the generated lib/ou_train_x.py, functions route_terms / teacher_forward).
# Teachers = srcx/in2mav/ou/T_{irlid0,irlid1,kdir0,kdir1,lid0,lid1}.pth: the same AdaBN'd (real train crops, <= 80/seq) frozen teacher states that
#   produced pseudo_tr1 (read only). Real crops = the same gated pool (pseudo_tr1 gate, 4,301 crops) and per-seed 80 % bag as the hard recipe; in the
#   feat route the pseudo boxes only pass through the data pipeline (augmentation bookkeeping), no loss reads them.
# Loss weights: fixed ONCE at initialisation by step 1 (calibration, no training): w_feat = mean L_pl^ori / mean L_feat and w_soft = mean L_pl^ori /
#   mean L_soft over 50 real batches (smoke, 20 batches: L_pl 2.19, L_feat 0.757, L_soft 2.19 -> w_feat 2.89, w_soft 1.00) of the seed-0 loader (student = IR+LiDAR member 0, train-mode BN, no grad) -> calib/calib.json, reused for every
#   arm and seed. Nothing is tuned on val or test (val = sanity, test scored once).
# Everything else = run_x.sh / nogate_commands.sh: init = same-seed IR+LiDAR member, 2,250 steps, bs 8 sim + 4 real, LR 2e-4, mem 0.3, seeds 0, 1;
#   then AdaBN (all train crops), 2-student readings with the test-time mirror view, deploy with the frozen omega 0.190, score val + test once.
#   FULL=0 (default, = nomir/nogate): depth refiners, c_k, r_k, s_bar frozen (copied from srcx/in2mav); only rotations and the corner-PnP distance
#     of the test-time fusion change. FULL=1: also re-reads train crops and retrains the step-3 refiners (seeds 5, 6) on targets from the new
#     students (omega frozen).
# Writes ONLY under review1113/B3/ (lib, calib, feat, both). Read only: srcx/in2mav/{recs,ref,ou/T_*,kps_*}, D:/S2MUAV3D/srcx_work/in2mav/
#   {pseudo_tr1.npz, oc_*, ou_bank.npy} (crop arrays read in place via OU_CROPS / OU_BANK / _wf()).
# Robustness: students resume from <run>/resume.pth (every 250 steps) and every step is skipped once its output exists; on CUDA OOM the batch is
#   halved and the step retried (max 4 retries, logs/retry.log; a halved student batch is a RECIPE DEVIATION and is flagged). Workers 0. Inference
#   (AdaBN, readings, refiner) through gpu_slot.sh light + infer_wrap.py (memory fraction 0.2); the calibration pass runs at --mem 0.2.
#   Max 2 concurrent trainings. Never kills processes.
# Time: calibration ~2 min (measured: 20 batches < 1 min after a ~40 s load); students: the hard pair took 24 min (0.63 s/step, 2 concurrent);
#   smoke-measured feat step ~1.0 s alone with the 6 teachers (3.8 GB peak) -> ~45-50 min per concurrent pair, 2 pairs (feat, both) ~1 h 40 min;
#   per arm AdaBN ~3 + readings ~5 min; deploy + score ~2 min.  Total FULL=0 ~2 h alone on the GPU, ~2.5-3 h with queueing
#   (FULL=1: +~45 min per arm).
#   launch:  bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/route_commands.sh
FULL=${FULL:-0}; MIRVIEW=${MIRVIEW:-1}; ARMS=${ARMS:-"feat both"}; FEAT_TARGET=${FEAT_TARGET:-adapt}
case $FEAT_TARGET in adapt|mean) ;; *) echo "FEAT_TARGET must be adapt or mean"; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 SX_INFER_MEMFRAC=0.2
unset SX_SEEDSHIFT
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; B3=$GP/review1113/B3; L=$B3/lib
BO=$B3; [ $FEAT_TARGET = mean ] && BO=$B3/mean                   # outputs: B3/{calib,feat,both,logs} (adapt) or B3/mean/{...} (mean)
X=$GP/srcx/in2mav; XW=D:/S2MUAV3D/srcx_work/in2mav
export OU_CROPS=$XW/oc_tr1.npy OU_BANK=$XW/ou_bank.npy          # read in place (5.7 GB / 2.1 GB, not copied)
export OU_TEACHERS=$X/ou/T_irlid0.pth,$X/ou/T_irlid1.pth,$X/ou/T_kdir0.pth,$X/ou/T_kdir1.pth,$X/ou/T_lid0.pth,$X/ou/T_lid1.pth
W_FROZEN=0.190282282581008                                       # = srcx/in2mav/w_sim.json (deployed omega)
mkdir -p $BO/logs $BO/calib/runs
LOG=$BO/logs/route.log; RLOG=$BO/logs/retry.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
die() { say "!! $1"; exit 2; }
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

# 0) lib copies + per-arm read-only inputs (~20 MB per arm; FULL adds ~1.2 GB per arm)
$PY $B3/make_route_lib.py >> $LOG 2>&1 || die "make_route_lib"
for p in ${OU_TEACHERS//,/ }; do [ -f $p ] || die "missing teacher $p"; done
prep_arm() {   # $1 = arm
  local A=$BO/$1; mkdir -p $A/recs $A/ref $A/ou/runs $A/preds $A/logs $A/work
  [ -f $A/recs/det1.pkl ] || cp $X/recs/det1.pkl $A/recs/
  for f in kps_tr1.pkl kps_val1.pkl kps_te1.pkl; do [ -f $A/$f ] || cp $X/$f $A/; done
  for f in pseudo_tr1.npz oc_tr1.pkl oc_val1.pkl oc_te1.pkl; do [ -f $A/work/$f ] || cp $XW/$f $A/work/; done
  if [ $FULL = 0 ]; then for sp in train val test; do [ -f $A/ref/inf_fin_det1_$sp.pkl ] || cp $X/ref/inf_fin_det1_$sp.pkl $A/ref/; done
  else [ -f $A/recs/det0.pkl ] || cp $X/recs/det0.pkl $A/recs/
    for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st2_s5.pt ref_st2_s6.pt inf_st2_det1_train.pkl; do [ -f $A/ref/$f ] || cp $X/ref/$f $A/ref/; done
  fi
}
for arm in $ARMS; do case $arm in feat|both) prep_arm $arm ;; *) die "bad arm $arm" ;; esac; done
[ -d $BO/feat ] || prep_arm feat           # the calibration reads pseudo_tr1 via CL_AO=$BO/feat
eval "$(CL_AO=$BO/feat $PY -c "
import sys; sys.path.insert(0, '$L'); import sx
print('IRL0=%s IRL1=%s' % tuple(sx.S['irlid']))")" || die "sx"
INIT=($IRL0 $IRL1)
say "start B3 route (ARMS=$ARMS FULL=$FULL MIRVIEW=$MIRVIEW FEAT_TARGET=$FEAT_TARGET -> $BO); teachers $OU_TEACHERS"

# 1) init calibration of the KD weights (once; no training; seed-0 loader, student = IR+LiDAR member 0, 50 real batches)
CAL=$BO/calib/calib.json
if [ ! -f $CAL ]; then
  bt=4; try=0
  while [ ! -f $CAL ]; do
    CL_AO=$BO/feat OU_INIT=$IRL0 $SLOT light $PY $L/ou_train_x.py --seed 0 --tag CALIB --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $BO/calib/runs \
      --bs-src 8 --bs-tgt $bt --mem 0.2 --calib-out $CAL --calib-n 50 --feat-target $FEAT_TARGET > $BO/calib/calib.log.try 2>&1
    cat $BO/calib/calib.log.try >> $BO/calib/calib.log
    [ -f $CAL ] && break
    if isoom $BO/calib/calib.log.try && [ $try -lt 4 ] && [ $bt -gt 1 ]; then try=$((try + 1)); bt=$((bt / 2))
      echo "$(date '+%m-%d %H:%M:%S') calibration OOM -> retry with --bs-tgt $bt" | tee -a $RLOG >> $LOG
    else rm -f $BO/calib/calib.log.try; die "calibration failed, see $BO/calib/calib.log"; fi
  done; rm -f $BO/calib/calib.log.try
fi
eval "$($PY -c "import json; J = json.load(open('$CAL')); print('WF=%r WS=%r' % (J['w_feat'], J['w_soft']))")" || die "read calib"
say "calibration (fixed for all arms/seeds): $($PY -c "import json; J = json.load(open('$CAL')); print('L_pl %.4f L_feat %.4f L_soft %.4f (hm %.4f kp %.4f) L_mir %.4f -> w_feat %.4g w_soft %.4g; teacher feature cos min %.3f' % (J['L_pl'], J['L_feat'], J['L_soft'], J['L_hm'], J['L_kp'], J['L_mir'], J['w_feat'], J['w_soft'], min(min(r) for r in J['teacher_feature_cos'])))")"

# 2) students per arm, seeds 0 and 1 concurrently (2 trainings at a time). Resume from <run>/resume.pth; OOM -> halve batch.
train_student() {   # $1 = arm, $2 = seed
  local arm=$1 s=$2 bs=8 bt=4 try=0 A=$BO/$1 TAG=ROUTE_${1^^} lg=$BO/$1/logs/ou_train_s$2.log
  local -a extra=(--w-feat $WF); [ $arm = feat ] && extra+=(--w-soft $WS)
  while [ ! -f $A/ou/runs/${TAG}_s$s/ckpt_last.pth ]; do
    CL_AO=$A OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag $TAG --tgt gate --mir 12 --pseudo pseudo_tr1 --route $arm --feat-target $FEAT_TARGET "${extra[@]}" \
      --runs-dir $A/ou/runs --bs-src $bs --bs-tgt $bt > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $A/ou/runs/${TAG}_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      echo "$(date '+%m-%d %H:%M:%S') $arm student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)" | tee -a $RLOG >> $LOG
    else echo "$(date '+%m-%d %H:%M:%S') $arm student s$s FAILED (not OOM or 4 retries used), see $lg" | tee -a $RLOG >> $LOG; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for arm in $ARMS; do
  for s in 0 1; do ( train_student $arm $s ) & sleep 100; done; wait
  for s in 0 1; do [ -f $BO/$arm/ou/runs/ROUTE_${arm^^}_s$s/ckpt_last.pth ] || die "$arm student s$s missing"; done
  say "$arm students done ($(grep -ho '"max_mem_gb": [0-9.]*' $BO/$arm/ou/runs/ROUTE_${arm^^}_s*/DONE.json | tr '\n' ' '))"
done

# 3) per arm: AdaBN (all train crops), readings (+ test-time mirror view), FULL step-3, deployment (frozen omega), scoring (val sanity, test once)
for arm in $ARMS; do
  A=$BO/$arm; TAG=ROUTE_${arm^^}
  for s in 0 1; do [ -f $A/ou/S_ab$s.pth ] || CL_AO=$A oomrun $A/logs/adabn_s$s.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py adabn $A/ou/runs/${TAG}_s$s/ckpt_last.pth tr1 $A/ou/S_ab$s.pth 0 || die "$arm adabn s$s"; done
  SETS="val1 te1"; [ $FULL = 1 ] && SETS="tr1 val1 te1"
  for n in $SETS; do [ -f $A/ou/read_${n}_stu.pkl ] || CL_AO=$A ORI_MIRROR=$MIRVIEW oomrun $A/logs/read_$n.log "ORI_BS=16" $SLOT light $PY $L/infer_wrap.py $L/ori_x.py read $n $A/ou/read_${n}_stu.pkl $A/ou/S_ab0.pth,$A/ou/S_ab1.pth || die "$arm read $n"; done
  if [ $FULL = 1 ]; then
    [ -f $A/ref/dp_fused.npy ] || CL_AO=$A $PY $L/chain_x.py targets $W_FROZEN > $A/logs/targets.log 2>&1 || die "$arm targets"
    for j in 5 6; do ( [ -f $A/ref/ref_st3_s$j.pt ] || CL_AO=$A REF_REAL=d0 REF_DP=$A/ref/dp_fused.npy oomrun $A/logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
    for j in 5 6; do [ -f $A/ref/ref_st3_s$j.pt ] || die "$arm ref st3 s$j missing"; done
    for sp in train val test; do
      for j in 5 6; do [ -f $A/ref/fin_s${j}_$sp.pkl ] || CL_AO=$A oomrun $A/logs/infer.log "REF_BS=128" $SLOT light $PY $L/infer_wrap.py $L/refiner_x.py infer $A/ref/ref_st3_s$j.pt $A/recs/det1.pkl $sp $A/ref/fin_s${j}_$sp.pkl || die "$arm infer $j $sp"; done
      $PY $GP/srcx/lib/avg_inf.py $A/ref/inf_fin_det1_$sp.pkl $A/ref/fin_s5_$sp.pkl $A/ref/fin_s6_$sp.pkl >> $A/logs/infer.log 2>&1
    done
  fi
  [ -f $A/deploy_$arm.json ] || CL_AO=$A $PY $L/chain_x.py deploy $W_FROZEN $arm > $A/logs/deploy.log 2>&1 || die "$arm deploy"
  for p in ${arm}_fused_val ${arm}_nofuse_val; do [ -f $A/preds/$p.pkl.score_val.json ] || CL_AO=$A $PY $L/score_x.py $A/preds/$p.pkl val >> $A/logs/score.log 2>&1; done
  for p in ${arm}_fused_test ${arm}_nofuse_test; do [ -f $A/preds/$p.pkl.score_real_test.json ] || CL_AO=$A $PY $L/score_x.py $A/preds/$p.pkl real_test >> $A/logs/score.log 2>&1 || die "$arm score $p"; done
  say "$arm scored"
done

# 4) summary: hard = srcx/in2mav (reused), feat / both = this run
SUM=$($PY -c "
import json, os
f = lambda p: json.load(open(p))['all']
fm = lambda d: 'UDS %.2f AP %.2f ATE %.3f AOE %.2f' % (d['UDS'], d['AP'], d['ATE'], d['AOE'])
P = {'hard': '$X/preds/pcmd_fused_%s.pkl.score_%s.json'}
for a in '$ARMS'.split():
    P[a] = '$BO/' + a + '/preds/' + a + '_fused_%s.pkl.score_%s.json'
out = []
for sp, tag in (('test', 'real_test'), ('val', 'val')):
    out.append('%s ' % sp + ' / '.join('%s %s' % (a, fm(f(p % (sp, tag)))) for a, p in P.items() if os.path.exists(p % (sp, tag))))
print(' | '.join(out))")
say "ALL DONE: $SUM   (hard = srcx/in2mav reused; FULL=$FULL; feat target $FEAT_TARGET; w_feat $WF w_soft $WS)"
# paired bootstrap vs PST afterwards: add ('PST', 'feat') / ('PST', 'both') to review1113/clean3/boot12.py ARMS with $BO/<arm>/preds/<arm>_fused_%s.pkl
