#!/bin/bash
# [review1113 stage1seeds] ONE full-reseed run: downstream chain on the RE-SEEDED stage-1 networks of seed set <sd>. NOT RUN by its author.
#   usage: bash run_group.sh <group full|E1|E4> <tgt mav|antifs> <sd 30|40>        DRY=1: build the lib, print the config, check inputs; no GPU
#   group  full = PST (SX_TEACH=full)   E1 = sensor-free control (RGB-only teachers + box-hull mattes)   E4 = pure RGB self-training (no composites)
# Chain = seeds/run_seed_chain.sh (PST / teach none) with sensorfree/run_sf.sh's E1 and E4 switches, except that the stage-1 outputs are NOT
#   copied from the reported run: they come from shared/<tgt>_sd<sd> (prep_shared.sh, re-seeded sim detectors), and every stage-1 network the
#   chain loads is the re-seeded one (lib/sx.py SOURCES override, s1paths.py):
#     self-trained detectors x3: init = re-seeded sim detector k, seed k+sd; refiner stages 1-3: seeds 5+sd, 6+sd; students: seeds 0+sd, 1+sd,
#     init = re-seeded IR+LiDAR member k (full) or re-seeded plain-RGB member k (E1/E4); teachers (full) = re-seeded IR+LiDAR / IR-distilled /
#     LiDAR pairs; height (full) = re-seeded height net, (E1/E4) constant aspect of s_bar.
#   E1: refiner composites ON with alpha = shared/ref/sim_hull.npy (as ref/sim_sil.npy, also the region-head target, SILW 1.0), students'
#       composite alpha = sensorfree hull_<src>.npy (stage-1-independent source crops), paste 0.5.  E4: refiner none-arm, students --paste 0.
#   Run k of the three groups uses the same re-seeded detectors (=> identical det0) and the same downstream seed shift.
# Real labels: TEST only, via score_x.py, once (pcmd_fused_test; the full run also scores srconly_test = the re-seeded sim-only detectors).
#   No validation scoring. Summary: summarize.py (pooled with the reported run, heads_boot.py bootstrap).
# Writes ONLY: review1113/stage1seeds/runs/<tgt>_<group>_s1sd<sd>/, D:/S2MUAV3D/srcx_work/s1seeds/<same>/ (crops, ~9 GB MAV6D),
#   tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_<C>_s1<group>sd<sd>.yaml, <target cache>/train/index_pseudo_srcx_<C>_s1<group>sd<sd>.pkl,
#   D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm/srcx/st_<C>_s1<group>sd<sd>/DET1_s{0,1,2}/ (via s1_train.py: E: is almost full)
#   (C = in2mav | in2mav_none | out2antifs | out2antifs_none).
# Robustness (= run_seed_chain.sh): inference via gpu_slot light + infer_wrap (memory fraction 0.2); CUDA OOM -> batch halved (max 4, logs/retry.log,
#   RECIPE DEVIATION for training batches); trainings resume; steps skipped once their output exists; workers 0; SERIAL=1 runs paired trainings
#   one after another; markers .done/.failed/.skipped; LATEST_START=MMDDHHMM (default 11241000) = do not start a new run after that; never kills.
GROUP=$1; TGT=$2; SD=$3; DRY=${DRY:-0}; SERIAL=${SERIAL:-0}
case "$GROUP/$TGT/$SD" in full/mav/30|full/mav/40|full/antifs/30|full/antifs/40|E1/mav/30|E1/mav/40|E1/antifs/30|E1/antifs/40|E4/mav/30|E4/mav/40|E4/antifs/30|E4/antifs/40) ;;
  *) echo "usage: run_group.sh <full|E1|E4> <mav|antifs> <30|40>"; exit 2 ;; esac
source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/common.sh
SRC=in; SPLIT=real_test; [ $TGT = antifs ] && { SRC=out; SPLIT=antifs_test; }
TEACH=full; [ $GROUP = full ] || TEACH=none
RUN=${TGT}_${GROUP}_s1sd$SD; X=$S1/runs/$RUN; XW=$(wp $X); L=$XW/lib; IW="$PY $L/infer_wrap.py"
SH=$S1/shared/${TGT}_sd$SD; SHW=$(wp $SH)
HULLS=/d/S2MUAV3D/srcx_work/sensorfree/hull_$SRC.npy
export SX_SEEDSHIFT=$SD SX_TEACH=$TEACH SX_SRC=$SRC SX_TGT=$TGT
mkdir -p $S1/logs; TAGL="$GROUP $TGT sd$SD"; MARK=$X/
case $GROUP in full) NETS="det:0 det:1 det:2 irlid:0 irlid:1 kdir:0 kdir:1 lid:0 lid:1 kps:0" ;; *) NETS="det:0 det:1 det:2 rgb:0 rgb:1" ;; esac

if [ $DRY = 1 ]; then
  LOG=/dev/null; GLOG=/dev/null
  $PY $S1W/make_s1_lib.py $GROUP $TGT $SD || exit 2
  SX_NOMKDIR=1 CUDA_VISIBLE_DEVICES="" $PY -c "
import sys, os; sys.path.insert(0, '$L'); import sx; S = sx.S
print('  run %s | combo %s | teach %s NOIR %s | seedshift %s | score split %s' % ('$RUN', sx.COMBO, sx.TEACH, sx.NOIR, os.environ['SX_SEEDSHIFT'], sx.T['score_split']))
print('  O %s | W %s' % (sx.O, sx.W)); print('  REAL_CFG %s | ST_CFG %s | PSEUDO_TAG %s' % (sx.REAL_CFG, sx.ST_CFG, sx.PSEUDO_TAG))
use = {'full': ['det', 'irlid', 'kdir', 'lid', 'kps'], 'none': ['det', 'rgb']}[sx.TEACH]
for k in use:
    v = S[k] if isinstance(S[k], list) else [S[k]]
    print('  stage-1 %-5s %s' % (k, ' | '.join('%s (%s)' % (p.split('/')[-3] if 'ckpt' in p else p.split('/')[-1], 'exists' if os.path.exists(p) else 'to be trained') for p in v)))
st = 'D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm/srcx/' + os.path.basename(sx.ST_CFG)[:-5]
for p in (sx.O + '/.done', sx.W, sx.ROOT + '/tools/' + sx.ST_CFG, '%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG), st):
    assert not os.path.exists(p), 'name collision / already exists: ' + p
print('  new names free: W, ST_CFG, pseudo index, self-trained detector dir', st)
assert sx.O == '$XW' and sx.TEACH == '$TEACH' and sx.PSEUDO_TAG.endswith('_s1${GROUP}sd$SD')" || exit 2
  echo "  shared stage-1 outputs: $SHW ($([ -f $SH/.done ] && echo ready || echo 'built by prep_shared.sh before this run'))"
  if [ $GROUP = E1 ]; then [ -f $HULLS ] && echo "  student box hull $HULLS ok" || { echo "  !! student hull $HULLS missing"; exit 2; }
    grep -q "SF_COMPOSITE = True" $L/refiner_x.py && grep -q "^SILW = 1.0" $L/refiner_x.py && grep -q "sensorfree E1\] student composite matte" $L/ou_train_x.py && echo "  E1 patches present (refiner hull composites + SILW 1, student hull)" || exit 2; fi
  [ $GROUP = E4 ] && echo "  E4: students --paste 0; refiner none-arm (no composites, SILW 0)"
  exit 0
fi

[ -f $X/.done ] && { mkdir -p $X/logs; LOG=$X/logs/run.log; say "already done: $(grep 'ALL DONE' $LOG | tail -1 | cut -c1-250)"; exit 0; }
mkdir -p $X; LOG=/dev/null; latest_start 11241000 $X/logs
mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs; LOG=$X/logs/run.log; RLOG=$X/logs/retry.log
rm -f $X/.failed $X/.skipped
say "start $RUN (SX_TEACH=$TEACH, SX_SEEDSHIFT=$SD, SERIAL=$SERIAL, stage-1 seed set $SD)"
$PY $S1W/make_s1_lib.py $GROUP $TGT $SD >> $LOG 2>&1 || die make_s1_lib
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
assert sx.O == '$XW' and sx.TEACH == '$TEACH', (sx.O, sx.TEACH)
print('DET0=%s DET1=%s DET2=%s KPS=%s RG0=%s RG1=%s' % (tuple(S['det']) + (S['kps'],) + tuple(S['rgb'])))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid'])))
print('REAL=%s ST_CFG=%s TAG=%s WK=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.W))")" || die "sx config"
WKP=$(pp $WK); mkdir -p $WKP; cd $X
par() { local s=$1; shift; if [ $SERIAL = 1 ]; then "$@"; else "$@" & sleep $s; fi; }
waitf $SH/.done
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg

# ---------------- shared stage-1 outputs of this seed set (re-seeded sim detectors)
cpf $SH/recs/det0.pkl recs/det0.pkl
for sp in val test; do cpf $SH/preds/srconly_$sp.pkl preds/srconly_$sp.pkl; done
for f in sim_meta.pkl real_d0_meta.pkl; do cpf $SH/ref/$f ref/$f; done
for f in sim_rgb.npy bank.npy real_d0_rgb.npy; do lnk $SH/ref/$f ref/$f; [ -f ref/$f ] || die "link ref/$f"; done
if [ $GROUP = E1 ]; then lnk $SH/ref/sim_hull.npy ref/sim_sil.npy; cmp -s ref/sim_sil.npy $SH/ref/sim_hull.npy || die "hull array differs"
  say "refiner silhouette/matte array = box hull of the same sim samples (shared/ref/sim_hull.npy)"
else lnk $SH/ref/sim_sil.npy ref/sim_sil.npy; cmp -s ref/sim_sil.npy $SH/ref/sim_sil.npy || die "sim_sil differs"; fi
[ -f wsim_cache.pkl ] || { [ -f $SH/wsim_cache.pkl ] && cp $SH/wsim_cache.pkl wsim_cache.pkl; }
say "shared stage-1 outputs linked from $SHW"

# ---------------- Module 4 stages 1-2 (seeds 5+sd, 6+sd)
ref12() { local j=$1
  [ -f ref/ref_st1_s$j.pt ] || oomrun logs/ref_st1_s$j.log "REF_ST1_BS=128" $SLOT light $PY $L/refiner_x.py train 1 $j
  [ -f ref/ref_st1_s$j.pt ] && { [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 oomrun logs/ref_st2_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 2 $j; }; }
for j in 5 6; do par 30 ref12 $j; done; wait
for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done
[ $GROUP = E1 ] && say "refiner: $(grep -h '\[sensorfree\] composites' logs/ref_st1_s5.log | tail -1)"
say "refiners stage 1-2 done"

# ---------------- Module 3: pseudo-labels -> self-trained detectors x3 (init = re-seeded sim detector k, seed k+sd)
for j in 5 6; do [ -f ref/st2_s${j}_det0_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det0.pkl train $XW/ref/st2_s${j}_det0_train.pkl || die "inf st2 det0 $j"; done
[ -f ref/inf_st2_det0_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det0_train.pkl $XW/ref/st2_s5_det0_train.pkl $XW/ref/st2_s6_det0_train.pkl >> $LOG 2>&1 || die "avg st2 det0"
[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
say "pseudo-labels: $(tr -d '\n ' < pseudo_$TAG.json | cut -c1-200)"
STN=$(basename $ST_CFG .yaml); MS1=D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm; stck() { echo $MS1/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
DETS=($DET0 $DET1 $DET2)
train_det1() {
  local k=$1 bs=8 try=0 lg=$X/logs/train_det1_s$1.log
  while [ ! -f $(stck $k) ]; do
    ( cd /e/Open3DUAVDet/tools; $SLOT light $PY $S1W/s1_train.py E:/Open3DUAVDet/tools/train.py --cfg_file $ST_CFG --batch_size $bs --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
        --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $((k + SD)) --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} ) > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f $(stck $k) ] && break
    if isoom $lg.try && [ $try -lt 4 ] && [ $bs -gt 1 ]; then try=$((try + 1)); bs=$((bs / 2))
      dev "self-trained detector s$k OOM -> retry $try/4 with --batch_size $bs, resuming (RECIPE DEVIATION)"
    else dev "self-trained detector s$k FAILED, see $lg"; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for k in 0 1 2; do par 100 train_det1 $k; done; wait
for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
[ -f recs/det1.pkl ] || oomrun logs/det1_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl || die "det1 infer"
grep -E "s_bar|frames" logs/det1_infer.log | tail -4 >> $LOG
say "detectors done"

# ---------------- Module 5: crops, bank, teachers (re-seeded), gate, students
for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WKP/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
[ -f $WKP/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
waitnet $SRC $SD $NETS
case $TEACH in
  full) TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; INIT=($IRL0 $IRL1) ;;
  none) TL="rgb0 $RG0 rgb1 $RG1"; INIT=($RG0 $RG1) ;;
esac
set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); [ -f $2 ] || die "teacher $1 missing $2"
  [ -f ou/T_$1.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 || die "adabn $1"; shift 2; done
T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
[ -f ou/read_tr1_teach.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 || die "read teachers"
q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
if [ $TEACH = full ]; then GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
else GRP="{\"T\": [$(q ${TN[0]}), $(q ${TN[1]})], \"IR\": [$(q ${TN[0]})], \"LiD\": [$(q ${TN[1]})]}"; fi
[ -f $WKP/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
say "$(grep 'pseudo pseudo_tr1' $LOG | tail -1)"
SARGS=""; [ $GROUP = E4 ] && SARGS="--paste 0"
train_student() {
  local s=$1 bs=8 bt=4 try=0 lg=$X/logs/ou_train_s$1.log
  while [ ! -f ou/runs/S6I_s$s/ckpt_last.pth ]; do
    OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $XW/ou/runs --bs-src $bs --bs-tgt $bt $SARGS > $lg.try 2>&1
    cat $lg.try >> $lg
    [ -f ou/runs/S6I_s$s/ckpt_last.pth ] && break
    if isoom $lg.try && [ $try -lt 4 ]; then try=$((try + 1)); bs=$(( bs > 1 ? bs / 2 : 1 )); bt=$(( bt > 1 ? bt / 2 : 1 ))
      dev "student s$s OOM -> retry $try/4 with --bs-src $bs --bs-tgt $bt, resuming (RECIPE DEVIATION)"
    else dev "student s$s FAILED, see $lg"; rm -f $lg.try; return 1; fi
  done; rm -f $lg.try
}
for s in 0 1; do par 100 train_student $s; done; wait
for s in 0 1; do [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || die "student s$s"
  grep -h -m1 "INFO  init " logs/ou_train_s$s.log | sed 's/ | steps.*//' >> $LOG
  [ $GROUP = E1 ] && { grep -q "sensorfree E1\] student composite matte" logs/ou_train_s$s.log || die "student s$s did not use the box-hull matte"; }
  [ $GROUP = E4 ] && { grep -q "'paste': 0.0" logs/ou_train_s$s.log || die "student s$s: paste not 0"; }
  [ -f ou/S_ab$s.pth ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 || die "student adabn $s"; done
for n in tr1 val1 te1; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth || die "read $n"; done
if [ $TEACH = full ]; then
  [ -f kps_te1.pkl ] || oomrun logs/ori_infer.log "ORI_BS=16" $SLOT light $IW $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 || die kps
else
  [ -f kps_te1.pkl ] || $PY -c "
import pickle, numpy as np
s = np.asarray(pickle.load(open('$XW/recs/det1.pkl', 'rb'))['size'], float); r = float(s[2] / ((s[0] + s[1]) / 2))
for n in ('tr1', 'val1', 'te1'):
    I = pickle.load(open('$WK/oc_%s.pkl' % n, 'rb'))['info']
    pickle.dump({(f['seq'], f['frame']): r for f in I}, open('$XW/kps_%s.pkl' % n, 'wb'))
print('constant height aspect', r)" >> $LOG 2>&1 || die "const kps"
fi
say "orientation branch done"

# ---------------- Module 4 stage 3 (fused pseudo-targets; w by inverse variance on SIM validation)
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1 || die "avg st2 det1"
[ -f wsim_cache.pkl ] || { [ -f $SH/wsim_cache.pkl ] && cp $SH/wsim_cache.pkl wsim_cache.pkl; }
[ -f w_sim.json ] || oomrun logs/wsim.log "ORI_BS=16 REF_BS=128" $SLOT light $IW $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json || die wsim
[ -f $SH/wsim_cache.pkl ] || { cp wsim_cache.pkl $SH/wsim_cache.pkl.tmp$$ && mv -n $SH/wsim_cache.pkl.tmp$$ $SH/wsim_cache.pkl; rm -f $SH/wsim_cache.pkl.tmp$$; }   # sim-detector-s0 inference on sim val: identical for every group of this seed set
WW=$($PY -c "import json;print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
ref3() { local j=$1; [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy oomrun logs/ref_st3_s$j.log "REF_TRAIN_BS=64" $SLOT light $PY $L/refiner_x.py train 3 $j; }
for j in 5 6; do par 30 ref3 $j; done; wait
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || oomrun logs/infer.log "REF_BS=128" $SLOT light $IW $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1 || die "avg fin $sp"
done

# ---------------- Module 6: constants + single-frame deployment; TEST scored once
[ -f deploy_pcmd.json ] || $PY $L/chain_x.py deploy $WW pcmd >> $LOG 2>&1 || die deploy
SC="pcmd_fused_test"; [ $GROUP = full ] && SC="srconly_test pcmd_fused_test"
for p in $SC; do [ -f preds/$p.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/$p.pkl $SPLIT >> $LOG 2>&1 || die "score $p"; done
SUM=$($PY -c "
import json; f = lambda p: json.load(open(p))['all']
fm = lambda d: 'UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f' % (d['UDS'], d['AP'], d['ATE'], d['ASE'], d['AOE'], d['mDR'])
print('test %s | %s w %.4f' % (fm(f('$XW/preds/pcmd_fused_test.pkl.score_$SPLIT.json')), '$RUN', $WW))") || die summary
NDEV=$(grep -c "RECIPE DEVIATION" $RLOG 2>/dev/null); NDEV=${NDEV:-0}
say "ALL DONE: $SUM || recipe deviations: $NDEV"
touch $X/.done
