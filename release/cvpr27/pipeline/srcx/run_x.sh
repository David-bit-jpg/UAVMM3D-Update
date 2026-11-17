#!/bin/bash
# [srcx 11-11] Source-domain ablation of PCMD: the ORIGINAL indoor-final simplified chain (wf24 README §6-7), only the simulated source swapped.
#   usage: bash run_x.sh <in|out|mix> <anti|antifs|mav>        outputs: srcx/<src>2<tgt>/ (+ large crops in D:/S2MUAV3D/srcx_work/<src>2<tgt>/)
# Method (unchanged): detector 3 seeds (ensemble + flip) -> Module 3 pseudo-labels (LAM .75, conf >= .5, refiner stage-2 x2) -> self-trained
# detector x3 (3 ep, init = same-seed sim detector) -> ensemble + flip; teachers 2 per group (IR+LiDAR, IR-distilled, LiDAR) -> pseudo-orientations,
# IR-vs-LiDAR gate 10 deg; orientation students S6I seeds 0,1 (init = same-seed IR+LiDAR member, mirror loss 12) + mirror view; depth refiner
# seeds 5,6 (stage 1 sim, 2 real adaptation, 3 fused pseudo-targets), outputs averaged; height network x1 (kps seed 0); w = inverse-variance on SIM
# validation; constants from unlabeled target images; no re-ranking. Real labels: read only by score_x.py (val sanity + ONE test scoring at the end).
# Every step skips if its output exists. GPU steps go through gproto/gpu_slot.sh.
# 11-12: self-trained detectors use --workers 0 via the light slot (no spawn dataloader) so they never wait behind other heavy trainings; recipe unchanged.
SRC=$1; TGT=$2; TEACH=${SX_TEACH:-full}; C=${SRC}2${TGT}; [ $TEACH = full ] || C=${C}_$TEACH
# 11-12 no-teacher control (SX_TEACH = full | ir | lid | none): only the privileged knowledge changes, every other step / constant is the same.
#   ir   : orientation teachers = the 2 IR-distilled nets (gate = their agreement <= 10 deg), students init from them; refiner keeps IR silhouettes;
#          height = constant aspect of s_bar (the height network is IR+LiDAR). Refiner / pseudo-labels / detectors are identical to full -> copied.
#   lid  : teachers = the 2 LiDAR nets (gate = their agreement), students init from them; refiner WITHOUT IR silhouettes (one colour class); constant height.
#   none : teachers = 2 plain RGB nets (noteach/OKZrgb_s0,1), students init from them; refiner without IR; constant height.
export SX_TEACH=$TEACH SX_SRC=$SRC SX_TGT=$TGT PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
H=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx; HW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx
X=$H/$C; XW=$HW/$C; L=$HW/lib; WK=/d/S2MUAV3D/srcx_work/$C
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $H/../gpu_slot.sh"; M=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm
mkdir -p $X/recs $X/ref $X/ou $X/preds $X/logs $WK; cd $X
LOG=$X/logs/run_x.log; say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }; die() { say "!! $1"; exit 2; }
say "start $C"
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))
print('IRL0=%s IRL1=%s KDI0=%s KDI1=%s LID0=%s LID1=%s KPS=%s' % (tuple(S['irlid']) + tuple(S['kdir']) + tuple(S['lid']) + (S['kps'],)))
print('REAL=%s ST_CFG=%s TAG=%s SPLIT=%s' % (sx.REAL, sx.ST_CFG, sx.PSEUDO_TAG, sx.T['score_split']))
")" || die "sx config"
waitf() { local n=0; until [ -f "$1" ]; do [ $n = 0 ] && say "waiting for $1"; n=1; sleep 120; done; }
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg
if [ $TEACH = ir ] && [ ! -f recs/det1.pkl ]; then   # identical recipe up to the orientation branch -> reuse the full run's outputs
  F=$H/${SRC}2${TGT}; FW=/d/S2MUAV3D/srcx_work/${SRC}2${TGT}; [ -f $F/recs/det1.pkl ] || die "full run missing"
  cp $F/recs/det0.pkl $F/recs/det1.pkl recs/; cp $F/preds/srconly_*.pkl preds/
  for f in sim_meta.pkl sim_rgb.npy sim_sil.npy bank.npy real_d0_meta.pkl real_d0_rgb.npy ref_st1_s5.pt ref_st1_s6.pt ref_st2_s5.pt ref_st2_s6.pt            st2_s5_det0_train.pkl st2_s6_det0_train.pkl inf_st2_det0_train.pkl st2_s5_det1_train.pkl st2_s6_det1_train.pkl inf_st2_det1_train.pkl; do cp $F/ref/$f ref/; done
  FT=$($PY -c "import sys; sys.path.insert(0,'$L'); import os; os.environ['SX_TEACH']='full'; import sx; print(sx.PSEUDO_TAG)")
  cp $F/pseudo_$FT.json pseudo_$TAG.json; cp $($PY -c "import sys; sys.path.insert(0,'$L'); import sx; print(sx.REAL)")/train/index_pseudo_$FT.pkl $($PY -c "import sys; sys.path.insert(0,'$L'); import sx; print(sx.REAL)")/train/index_pseudo_$TAG.pkl
  for f in oc_tr1.pkl oc_tr1.npy oc_val1.pkl oc_val1.npy oc_te1.pkl oc_te1.npy ou_bank.npy; do cp $FW/$f $WK/; done
  echo "$(date '+%m-%d %H:%M:%S') ir: copied detector / refiner / pseudo-label / crop outputs from ${SRC}2${TGT}" >> $LOG
fi

# ---------------- Module 2a/3 input: sim-only detectors x3 (ensemble + flip) on the target; s_bar from unlabeled val images
for c in $DET0 $DET1 $DET2; do waitf $c; done
[ -f recs/det0.pkl ] || $SLOT light $PY $L/det_infer_x.py --ckpt $DET0 $DET1 $DET2 --splits train val test --out $XW/recs/det0.pkl > logs/det0_infer.log 2>&1 || die "det0 infer"
grep -E "s_bar|frames" logs/det0_infer.log | tee -a $LOG
for sp in val test; do [ -f preds/srconly_$sp.pkl ] || $PY $L/to_pred.py $XW/recs/det0.pkl $sp $XW/preds/srconly_$sp.pkl >> $LOG 2>&1; done

# ---------------- Module 4 stages 1-2 (seeds 5, 6): sim samples (sim detector s0 on source-sim train), real background bank, real crops
[ -f ref/sim_meta.pkl ] || $SLOT light $PY $L/refiner_x.py sim $DET0 > logs/ref_sim.log 2>&1 || die "ref sim"
tail -1 logs/ref_sim.log | tee -a $LOG
[ -f ref/bank.npy ] || $PY $L/refiner_x.py bank $XW/recs/det0.pkl >> $LOG 2>&1 || die "ref bank"
[ -f ref/real_d0_meta.pkl ] || $PY $L/refiner_x.py real $XW/recs/det0.pkl d0 >> $LOG 2>&1 || die "ref real"
for j in 5 6; do ( [ -f ref/ref_st1_s$j.pt ] || $SLOT light $PY $L/refiner_x.py train 1 $j > logs/ref_st1_s$j.log 2>&1
                  [ -f ref/ref_st2_s$j.pt ] || REF_REAL=d0 $SLOT light $PY $L/refiner_x.py train 2 $j > logs/ref_st2_s$j.log 2>&1 ) & sleep 30; done; wait
for j in 5 6; do [ -f ref/ref_st2_s$j.pt ] || die "refiner st2 s$j"; done

# ---------------- Module 3: refiner stage-2 corrections (mean of seeds 5,6) on det0 train -> pseudo-labels -> self-trained detectors x3
for j in 5 6; do [ -f ref/st2_s${j}_det0_train.pkl ] || $SLOT light $PY $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det0.pkl train $XW/ref/st2_s${j}_det0_train.pkl >> $LOG 2>&1 || die "inf st2 det0 $j"; done
[ -f ref/inf_st2_det0_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det0_train.pkl $XW/ref/st2_s5_det0_train.pkl $XW/ref/st2_s6_det0_train.pkl >> $LOG 2>&1
[ -f pseudo_$TAG.json ] || $PY $L/make_pseudo_x.py $XW/recs/det0.pkl $XW/ref/inf_st2_det0_train.pkl $TAG >> $LOG 2>&1 || die pseudo
cat pseudo_$TAG.json >> $LOG
STN=$(basename $ST_CFG .yaml); stck() { echo $M/srcx/$STN/DET1_s$1/ckpt/checkpoint_epoch_3.pth; }
[ $TEACH = ir ] && STN=st_${SRC}2${TGT}   # ir reuses the full run's self-trained detectors
DETS=($DET0 $DET1 $DET2)
for k in 0 1 2; do
  ( [ -f $(stck $k) ] || { cd /e/Open3DUAVDet/tools; $SLOT light $PY train.py --cfg_file $ST_CFG --batch_size 8 --workers 0 --fix_random_seed --cudnn_benchmark --use_amp \
      --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval --seed $k --extra_tag DET1_s$k --pretrained_model ${DETS[$k]} > $X/logs/train_det1_s$k.log 2>&1; } ) &
  sleep 100
done; wait
for k in 0 1 2; do [ -f $(stck $k) ] || die "self-trained detector s$k"; done
[ -f recs/det1.pkl ] || $SLOT light $PY $L/det_infer_x.py --ckpt $(stck 0) $(stck 1) $(stck 2) --splits train val test --out $XW/recs/det1.pkl --size-fallback $XW/recs/det0.pkl > logs/det1_infer.log 2>&1 || die "det1 infer"
grep -E "s_bar|frames" logs/det1_infer.log | tee -a $LOG
say "detectors done"

# ---------------- Module 5: orientation crops on the final detector, real background bank, 6 teachers (BN on real train crops <= 80/seq)
for x in "train tr1" "val val1" "test te1"; do set -- $x; [ -f $WK/oc_$2.pkl ] || $PY $L/ori_x.py crops $XW/recs/det1.pkl $1 1 $2 >> $LOG 2>&1 || die "crops $2"; done
[ -f $WK/ou_bank.npy ] || $PY $L/ori_x.py bank $XW/recs/det1.pkl ou_bank >> $LOG 2>&1 || die "ou bank"
case $TEACH in
  full) TL="irlid0 $IRL0 irlid1 $IRL1 kdir0 $KDI0 kdir1 $KDI1 lid0 $LID0 lid1 $LID1"; GA=kdir0; GB=lid0; GA2=kdir1; GB2=lid1; INIT=($IRL0 $IRL1) ;;
  ir)   TL="kdir0 $KDI0 kdir1 $KDI1"; INIT=($KDI0 $KDI1) ;;
  lid)  TL="lid0 $LID0 lid1 $LID1"; INIT=($LID0 $LID1) ;;
  none) RG=$M/sim_out_r34_kp_w12_zoom_out; RG0=$RG/OKZrgb_s0/ckpt/checkpoint_epoch_30.pth; RG1=$RG/OKZrgb_s1/ckpt/checkpoint_epoch_30.pth
        # [11-13] in source: the plain RGB nets are the indoor KZ seeds 0,1 (source kp_cfg sim_full_r34_kp_w12_zoom, epoch 30 = OKZrgb's indoor twin)
        [ $SRC = in ] && { RG=$M/sim_full_r34_kp_w12_zoom; RG0=$RG/KZs0/ckpt/checkpoint_epoch_30.pth; RG1=$RG/KZs1/ckpt/checkpoint_epoch_30.pth; }
        [ $SRC = out ] || [ $SRC = in ] || die "none: plain RGB orientation nets only trained for the out / in sources"
        TL="rgb0 $RG0 rgb1 $RG1"; INIT=($RG0 $RG1) ;;
esac
set -- $TL; TN=(); while [ -n "$1" ]; do TN+=($1); waitf $2; [ -f ou/T_$1.pth ] || $SLOT light $PY $L/ori_x.py adabn $2 tr1 $XW/ou/T_$1.pth 80 >> $LOG 2>&1 || die "adabn $1"; shift 2; done
T6=$(printf "$XW/ou/T_%s.pth," "${TN[@]}"); T6=${T6%,}
[ -f ou/read_tr1_teach.pkl ] || $SLOT light $PY $L/ori_x.py read tr1 $XW/ou/read_tr1_teach.pkl $T6 >> $LOG 2>&1 || die "read teachers"
q() { printf '"%s/ou/T_%s.pth"' $XW $1; }
if [ $TEACH = full ]; then GRP="{\"T\": [$(q irlid0), $(q irlid1), $(q kdir0), $(q kdir1), $(q lid0), $(q lid1)], \"IR\": [$(q kdir0), $(q kdir1)], \"LiD\": [$(q lid0), $(q lid1)]}"
else GRP="{\"T\": [$(q ${TN[0]}), $(q ${TN[1]})], \"IR\": [$(q ${TN[0]})], \"LiD\": [$(q ${TN[1]})]}"; fi   # single modality: gate = agreement of its 2 seeds
[ -f $WK/pseudo_tr1.npz ] || $PY $L/ori_x.py pseudo tr1 $XW/ou/read_tr1_teach.pkl pseudo_tr1 "$GRP" >> $LOG 2>&1 || die pseudo_ori
grep "pseudo pseudo_tr1" $LOG | tail -1
# students S6I seeds 0, 1 (init = same-seed IR+LiDAR member; gated pseudo-rotations + mirror loss 12, background swap 0.5, 2250 steps)
for s in 0 1; do ( [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || OU_INIT=${INIT[$s]} $SLOT light $PY $L/ou_train_x.py --seed $s --tag S6I --tgt gate --mir 12 --pseudo pseudo_tr1 --runs-dir $XW/ou/runs > logs/ou_train_s$s.log 2>&1 ) & sleep 100; done; wait
for s in 0 1; do [ -f ou/runs/S6I_s$s/ckpt_last.pth ] || die "student s$s"
  [ -f ou/S_ab$s.pth ] || $SLOT light $PY $L/ori_x.py adabn $XW/ou/runs/S6I_s$s/ckpt_last.pth tr1 $XW/ou/S_ab$s.pth 0 >> $LOG 2>&1 || die "student adabn $s"; done
for n in tr1 val1 te1; do [ -f ou/read_${n}_stu.pkl ] || ORI_MIRROR=1 $SLOT light $PY $L/ori_x.py read $n $XW/ou/read_${n}_stu.pkl $XW/ou/S_ab0.pth,$XW/ou/S_ab1.pth >> $LOG 2>&1 || die "read $n"; done
# height network (1, kps seed 0): BN on train crops (<= 80/seq), readings
if [ $TEACH = full ]; then
  waitf $KPS
  [ -f kps_te1.pkl ] || $SLOT light $PY $L/ori_x.py kps $KPS tr1,val1,te1 "$XW/kps_%s.pkl" 80 >> $LOG 2>&1 || die kps
else   # no height network (it is IR+LiDAR trained): every crop gets the constant aspect h / mean(l, w) of s_bar
  [ -f kps_te1.pkl ] || $PY -c "
import pickle, numpy as np
s = np.asarray(pickle.load(open('$XW/recs/det1.pkl', 'rb'))['size'], float); r = float(s[2] / ((s[0] + s[1]) / 2))
for n in ('tr1', 'val1', 'te1'):
    I = pickle.load(open('D:/S2MUAV3D/srcx_work/$C/oc_%s.pkl' % n, 'rb'))['info']
    pickle.dump({(f['seq'], f['frame']): r for f in I}, open('$XW/kps_%s.pkl' % n, 'wb'))
print('constant height aspect', r)" >> $LOG 2>&1 || die "const kps"
fi
say "orientation branch done"

# ---------------- Module 4 stage 3 with fused pseudo-targets; w by inverse variance on SIM validation
for j in 5 6; do [ -f ref/st2_s${j}_det1_train.pkl ] || $SLOT light $PY $L/refiner_x.py infer $XW/ref/ref_st2_s$j.pt $XW/recs/det1.pkl train $XW/ref/st2_s${j}_det1_train.pkl >> $LOG 2>&1 || die "inf st2 det1 $j"; done
[ -f ref/inf_st2_det1_train.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_st2_det1_train.pkl $XW/ref/st2_s5_det1_train.pkl $XW/ref/st2_s6_det1_train.pkl >> $LOG 2>&1
[ -f w_sim.json ] || $SLOT light $PY $L/wsim_x.py $DET0 $XW/ref/ref_st2_s5.pt,$XW/ref/ref_st2_s6.pt $XW/ou/runs/S6I_s0/ckpt_last.pth,$XW/ou/runs/S6I_s1/ckpt_last.pth $XW/w_sim.json >> $LOG 2>&1 || die wsim
WW=$($PY -c "import json;print(json.load(open('$XW/w_sim.json'))['w'])"); say "w (sim val inverse variance) = $WW"
[ -f ref/dp_fused.npy ] || $PY $L/chain_x.py targets $WW >> $LOG 2>&1 || die targets
for j in 5 6; do ( [ -f ref/ref_st3_s$j.pt ] || REF_REAL=d0 REF_DP=$XW/ref/dp_fused.npy $SLOT light $PY $L/refiner_x.py train 3 $j > logs/ref_st3_s$j.log 2>&1 ) & sleep 30; done; wait
for j in 5 6; do [ -f ref/ref_st3_s$j.pt ] || die "ref st3 $j"; done
for sp in train val test; do
  for j in 5 6; do [ -f ref/fin_s${j}_$sp.pkl ] || $SLOT light $PY $L/refiner_x.py infer $XW/ref/ref_st3_s$j.pt $XW/recs/det1.pkl $sp $XW/ref/fin_s${j}_$sp.pkl >> $LOG 2>&1 || die "inf fin $j $sp"; done
  [ -f ref/inf_fin_det1_$sp.pkl ] || $PY $L/avg_inf.py $XW/ref/inf_fin_det1_$sp.pkl $XW/ref/fin_s5_$sp.pkl $XW/ref/fin_s6_$sp.pkl >> $LOG 2>&1
done

# ---------------- Module 6: constants + single-frame deployment; scoring (val = sanity only; test = once)
[ -f deploy_pcmd.json ] || $PY $L/chain_x.py deploy $WW pcmd >> $LOG 2>&1 || die deploy
for p in srconly_val pcmd_fused_val pcmd_nofuse_val; do [ -f preds/$p.pkl.score_val.json ] || $PY $L/score_x.py $XW/preds/$p.pkl val >> $LOG 2>&1; done
say "val sanity: $(grep -h -o '"UDS": [0-9.]*' preds/pcmd_fused_val.pkl.score_val.json | head -1)"
for p in srconly_test pcmd_fused_test; do [ -f preds/$p.pkl.score_$SPLIT.json ] || $PY $L/score_x.py $XW/preds/$p.pkl $SPLIT >> $LOG 2>&1 || die "score $p"; done
say "ALL DONE"
