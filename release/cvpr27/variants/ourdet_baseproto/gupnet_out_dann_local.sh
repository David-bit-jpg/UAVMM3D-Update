#!/bin/bash
# [11-16 user request] GUPNet + DANN on RealDrone3D (outdoor), run locally because the server's outdoor sim GUPNet checkpoint is not here.
#   step 1: GUPNet sim-only on SimDrone3D-outdoor (same local recipe as D:/cmpfull_runs/gupnet_s0: 140 epochs x 3712 samples, batch 8, seed 0)
#   step 2: predict RealDrone3D test with it (reported next to the server's sim-only row, to show the local start point)
#   step 3: DANN from that checkpoint = same protocol as the MAV6D GUPNet + DANN (cmpfull/run1110/gupnet_uda_run.sh): 40 epochs x 3712 source
#           samples, batch 8, constant lr 3.125e-5, unlabeled RealDrone3D train images (labels stripped), final checkpoint
#   step 4: predict RealDrone3D test, score once with the stable-sort scorer.
# Robust: every step skipped when its output exists; resume from newest checkpoint; CUDA OOM -> halve batch; workers 0 (no spawn workers).
# Runs through gpu_slot.sh light with "ourdet_baseproto" in the command (urgent main-paper rule). Never kills processes.
H=/e/Open3DUAVDet/output/camnorm/geo1004; C=$H/cmpfull/gupnet; PY=D:/Miniconda3/envs/city/python.exe
SLOT=$H/gproto/gpu_slot.sh; ME=$H/gproto/review1113/ourdet_baseproto/gupnet_out_dann_local.sh
S=D:/cmpfull_runs/gupnet_out_sim_local_s0; D=D:/cmpfull_runs/gupnet_out_dann_local_s0
LOG=$H/gproto/review1113/ourdet_baseproto/logs/gupnet_out_dann_local.log
export PYTHONUTF8=1
say() { echo "$(date '+%m-%d %H:%M:%S') [gupnet_out_dann] $1" | tee -a $LOG; }
p() { echo "$1" | sed -E 's#^([A-Za-z]):#/\L\1#'; }
score() { $PY - "$1" <<'EOF'
import sys, pickle, numpy as np
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts'); import score as S
M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'); key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(key)
P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
for r in pickle.load(open(sys.argv[1], 'rb')):
    i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], float).reshape(-1, 9); sc = np.asarray(r['scores'], float).reshape(-1)
    if i is None or not len(b): continue
    j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
m = S.metrics(P, M['gt'], M['K'], np.arange(N), S.CFG['outdoor_test'])
print('AP %.2f ATE %.1f%% ASE %.2f AOE %.2f mDR %.1f UDS %.2f' % (m['AP'], 100 * m['ATE'], m['ASE'], m['AOE'], m['mDR'], m['UDS']))
EOF
}
case "$1" in
  sim)  B=8; mkdir -p $(p $S)
        for try in 0 1 2 3; do
          R=$(ls -t $(p $S)/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | head -1); RA=; [ -n "$R" ] && RA="--resume $(echo $R | sed -E 's#^/([a-z])/#\U\1:/#')"
          ( cd $C && GPU_FRAC=0.3 bash train.sh --labels D:/cmpfull/sim_outdoor_train/labels.pkl --out $S --seed 0 --epochs 140 --batch-size $B \
              --workers 0 --mean-size-from D:/cmpfull/sim_outdoor_train/labels.pkl $RA ) >> $(p $S)/train.out 2>&1
          [ -f $(p $S)/checkpoints/checkpoint_final.pth ] && exit 0
          grep -q "out of memory" $(p $S)/train.out && [ $B -gt 1 ] && { B=$((B / 2)); echo "OOM -> batch $B (RECIPE DEVIATION)" >> $LOG; continue; }
          [ $try -ge 1 ] && exit 1
        done; exit 1 ;;
  dann) B=8; mkdir -p $(p $D)
        for try in 0 1 2 3; do
          GPU_FRAC=0.35 GUPNET_NO_PRETRAIN=1 $PY $C/code/tools/uda_train.py --method dann --init $S/checkpoints/checkpoint_final.pth \
            --src-labels D:/cmpfull/sim_outdoor_train/labels.pkl --tgt-labels D:/cmpfull/outdoor_train/labels.pkl --out $D --seed 0 \
            --epochs 40 --epoch-size 3712 --batch-size $B --workers 0 --resume auto >> $(p $D)/train.out 2>&1; rc=$?
          [ $rc = 0 ] && [ -f $(p $D)/checkpoints/checkpoint_final.pth ] && exit 0
          [ $rc = 3 ] && [ $B -gt 1 ] && { B=$((B / 2)); echo "OOM -> batch $B (RECIPE DEVIATION)" >> $LOG; continue; }
          [ $try -ge 1 ] && exit 1
        done; exit 1 ;;
  pred) GPU_FRAC=0.15 $PY $C/predict.py --ckpt $2/checkpoints/checkpoint_final.pth --split outdoor_test --out $2/pred_outdoor_test.pkl --workers 0 --batch-size 8 ;;
  "")   say "start"
        [ -f $(p $S)/checkpoints/checkpoint_final.pth ] || { say "step 1 GUPNet sim-only outdoor (140 ep)"; bash $SLOT light bash $ME sim || { say "!! sim training failed"; exit 1; }; }
        [ -f $(p $S)/pred_outdoor_test.pkl ] || bash $SLOT light bash $ME pred $S > $(p $S)/pred.log 2>&1 || { say "!! sim predict failed"; exit 1; }
        say "local sim-only GUPNet on RealDrone3D test: $(score $S/pred_outdoor_test.pkl)  (server sim-only row: UDS 14.61)"
        [ -f $(p $D)/checkpoints/checkpoint_final.pth ] || { say "step 3 DANN (40 ep)"; bash $SLOT light bash $ME dann || { say "!! DANN failed"; exit 1; }; }
        [ -f $(p $D)/pred_outdoor_test.pkl ] || bash $SLOT light bash $ME pred $D > $(p $D)/pred.log 2>&1 || { say "!! DANN predict failed"; exit 1; }
        say "ALL DONE GUPNet + DANN on RealDrone3D test: $(score $D/pred_outdoor_test.pkl)" ;;
esac
