#!/bin/bash
# [hparam_1117] gamma sensitivity (exponent scaling l, w with the depth correction at deployment; chain_x.py ALPHA, deployed 0.5).
# Deployment only (CPU): for each existing PST run, re-run chain_x.py deploy with HP_GAMMA in {0, 0.5, 1.0}, everything else = the run
# (its det1, stage-3 refiner outputs, height readings, student readings, its own w). gamma 0.5 must reproduce the run's reported prediction
# file byte-for-byte (checked; then the run's existing test score is reused, test NOT rescored). gamma 0 / 1: val then test scored once.
# Writes only review1113/hparam_1117/gamma/<run>/ (lib, preds, deploy json, logs). Real labels only via score_x.py.
GP=E:/Open3DUAVDet/output/camnorm/geo1004/gproto; H=$GP/review1113/hparam_1117; PY=D:/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1 CUDA_VISIBLE_DEVICES=""
RUNS=${RUNS:-"mav_s0 mav_s10 mav_s20 anti_s0 anti_s10 anti_s20"}
mkdir -p $H/logs; LOG=$H/logs/gamma.log; say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
for r in $RUNS; do
  case $r in mav_s0) B=$GP/srcx/in2mav; SS= ;; mav_s10) B=$GP/review1113/sup3/run; SS=10 ;; mav_s20) B=$GP/review1113/seeds/mav_full_s20; SS=20 ;;
             anti_s0) B=$GP/srcx/out2antifs; SS= ;; anti_s10) B=$GP/review1113/seeds/antifs_full_s10; SS=10 ;; anti_s20) B=$GP/review1113/seeds/antifs_full_s20; SS=20 ;; esac
  case $r in mav*) export SX_SRC=in SX_TGT=mav; SPL=real_test ;; anti*) export SX_SRC=out SX_TGT=antifs; SPL=antifs_test ;; esac
  export SX_TEACH=full; if [ -n "$SS" ]; then export SX_SEEDSHIFT=$SS; else unset SX_SEEDSHIFT; fi
  A=$H/gamma/$r; mkdir -p $A/preds $A/logs
  $PY $H/make_hp_lib.py $r $A >> $A/logs/lib.log 2>&1 || { say "!! lib $r"; continue; }
  W=$($PY -c "import json; print(json.load(open('$B/w_sim.json'))['w'])")
  for g in 0 0.5 1.0; do
    t=g$(printf '%03d' $($PY -c "print(int(round($g*100)))"))
    [ -f $A/deploy_$t.json ] || HP_GAMMA=$g HP_IN=$B HP_STU=$B $PY $A/lib/chain_x.py deploy $W $t > $A/logs/deploy_$t.log 2>&1 || { say "!! deploy $r $t"; continue; }
  done
  # check: gamma 0.5 == reported prediction files
  $PY -c "
import pickle, numpy as np
for sp in ('val', 'test'):
    a = pickle.load(open('$A/preds/g050_fused_%s.pkl' % sp, 'rb')); b = pickle.load(open('$B/preds/pcmd_fused_%s.pkl' % sp, 'rb'))
    assert len(a) == len(b) and all(x['seq'] == y['seq'] and x['frame'] == y['frame'] and np.array_equal(x['boxes9d'], y['boxes9d']) for x, y in zip(a, b)), sp
print('gamma 0.5 reproduces the reported predictions exactly')" >> $A/logs/check.log 2>&1 || { say "!! $r: gamma 0.5 does NOT reproduce the reported predictions -> stop"; continue; }
  for t in g000 g100; do
    [ -f $A/preds/${t}_fused_val.pkl.score_val.json ] || $PY $A/lib/score_x.py $A/preds/${t}_fused_val.pkl val >> $A/logs/score.log 2>&1
    [ -f $A/preds/${t}_fused_test.pkl.score_$SPL.json ] || $PY $A/lib/score_x.py $A/preds/${t}_fused_test.pkl $SPL >> $A/logs/score.log 2>&1 || say "!! score $r $t"
  done
  say "$r done (w $W)"
done
