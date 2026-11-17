#!/bin/bash
# [M12] MAV6D single-model no-TTA PST (1 detector NL075_s0 no flip, 1 depth refiner XFZnl075_s5, 1 orientation student S6I_s0 no mirror,
# height network unchanged (1 net)), same frozen constants recipe as nolabel/run_chain_nl.sh; writes only under M12/M1.
M=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12; MB=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12
A=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/nolabel/nl075; PY=D:/Miniconda3/envs/city/python.exe
S=$M/M1/single; SB=$MB/M1/single; L=$MB/M1/logs
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet NL_ARM=nl075
mkdir -p $SB/unc $SB/kps $SB/tags $MB/M1/single_ens
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $L/run_m1_mav.log; }
for sp in val test; do cp -n $MB/M1/single_$sp.pkl $MB/M1/single_ens/ens_$sp.pkl; done
for sp in test val; do until [ -f $MB/M1/repl/nl075_${sp}_kzt7q/kz_members_nl075_${sp}_kzt7q.npz ]; do sleep 30; done; done
say "orientation dumps present"
for sp in test val; do
  [ -f $SB/tags/nl075_${sp}_ouS1/kz_members_nl075_${sp}_ouS1.json ] || (cd /e/Open3DUAVDet/tools && RM_REPL=$M/M1/repl RM_TAGS=$S/tags $PY $M/memcap.py $M/recut_mirror_m12.py --src-tag nl075_${sp}_kzt7q --sets "nl075_${sp}_ouS1=$A/adabn_all/S6I_s0.pth" > $L/recut_$sp.log 2>&1) || { say "recut $sp failed"; exit 2; }
done
say "recut done"
until [ -f $MB/M1/single_train.pkl ]; do sleep 30; done
[ -f $SB/unc/unc_train_z2.pkl ] || M12_RECS=$M/M1/single_%s.pkl M12_UOUT=$S/unc $PY $M/memcap.py $M/final_unc_m12.py nl075 z2 XFZnl075 5 val,test,train > $L/unc.log 2>&1 || { say "unc failed"; exit 2; }
say "unc done"
[ -f $SB/kps/shape_test_nl075_k0.pkl ] || M12_REPL=$M/M1/repl M12_KOUT=$S/kps $PY $M/memcap.py $M/kps_shape_m12.py nl075 shape val test > $L/kps.log 2>&1 || { say "kps failed"; exit 2; }
cp -n $A/kps/shape_train_nl075_k0.pkl $SB/kps/
say "kps done; scoring (once)"
cd $MB && M12_NLA=$S M12_ENS=$M/M1/single_ens M12_REPL=$M/M1/repl FE_UNC=z2 FE_ORI=ouS1 FE_MEMS=0 M12_WS=w M12_OUT=$M/M1/single_result $PY fuse_eval_m12.py > $L/fuse_eval_single.log 2>&1 || { say "fuse_eval failed"; exit 2; }
grep -v Warn $L/fuse_eval_single.log | tee -a $L/run_m1_mav.log
say "ALL DONE"
