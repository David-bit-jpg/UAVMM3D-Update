#!/bin/bash
# [review1113/cleanfix B3] "No target self-training" row on the CLEAN MAV6D chain (srcx/in2mav): the six orientation TEACHERS (IR+LiDAR x2,
#   IR-distilled x2, LiDAR x2; = the nets whose average is the pseudo label, ou/T_*.pth incl. their BN re-estimated on real train crops <= 80/seq
#   by run_x.sh) are read directly on the deployment crops; the deployment then uses the teachers' symmetric-ensemble rotation and corner-PnP
#   distance z_cor in place of the two students'. No student is trained. Everything else frozen = deployed (detector det1, refiners, c_k, r_k,
#   s_bar, height readings, omega 0.190); b_k re-estimated by the deploy rule from the unlabeled calibration images (val1, labels never read).
#   Reading = deployed reading recipe (MIRVIEW=1: + horizontal-mirror view, as the students; MIRVIEW=0 = the pseudo-label reading, no mirror).
#   The teacher readings are written as ou/read_{val1,te1}_stu.pkl inside this arm so that chain_x deploy (unchanged copy) consumes them.
#   Test scored ONCE. GPU: 6 nets x 2 views on 2,336 + 4,800 crops ~ 15-20 min (students 2 nets x 2 views took ~8 min in run_x.log).
# Output: B3_noself/arm/preds/noself_fused_test.pkl.score_real_test.json; logs/b3.log; done marker B3_noself/.done
#   launch: bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B3_noself/b3_noself.sh
source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/cf_common.sh
MIRVIEW=${MIRVIEW:-1}
B=$CF/B3_noself; A=$B/arm; mkdir -p $B/logs; LOG=$B/logs/b3.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $B/.done ] && { say "already done"; exit 0; }
wait_prereq $LOG
say "start B3 (teachers read on test crops, MIRVIEW=$MIRVIEW)"
setup_arm $A || { say "!! setup"; exit 2; }
T6=$(printf "$X/ou/T_%s.pth," irlid0 irlid1 kdir0 kdir1 lid0 lid1); T6=${T6%,}
for f in ${T6//,/ }; do [ -f $f ] || { say "!! teacher missing $f"; exit 2; }; done
for n in val1 te1; do
  [ -f $A/ou/read_${n}_stu.pkl ] || CL_AO=$A ORI_MIRROR=$MIRVIEW oomrun $A $A/logs/ori_infer.log 16 $SLOT light $PY $A/lib/infer_wrap.py $A/lib/ori_x.py read $n $A/ou/read_${n}_stu.pkl $T6 || { say "!! read $n"; exit 2; }
done
deploy_score $A noself 2>&1 | tee -a $LOG
[ -f $A/.done ] || { say "!! not done"; exit 2; }
SUM=$($PY -c "
import json
a = json.load(open('$X/preds/pcmd_fused_test.pkl.score_real_test.json'))['all']; b = json.load(open('$A/preds/noself_fused_test.pkl.score_real_test.json'))['all']
print('PST (students) AOE %.2f UDS %.2f AP %.2f -> teachers read directly AOE %.2f UDS %.2f AP %.2f' % (a['AOE'], a['UDS'], a['AP'], b['AOE'], b['UDS'], b['AP']))")
say "ALL DONE (test, 4,800 frames, MIRVIEW=$MIRVIEW): $SUM"; touch $B/.done
