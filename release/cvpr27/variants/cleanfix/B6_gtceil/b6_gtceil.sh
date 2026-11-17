#!/bin/bash
# [review1113/cleanfix B6] ORACLE ceiling on the CLEAN MAV6D chain: ground-truth rotations as the orientation pseudo labels, students only (FULL=0).
#   Replaces the dev number "14.05 -> 9.39 deg (earlier pipeline)" (geo1004/kzst9, arm O: same frames, pseudo R -> train GT R, t/lwh kept).
#   Here: srcx/in2mav pseudo_tr1.npz with R -> nearest train GT box rotation (build_gt_pseudo.py; t_pnp, lwh, gate, frame set unchanged),
#   then the run_x.sh S6I recipe unchanged (seeds 0,1, init kzm_s{0,1}, gate rows, mirror loss 12, composites 0.5, 2,250 steps), AdaBN, mirror-view
#   readings, deploy at frozen omega 0.190 (refiners / c_k / r_k / s_bar / heights frozen), test scored ONCE.
#   Train GT is read only to build the oracle labels (diagnostic upper bound, not a method; no choice made with it).
# Output: B6_gtceil/arm/preds/gtrot_fused_test.pkl.score_real_test.json, B6_gtceil/gt_pseudo_report.json, logs/b6.log; done marker B6_gtceil/.done
#   GPU ~35 min (= clean3/nomir students-only measured wall time).
#   launch: bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B6_gtceil/b6_gtceil.sh
source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/cf_common.sh
B=$CF/B6_gtceil; A=$B/arm; mkdir -p $B/logs; LOG=$B/logs/b6.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $B/.done ] && { say "already done"; exit 0; }
wait_prereq $LOG
say "start B6 (GT-rotation oracle labels)"
setup_arm $A || { say "!! setup"; exit 2; }
[ -f $A/work/pseudo_gt.npz ] || $PY $B/build_gt_pseudo.py $XW/pseudo_tr1.npz $A/work/pseudo_gt.npz $B/gt_pseudo_report.json >> $LOG 2>&1 || { say "!! build gt pseudo"; exit 2; }
students_arm $A gtrot pseudo_gt 2>&1 | tee -a $LOG
[ -f $A/.done ] || { say "!! not done (re-launch resumes)"; exit 2; }
SUM=$($PY -c "
import json
a = json.load(open('$X/preds/pcmd_fused_test.pkl.score_real_test.json'))['all']; b = json.load(open('$A/preds/gtrot_fused_test.pkl.score_real_test.json'))['all']
print('PST AOE %.2f UDS %.2f -> GT-rotation labels AOE %.2f UDS %.2f' % (a['AOE'], a['UDS'], b['AOE'], b['UDS']))")
say "ALL DONE (test, 4,800 frames): $SUM"; touch $B/.done
