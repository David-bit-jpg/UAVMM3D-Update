#!/bin/bash
# [review1113/cleanfix B4] Tab. orient rows on the CLEAN MAV6D chain (srcx/in2mav), students only (FULL=0), test scored once per arm.
#   arm nocomp : students trained WITHOUT the composite (simulated airframe pasted on a real single-frame background) ->  ou_train_x --paste 0
#                (deployed: --paste 0.5 default; with --paste 0 the source dataset gets xbg=None, i.e. plain simulated crops only).
#   arm nosym  : target keypoint loss WITHOUT the symmetric min over the 4 quadrotor-equivalent targets -> ou_train_x --no-sym
#                (existing flag; c4_kp_term then gets only the pseudo label's own target, g[:, :, :16]). Source loss / mirror loss unchanged.
#   Everything else = run_x.sh S6I recipe: seeds 0, 1 concurrently, init = same-seed IR+LiDAR member (kzm_s{0,1}), gated pseudo-rotations
#   pseudo_tr1 (copied from the deployed run), mirror loss 12, 2,250 steps, bs 8 sim + 4 real; AdaBN on all train crops; mirror-view
#   readings; deploy with frozen omega 0.190 (refiners, c_k, r_k, s_bar, height readings frozen = deployed; b_k re-estimated from the
#   unlabeled calibration images by the deploy rule). Only rotations + corner-PnP distance of the fusion change.
# Output: review1113/cleanfix/B4_orient/{nocomp,nosym}/preds/<arm>_fused_test.pkl.score_real_test.json ; summary line in logs/b4.log;
#   global done marker B4_orient/.done.  GPU ~35 min per arm (clean3/nomir measured 35 min wall), ~70 min total + queueing.
#   launch: bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/b4_orient.sh   (ARMS="nocomp nosym" default)
source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/cf_common.sh
B=$CF/B4_orient; mkdir -p $B/logs; LOG=$B/logs/b4.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $B/.done ] && { say "already done"; exit 0; }
wait_prereq $LOG
ARMS=${ARMS:-"nocomp nosym"}
for arm in $ARMS; do
  A=$B/$arm
  case $arm in nocomp) EX="--paste 0";; nosym) EX="--no-sym";; *) say "bad arm $arm"; exit 2;; esac
  say "arm $arm ($EX) start"
  setup_arm $A || { say "!! setup $arm"; exit 2; }
  [ -f $A/work/pseudo_tr1.npz ] || cp $XW/pseudo_tr1.npz $A/work/ || exit 2
  students_arm $A $arm pseudo_tr1 $EX 2>&1 | tee -a $LOG
  [ -f $A/.done ] || { say "!! arm $arm not done (re-launch resumes)"; exit 2; }
  grep -h "args" $A/ou/runs/${arm}_s0/train_*.log | head -1 | cut -c1-400 >> $LOG
done
SUM=$($PY -c "
import json
r = {'PST': json.load(open('$X/preds/pcmd_fused_test.pkl.score_real_test.json'))['all']}
for a in '$ARMS'.split():
    r[a] = json.load(open('$B/%s/preds/%s_fused_test.pkl.score_real_test.json' % (a, a)))['all']
print(' | '.join('%s AOE %.2f UDS %.2f AP %.2f' % (k, v['AOE'], v['UDS'], v['AP']) for k, v in r.items()))")
say "ALL DONE (test, n = 4,800 test frames): $SUM"; touch $B/.done
