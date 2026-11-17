#!/bin/bash
# [xdomain_1117 B] cross-source full PST chain: outdoor SimDrone3D source -> MAV6D target (out2mav), the unchanged srcx chain.
# srcx/run_x.sh skips every finished step, so a stopped run (memory guard, host-RAM error, reboot) resumes by re-running it.
# Up to 6 attempts, 5 min apart. Test labels are read only by run_x.sh's final score_x.py call (once).
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; XD=$X/review1113/xdomain_1117; LOG=$X/srcx/out2mav/logs/run_x.log
say() { echo "$(date '+%m-%d %H:%M:%S') [B] $1" >> $XD/logs/run.log; }
for t in 1 2 3 4 5 6; do
  say "attempt $t: bash srcx/run_x.sh out mav"
  SX_TEACH=full bash $X/srcx/run_x.sh out mav >> $XD/logs/B_run_x.stdout 2>&1
  if grep -q "ALL DONE" $LOG 2>/dev/null; then say "ALL DONE"; break; fi
  say "attempt $t ended without ALL DONE: $(grep '!!' $LOG | tail -1)"; sleep 300
done
J=$X/srcx/out2mav/preds/pcmd_fused_test.pkl.score_real_test.json; J0=$X/srcx/out2mav/preds/srconly_test.pkl.score_real_test.json
[ -f $J ] && D:/Miniconda3/envs/city/python.exe -c "
import json
for n, f in (('srconly (3-seed ens + flip + s_bar)', '$J0'), ('PST full chain', '$J')):
    a = json.load(open(f))['all']; print('B out2mav %-36s AP %.2f ATE %.3f ASE %.3f AOE %.2f mDR %.2f UDS %.2f' % (n, a['AP'], a['ATE'], a['ASE'], a['AOE'], a['mDR'], a['UDS']))
" >> $XD/logs/run.log 2>&1
