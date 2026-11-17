#!/bin/bash
# [11-17] pooled E4 lines (= run_sf.sh pool() for E4) but through heads_boot_tol.py (tie-order tolerance). usage: bash pool_e4_tol.sh mav|antifs
T=$1; GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; SF=$GP/review1113/sensorfree; PY=D:/Miniconda3/envs/city/python.exe; export PYTHONUTF8=1
wp() { echo "$1" | sed -E 's|^/([a-z])/|\U\1:/|'; }
eval "$(sed -n '/^pst_of()/,/^none_of()/p' $SF/run_sf.sh; sed -n '/^none_of()/{n;p}' $SF/run_sf.sh)"
PP=(); PN=(); for s in 0 10 20; do d=$SF/E4/${T}_s$s; [ -f $d/.done ] || continue
  PP+=("$(wp $d)/preds/e4_fused_test.pkl" "$(wp $(pst_of $T $s))/preds/pcmd_fused_test.pkl"); PN+=("$(wp $d)/preds/e4_fused_test.pkl" "$(wp $(none_of $T $s))/preds/pcmd_fused_test.pkl"); done
n=$(( ${#PP[@]} / 2 ))
$PY $SF/heads_boot_tol.py $T $(wp $SF)/E4/pooled_${T}_vsPST.json "E4 $T pooled vs PST ($n seed pairs)" "${PP[@]}" > $SF/E4/pool_${T}_vsPST.log 2>&1 && echo "$(date '+%m-%d %H:%M:%S') [POOLED] $(tail -1 $SF/E4/pool_${T}_vsPST.log)" | tee -a $SF/logs/sensorfree.log
$PY $SF/heads_boot_tol.py $T $(wp $SF)/E4/pooled_${T}_vsNONE.json "E4 $T pooled vs RGB-only none ($n seed pairs)" "${PN[@]}" > $SF/E4/pool_${T}_vsNONE.log 2>&1 && echo "$(date '+%m-%d %H:%M:%S') [POOLED] $(tail -1 $SF/E4/pool_${T}_vsNONE.log)" | tee -a $SF/logs/sensorfree.log
