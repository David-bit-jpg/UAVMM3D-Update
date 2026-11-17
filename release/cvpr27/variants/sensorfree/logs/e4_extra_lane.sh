#!/bin/bash
# 11-17 extra E4 seeds (launched by subagent): runs the given E4 runs sequentially through run_sf.sh with LATEST_START=11171700
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; export LATEST_START=11171700
for r in "$@"; do t=${r%_*}; s=${r#*_}; bash $SF/run_sf.sh E4 $t $s > $SF/logs/E4_${t}_s${s}.stdout 2>&1; echo "$(date '+%m-%d %H:%M:%S') [e4_extra] E4 $t s$s exit $?" >> $SF/logs/sensorfree.log; done
