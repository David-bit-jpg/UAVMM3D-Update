#!/bin/bash
# [review1113 sensorfree E1geo, 11-15] waits for E1/mav_s{0,10,20}/.done (poll 10 min) and computes the DEPLOY-ONLY variant E1geo for each (CPU only,
# no GPU step): two brightness classes (PST rule, E1's own region head) + per-class height ratio r_k from box-corner PnP of the RGB-only KZ teachers.
# Scores test ONCE per run, paired sequence bootstrap vs the same-seed E1 and PST, pooled lines. Never touches running jobs or existing run dirs.
# Logs: logs/sensorfree.log ([E1geo] lines), E1geo/mav_s<k>/e1geo.log, E1geo/RESULT.md. Marker E1geo/mav_s<k>/.done | .failed | .skipped.
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GPW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto
SF=$GP/review1113/sensorfree; SFW=$GPW/review1113/sensorfree; EG=$SF/E1geo; EGW=$SFW/E1geo
PY=D:/Miniconda3/envs/city/python.exe; GLOG=$SF/logs/sensorfree.log; BOOT=$GPW/review1113/heads/heads_boot.py
export PYTHONUTF8=1 SX_SRC=in SX_TGT=mav SX_TEACH=none PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
say() { echo "$(date '+%m-%d %H:%M:%S') [E1geo] $1" >> $GLOG; echo "$(date '+%m-%d %H:%M:%S') $1" >> $EG/e1geo_driver.log; }
pstw() { case $1 in 0) echo $GPW/srcx/in2mav ;; 10) echo $GPW/review1113/sup3/run ;; 20) echo $GPW/review1113/seeds/mav_full_s20 ;; esac; }
pool() {
  local PE=() PP=() s
  for s in 0 10 20; do [ -f $EG/mav_s$s/.done ] || continue
    PE+=("$EGW/mav_s$s/preds/e1geo_fused_test.pkl" "$SFW/E1/mav_s$s/preds/e1_fused_test.pkl")
    PP+=("$EGW/mav_s$s/preds/e1geo_fused_test.pkl" "$(pstw $s)/preds/pcmd_fused_test.pkl"); done
  [ ${#PE[@]} -ge 2 ] || return 0
  local n=$(( ${#PE[@]} / 2 ))
  $PY $BOOT mav $EGW/pooled_vsE1.json "E1geo mav pooled vs E1 ($n seed pairs)" "${PE[@]}" > $EG/pool_vsE1.log 2>&1 && say "POOLED $(tail -1 $EG/pool_vsE1.log)" || say "!! pooled vs E1 failed"
  $PY $BOOT mav $EGW/pooled_vsPST.json "E1geo mav pooled vs PST ($n seed pairs)" "${PP[@]}" > $EG/pool_vsPST.log 2>&1 && say "POOLED $(tail -1 $EG/pool_vsPST.log)" || say "!! pooled vs PST failed"
}
proc() {   # $1 shift
  local s=$1 R=$SFW/E1/mav_s$1 O=$EGW/mav_s$1 o=$EG/mav_s$1
  mkdir -p $o; say "mav_s$s: E1 done -> deploy-only variant"
  if ! SX_SEEDSHIFT=$s $PY $EGW/e1geo.py $R $O --ref-prefix e1 > $o/e1geo.log 2>&1; then say "!! mav_s$s e1geo.py failed, see $o/e1geo.log"; touch $o/.failed; return 1; fi
  grep '^\[E1geo\]' $o/e1geo.log | sed 's/^\[E1geo\] //' | while read -r l; do say "mav_s$s $l"; done
  $PY $BOOT mav $O/boot_vsE1.json "E1geo mav_s$s vs E1 mav_s$s" $O/preds/e1geo_fused_test.pkl $R/preds/e1_fused_test.pkl > $o/boot_vsE1.log 2>&1 || { say "!! boot vs E1 failed"; touch $o/.failed; return 1; }
  $PY $BOOT mav $O/boot_vsPST.json "E1geo mav_s$s vs PST mav_s$s" $O/preds/e1geo_fused_test.pkl $(pstw $s)/preds/pcmd_fused_test.pkl > $o/boot_vsPST.log 2>&1 || { say "!! boot vs PST failed"; touch $o/.failed; return 1; }
  say "mav_s$s $(tail -1 $o/boot_vsE1.log)"; say "mav_s$s $(tail -1 $o/boot_vsPST.log)"
  touch $o/.done; pool; $PY $EGW/e1geo_report.py >> $EG/e1geo_driver.log 2>&1
}
say "driver started (pid $$): waiting for E1/mav_s{0,10,20}/.done, poll 10 min; CPU only"
while true; do
  left=0
  for s in 0 10 20; do o=$EG/mav_s$s
    [ -f $o/.done ] || [ -f $o/.failed ] || [ -f $o/.skipped ] && continue
    if [ -f $SF/E1/mav_s$s/.done ]; then proc $s
    elif [ -f $SF/E1/mav_s$s/.failed ] || [ -f $SF/E1/mav_s$s/.skipped ]; then mkdir -p $o; touch $o/.skipped; say "mav_s$s: E1 run failed/skipped -> no E1geo"; $PY $EGW/e1geo_report.py >> $EG/e1geo_driver.log 2>&1
    else left=1; fi
  done
  [ $left = 0 ] && break
  sleep 600
done
$PY $EGW/e1geo_report.py >> $EG/e1geo_driver.log 2>&1
say "driver done; see $EGW/RESULT.md"
