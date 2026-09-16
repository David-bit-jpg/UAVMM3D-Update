#!/bin/bash
# 用户 2026-09-15 20:2x：只要纯仿真训练 + 真实测试。等 S2/S3 训练进程（Windows PID 15456 / 3684）结束，
# 只评 MAV6D 真实 test（带 ADS），不评 MAV6D val、不评仿真 test。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/simfix
JS=$LOG/json
M=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm
alive() { powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { 'y' } else { 'n' }" | tr -d '\r'; }
while [ "$(alive 15456)" = "y" ] || [ "$(alive 3684)" = "y" ]; do sleep 60; done
echo "=== trainings finished $(date)" >> $LOG/driver.log
for arm in "S2 sim_indoor8_mz_nose" "S3 sim_indoor8_mz_nosescale"; do
  set -- $arm
  ck=$M/$2/$1/ckpt/best.pth
  if [ ! -f "$ck" ]; then echo "=== $1 missing best.pth" >> $LOG/driver.log; continue; fi
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $ck --tag Z_$1 --split test --workers 0 \
      --json $JS/Z_$1_test.json --ads > $LOG/Z_$1_test.log 2>&1
  echo "=== Z_$1 test eval exit $? $(date)" >> $LOG/driver.log
done
$PY report_simfix.py > $LOG/report.txt 2>&1
echo "=== EVAL DONE $(date)" >> $LOG/driver.log
