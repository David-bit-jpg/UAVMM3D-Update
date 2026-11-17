#!/bin/bash
# 显卡让路开关（我的显卡优先级最低）。用法：
#   bash gpu_ctl.sh pause [原因]   立刻停下本会话所有对比实验训练/预测，腾出显卡；排队脚本原地等待
#   bash gpu_ctl.sh resume         撤掉暂停，各任务从存档接着训
#   bash gpu_ctl.sh status
H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull; F=$H/PAUSE
case $1 in
  pause)  echo "${2:-manual} $(date '+%m-%d %T')" > $F
          powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $H/kill_ours.ps1)"
          echo "$(date '+%m-%d %T') 暂停（${2:-manual}）" >> $H/driver_gpu.log ;;
  resume) rm -f $F; echo "$(date '+%m-%d %T') 恢复" >> $H/driver_gpu.log ;;
  *)      [ -f $F ] && echo "暂停中：$(cat $F)" || echo "运行中"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader ;;
esac
