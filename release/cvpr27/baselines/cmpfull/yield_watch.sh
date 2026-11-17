#!/bin/bash
# 自动让路：每分钟查一次。显卡上出现别人的 Python 进程，或显存 > 21 GB → 自动暂停；
# 只撤销自己设的自动暂停（手动暂停要 gpu_ctl.sh resume），条件是别人连续 10 分钟不在且显存 < 8 GB。
H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull; F=$H/PAUSE; clear=0
while true; do
  n=$(powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $H/others_on_gpu.ps1)" | tr -d '\r ')
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ ! -f $F ]; then
    if [ "${n:-0}" -gt 0 ] || [ "$m" -gt 21000 ]; then bash $H/gpu_ctl.sh pause "auto:别人的进程=$n,显存=${m}MiB"; clear=0; fi
  elif grep -q '^auto' $F; then
    if [ "${n:-0}" -eq 0 ] && [ "$m" -lt 8000 ]; then clear=$((clear+1)); else clear=0; fi
    [ $clear -ge 10 ] && { bash $H/gpu_ctl.sh resume; clear=0; }
  fi
  sleep 60
done
