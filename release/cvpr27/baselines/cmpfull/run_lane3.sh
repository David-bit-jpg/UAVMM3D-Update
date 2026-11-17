#!/bin/bash
# 训练队列 v3（10-30）：认暂停（PAUSE 文件在就原地等，不算失败）；以"最终存档存在"判完成；断了就从存档接着训。
# 用法：bash run_lane3.sh <车道> <job> ...；job = gupnet_s0 / monocd_s0 / monodgp_s0 ...
H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull; R=D:/cmpfull_runs; LANE=$1; shift
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home DLA34_PTH=E:/torch_home/hub/checkpoints/dla34-ba72cf86.pth GUPNET_DLA34_WEIGHTS=E:/torch_home/hub/checkpoints/dla34-ba72cf86.pth
ok() { local m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ ! -f $H/PAUSE ] && [ "$m" -lt 14000 ]; }
final() { case $1 in gupnet) echo $R/gupnet_s$2/checkpoints/checkpoint_final.pth ;; monocd) echo $R/monocd_s$2/model_final.pth ;; monodgp) echo $R/monodgp_s$2/model_epoch_48.pth ;; esac; }
running() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'cmpfull_runs/$1 ' -or \$_.CommandLine -match 'cmpfull_runs/$1\$' }).Count" | tr -d '\r '; }
L=$H/driver_$LANE.log
for job in "$@"; do
  m=${job%_s*}; s=${job##*_s}; fail=0
  while [ ! -f "$(final $m $s)" ]; do
    if [ "$(running $job)" -gt 0 ]; then sleep 120; continue; fi        # 上一版队列启动的还在跑：等它
    while ! { ok && sleep 60 && ok; }; do sleep 120; done
    RES=""
    case $m in
      gupnet)  c=$(ls -v $R/gupnet_s$s/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | tail -1); [ -n "$c" ] && RES="--resume $c" ;;
      monocd)  [ -f $R/monocd_s$s/model_checkpoint.pth ] && RES="--resume" ;;
      monodgp) [ -f $R/monodgp_s$s/checkpoint.pth ] && RES="--resume" ;;
    esac
    echo "$(date '+%m-%d %T') 开始 $job $RES" >> $L
    case $m in
      gupnet)  GPU_FRAC=0.3  bash $H/gupnet/train.sh  --labels D:/cmpfull/sim_train/labels.pkl --out $R/gupnet_s$s  --seed $s --epochs 140 --batch-size 8 $RES ;;
      monocd)  GPU_FRAC=0.4  bash $H/monocd/train.sh  --labels D:/cmpfull/sim_train/labels.pkl --out $R/monocd_s$s  --seed $s --epochs 19 --batch 4 $RES ;;
      monodgp) GPU_FRAC=0.25 bash $H/monodgp/train.sh --labels D:/cmpfull/sim_train/labels.pkl --out $R/monodgp_s$s --seed $s --epochs 48 --batch-size 2 $RES ;;
    esac >> $H/train_$job.log 2>&1
    rc=$?
    if [ -f "$(final $m $s)" ]; then break; fi
    if [ -f $H/PAUSE ]; then echo "$(date '+%m-%d %T') 暂停打断 $job（退出码 $rc），恢复后接着训" >> $L; continue; fi
    fail=$((fail+1)); echo "$(date '+%m-%d %T') $job 异常退出 $rc（第 $fail 次）" >> $L
    [ $fail -ge 2 ] && break
  done
  if [ -f "$(final $m $s)" ]; then echo "$(date '+%m-%d %T') 结束 $job 退出码 0" >> $L; else echo "$(date '+%m-%d %T') 结束 $job 退出码 1（放弃）" >> $L; fi
done
