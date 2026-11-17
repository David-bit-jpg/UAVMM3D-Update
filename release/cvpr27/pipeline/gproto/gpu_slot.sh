#!/bin/bash
# 统一显卡排队（11-11 源域对照实验，所有 agent 共用）：用法  bash gpu_slot.sh <heavy|light> <命令...>
# 11-11 srcx agent: count() wrapped in @(); regex also counts *_x.py copies (srcx).
# 11-11 21:3x 优先级：run_x.sh（源域对照主链）的任务 = 优先（命令含 srcx/lib/ 或 st_*2anti/st_*2mav 配置）；
#   11-12 起：不再整体挡住非优先任务（主链已无多进程训练），优先任务门槛更宽（5 个任务 / 22 GB），非优先 4 个 / 18 GB；优先任务显存门槛放宽到 22 GB。
#   heavy = 带 spawn 数据加载进程的训练；heavy < 2 对所有人都硬性保留（Windows 上 ≥3 个会锁死）。
KIND=$1; shift
LOCK=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/.gpu_slot.lock
count() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' -and \$_.CommandLine -notmatch 'multiprocessing' }).Count" | tr -d '\r '; }
runx() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'run_x\.sh' }).Count" | tr -d '\r '; }
PRIO=0; echo "$*" | grep -q -E "srcx/lib/|st_[a-z]+2(anti|mav)" && PRIO=1
# 11-16 正文截稿前的主表任务（ourdet_baseproto）= 紧急：不受任务数上限限制，只看显存 < 20 GB 和 heavy < 2。
URG=0; echo "$*" | grep -q -E "ourdet_baseproto|sensorfree/E4/|heads/H[456]/antifs|k249" && URG=1
while true; do
  if mkdir $LOCK 2>/dev/null; then
    h=$(count 'train\.py|train_mm\.py|train_kz_lid\.py|train_mmk\.py')
    a=$(count 'train\.py|train_mm\.py|train_kz_lid\.py|train_mmk\.py|ou_train|refiner_(out|x).py|det_infer|ori_(out|x).py|wsim_(out|x).py|predict\.py')
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    ok=1
    if [ $PRIO = 1 ]; then [ "${a:-9}" -ge 5 ] && ok=0; [ "${m:-99999}" -ge 22000 ] && ok=0
    else [ "${a:-9}" -ge 4 ] && ok=0; [ "${m:-99999}" -ge 18000 ] && ok=0; fi
    [ "$KIND" = heavy ] && [ "${h:-9}" -ge 2 ] && ok=0
    if [ $URG = 1 ]; then ok=1; [ "${m:-99999}" -ge 20000 ] && ok=0; [ "$KIND" = heavy ] && [ "${h:-9}" -ge 2 ] && ok=0; fi
    # 11-17 top-priority hold: while gproto/.prio_hold exists, only urgent (ourdet_baseproto) jobs may start
    [ -f /e/Open3DUAVDet/output/camnorm/geo1004/gproto/.prio_hold ] && [ $URG = 0 ] && ok=0
    # 11-16 内存闸门（用户：不许再因内存爆炸重跑）：任何新任务（含紧急）都要求 空闲物理内存 >= 8 GB 且 可提交内存余量 >= 16 GB。
    if [ $ok = 1 ]; then read fp fc <<< $(powershell.exe -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; "{0} {1}" -f [int]($o.FreePhysicalMemory/1024), [int]($o.FreeVirtualMemory/1024)' | tr -dc '0-9 ')
      { [ "${fp:-0}" -lt 8192 ] || [ "${fc:-0}" -lt 16384 ]; } && ok=0; fi
    # 11-16 让路模式：用户有作业在用显卡时 touch gproto/.gpu_yield；新任务只在 heavy<1、总任务<3、显存<12 GB 时才开（在跑的不动）。
    if [ -f /e/Open3DUAVDet/output/camnorm/geo1004/gproto/.gpu_yield ]; then
      [ "$KIND" = heavy ] && [ "${h:-9}" -ge 1 ] && ok=0; [ "${a:-9}" -ge 3 ] && ok=0; [ "${m:-99999}" -ge 12000 ] && ok=0; fi
    if [ $ok = 1 ]; then
      "$@" & pid=$!; sleep 30; rmdir $LOCK; wait $pid; exit $?
    fi
    rmdir $LOCK
  fi
  sleep 30
done
