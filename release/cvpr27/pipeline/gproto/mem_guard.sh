#!/bin/bash
# 11-16 内存守护（用户：不许再因内存爆炸重跑；所有计算资源都归本会话调度）。每 30 s：
#  1) 清理孤儿数据加载子进程（命令行含 spawn_main(parent_pid=X) 且 X 已不存在）——它们只占内存，不做任何事。
#  2) 记录空闲物理内存 / 可提交内存余量（每 10 分钟一行，异常时每次一行）。
#  3) 紧急：可提交余量 < 8 GB 或 空闲物理 < 1 GB 时，按优先级从低到高停掉一个补充材料任务（连同其子进程），
#     保住正文主表任务（命令行含 ourdet_baseproto 的永远不停）。被停的任务都能从存档续跑；写入 MEM_ALERT.txt 待巡检时重排。
#     优先级（先停）：lidraw → heads（单项消融补跑）→ sensorfree → stage1seeds 下游（runs/）→ stage1seeds 第一阶段训练（S1X_）。
#  4) 预警：余量 < 14 GB 时写 MEM_ALERT.txt（不停任务），gpu_slot.sh 的内存闸门会挡住新任务。
# 只按核对过命令行的 PID 停；只停本项目 gproto/ 下的任务。
G=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; LOG=$G/mem_guard.log; AL=$G/MEM_ALERT.txt
say() { echo "$(date '+%m-%d %H:%M:%S') $1" >> $LOG; }
alert() { echo "$(date '+%m-%d %H:%M:%S') $1" >> $AL; say "ALERT $1"; }
PS() { powershell.exe -NoProfile -Command "$1" | tr -d '\r'; }
say "mem_guard start (pid $$)"
n=0
while true; do
  # 1) orphans
  PS '$all = @{}; Get-CimInstance Win32_Process | ForEach-Object { $all[[int]$_.ProcessId] = 1 }; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "spawn_main\(parent_pid=(\d+)" } | ForEach-Object { $pp = [int]([regex]::Match($_.CommandLine, "parent_pid=(\d+)").Groups[1].Value); if (-not $all.ContainsKey($pp)) { Stop-Process -Id $_.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue; "orphan $($_.ProcessId) (parent $pp gone) stopped" } }' | while read -r l; do [ -n "$l" ] && say "$l"; done
  # 2) memory
  read fp fc <<< $(PS '$o=Get-CimInstance Win32_OperatingSystem; "{0} {1}" -f [int]($o.FreePhysicalMemory/1024), [int]($o.FreeVirtualMemory/1024)' | tr -dc '0-9 ')
  n=$((n + 1)); [ $((n % 20)) = 1 ] && say "free phys ${fp} MB, free commit ${fc} MB"
  # 3) emergency
  if [ "${fc:-99999}" -lt 8192 ] || [ "${fp:-99999}" -lt 1024 ]; then
    for pat in 'review1113[/\\]lidraw' 'review1113[/\\]heads' 'review1113[/\\]sensorfree' 'stage1seeds[/\\]runs' 'S1X_'; do
      v=$(PS "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$pat' -and \$_.CommandLine -notmatch 'ourdet_baseproto|sensorfree[/\\]E4|k249|multiprocessing-fork' } | Sort-Object PrivatePageCount -Descending | Select-Object -First 1 | ForEach-Object { \"\$(\$_.ProcessId)\" }" | tr -dc '0-9')
      if [ -n "$v" ]; then
        PS "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'parent_pid=$v[^0-9]' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -Confirm:\$false -ErrorAction SilentlyContinue }; Stop-Process -Id $v -Force -Confirm:\$false -ErrorAction SilentlyContinue" > /dev/null
        alert "EMERGENCY free commit ${fc} MB / phys ${fp} MB -> stopped supplement job pid $v (pattern $pat); it resumes from its checkpoint when re-queued"
        sleep 60; break
      fi
    done
  elif [ "${fc:-99999}" -lt 14336 ]; then
    [ $((n % 4)) = 1 ] && alert "WARNING free commit ${fc} MB / phys ${fp} MB (new jobs held by gpu_slot gate)"
  fi
  sleep 30
done
