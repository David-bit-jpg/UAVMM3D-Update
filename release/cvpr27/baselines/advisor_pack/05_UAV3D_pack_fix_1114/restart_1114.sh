#!/usr/bin/env bash
# 11-14：服务器卡住时用。停掉本实验包的所有任务，再从各自的存档接着训。
# 已完成的任务（.done）不会重跑；被停的任务会从最近一次存档继续：
#   MonoDGP / MonoDETR / MonoCoP / MonoCD 每轮存一次，GUPNet 每 10 轮存一次。
# 用法：bash restart_1114.sh [实验包根目录]（和 fix_1114.sh 放在同一个文件夹）
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$P/scripts/run.sh" ] || { echo "!! $P 不是实验包根目录"; exit 1; }
echo "== 1. 停掉本实验包的任务（只停你自己账号下、命令行里带这个实验包路径的进程）"
pkill -u "$USER" -f "runs/queue" ; pkill -u "$USER" -f "scripts/run.sh" ; pkill -u "$USER" -f "$P"
sleep 15; pkill -9 -u "$USER" -f "$P" 2>/dev/null; sleep 5
left=$(pgrep -u "$USER" -f "$P" | wc -l); echo "   剩余进程：$left"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "== 2. 清掉被停任务的'进行中'标记（.done 完成标记保留）"
n=$(ls -d "$P"/runs/queue/*.lock 2>/dev/null | wc -l); rm -rf "$P"/runs/queue/*.lock; echo "   清掉 $n 个"
echo "== 3. 修复 + 排序 + 在空闲显卡上重新开工"
bash "$HERE/fix_1114.sh" "$P"
