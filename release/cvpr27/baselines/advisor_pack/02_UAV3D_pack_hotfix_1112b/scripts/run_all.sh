#!/usr/bin/env bash
# 一键全跑：每张卡一个工人，按 scripts/tasks.txt 从上往下领任务，卡一空就领下一个。后台运行，关掉终端也不停。
# 用法：GPUS="0 1 2 3 4 5 6 7" bash scripts/run_all.sh        （缺省用全部 8 张卡）
# 看进度：cat runs/STATUS.txt      结果随时打包：bash scripts/collect_results.sh
cd "$(dirname "$0")/.."
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
mkdir -p runs/queue runs/logs
# 11-12：每次启动先把代码里的路径占位符换成本包位置（安装脚本中途失败时这一步会被跳过；重复运行安全）
${PY:-python} setup/fix_paths.py || { echo "setup/fix_paths.py 失败，先看报错"; exit 1; }
PH="@@"; PH="${PH}ROOT${PH}"   # placeholder spelled in two parts so fix_paths.py does not rewrite this line too
if grep -rl "$PH" Open3DUAVDet/output/camnorm/geo1004/cmpfull/*/predict.py pretrained 2>/dev/null | grep -q .; then echo "仍有路径占位符没替换，停止"; exit 1; fi
[ -f Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/runs/monocd_uav.yaml ] || { echo "缺 monocd/runs/monocd_uav.yaml（11-12 补丁 b 里有）"; exit 1; }
worker() {
  g=$1
  while true; do
    task=""; pending=""
    while read -r a b c d e rest; do
      [ -z "$a" ] || [ "${a:0:1}" = "#" ] || [ -z "$e" ] && continue   # 11-12：缺字段的行（空白行等）跳过
      id="${a}_${c}_${d}_${e}_s${b}"
      [ -f "runs/queue/$id.done" ] && continue
      # st 要用源域模型打伪标签：源域模型（本包训的，或随包附带的 pretrained/）还没好就先跳过，卡去做别的任务（源域任务失败了就不等）
      if { [ "$d" = st ] || { [ "$d" = sim ] && [ "$e" = antifu ]; }; } && [ ! -f "runs/queue/${a}_${c}_sim_both_s${b}.fail" ] && [ ! -f "runs/queue/${a}_${c}_sim_both_s${b}.done" ] && [ ! -d "pretrained/${a}_${c}_sim_s${b}" ]; then pending=1; continue; fi
      if mkdir "runs/queue/$id.lock" 2>/dev/null; then task="$a $b $c $d $e"; break; fi
    done < scripts/tasks.txt
    if [ -z "$task" ]; then
      [ -n "$pending" ] && { sleep 600; continue; }
      echo "[$(date '+%F %T')] 卡 $g：队列空了，退出" >> runs/STATUS.txt; return
    fi
    echo "[$(date '+%F %T')] 卡 $g 开始：$task" >> runs/STATUS.txt
    if CUDA_VISIBLE_DEVICES=$g bash scripts/run.sh $task > "runs/logs/$id.out" 2>&1; then
      touch "runs/queue/$id.done"; echo "[$(date '+%F %T')] 卡 $g 完成：$task" >> runs/STATUS.txt
    else
      touch "runs/queue/$id.fail"
      echo "[$(date '+%F %T')] 卡 $g 失败：$task（看 runs/logs/$id.out；修好后删掉 runs/queue/$id.lock 和 $id.fail 再运行本脚本）" >> runs/STATUS.txt
    fi
  done
}
for g in $GPUS; do nohup bash -c "$(declare -f worker); worker $g" > /dev/null 2>&1 & done
echo "已在后台启动（卡：$GPUS）。看进度：cat runs/STATUS.txt"
