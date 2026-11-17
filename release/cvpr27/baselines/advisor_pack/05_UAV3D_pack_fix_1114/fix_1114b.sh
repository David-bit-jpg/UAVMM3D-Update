#!/usr/bin/env bash
# 11-14 补丁 b：把空着的显卡用起来（先运行过 fix_1114.sh 才能用本脚本）。
# 单个任务拆到多张卡上不可行（官方代码的多卡开关会把每张图的标签列表切乱；多卡还会改变批大小，和其他方法不可比），
# 所以空卡改跑 4 个能直接进论文的任务，正在跑的任务一律不动：
#   monocd 1 in kdc mav  : MonoCD + 干净版伪标签（MAV6D），约 6 h（我们本机同一个任务要 37 h）
#   monocd 1 in st  mav  : MonoCD 无标签自训练（MAV6D），从服务器已训好的 monocd_in_sim_s1 出发，补主表空格
#   gupnet 1 in kdc mav  : GUPNet + 干净版伪标签，第 2 个种子，约 6 h
#   gupnet 2 in kdc mav  : GUPNet + 干净版伪标签，第 3 个种子，约 6 h
# 用法：bash fix_1114b.sh [实验包根目录]      只检查不改动：DRYRUN=1 bash fix_1114b.sh
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; Q="$P/runs/queue"
say() { echo "== $1"; }
export PYTHONIOENCODING=utf-8

say "检查 fix_1114.sh 是否已运行过"
grep -q "st|kd|kdc)" scripts/run.sh && [ -f data/cmpfull/pcmd_teacher_pseudo_real_train_clean.pkl ] || { echo "!! 请先运行 bash fix_1114.sh"; exit 1; }
[ -f "$Q/monocd_in_sim_both_s1.done" ] || echo "   注意：monocd 种子 1 的 in sim 还没完成，monocd 1 in st mav 会等它（工人会先去领别的任务）"
echo "   通过"

say "加入 4 个任务（排在已有 kdc 任务之后、其余任务之前；已有的不重复加）"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; else
cp scripts/tasks.txt scripts/tasks.txt.bak_1114b
$PY - scripts/tasks.txt <<'PYEOF'
import sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
NEW = ['monocd    1 in  kdc mav    #   6 h（11-14b 空卡）',
       'monocd    1 in  st  mav    #   3 h（11-14b 空卡）',
       'gupnet    1 in  kdc mav    #   6.3 h（11-14b 空卡）',
       'gupnet    2 in  kdc mav    #   6.3 h（11-14b 空卡）']
def key(l):
    s = l.strip()
    if s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
have = {key(l) for l in lines}
add = [l for l in NEW if key(l) not in have]
idx = max([i for i, l in enumerate(lines) if key(l) and key(l)[3] == 'kdc'], default=-1) + 1
if idx == 0: idx = next((i for i, l in enumerate(lines) if key(l)), len(lines))
lines[idx:idx] = add
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(lines) + '\n')
print('   新加 %d 个' % len(add))
PYEOF
fi

say "在空闲显卡上开工"
IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{printf "%s ", $1}')
if [ -n "$DRYRUN" ]; then echo "   [只检查] 空闲显卡：${IDLE:-无}"
elif [ -n "$IDLE" ]; then GPUS="$IDLE" bash scripts/run_all.sh; else echo "   暂无空闲显卡：正在跑的工人跑完当前任务后会自动接新任务"; fi

sleep 20
say "各任务状态（只列 kdc 和新加的）"
for k in $(grep -v '^\s*#' scripts/tasks.txt | awk 'NF>=5 && ($4=="kdc" || $0 ~ /11-14b/){print $1"_"$3"_"$4"_"$5"_s"$2}'); do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"
  printf '   %-32s %s\n' "$k" "$s"
done
echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo; echo "完成。看进度：cat runs/STATUS.txt；GUPNet / MonoCD 约 6 小时后可以 bash scripts/collect_results.sh 把结果发回。"
