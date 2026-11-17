#!/usr/bin/env bash
# 11-13 重排任务队列（正文截稿前优先跑主表要用的对比实验）。
# 用法：bash prioritize_1113.sh <实验包根目录>     例：bash prioritize_1113.sh /userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack
# 原理：工人按"方法_源域_步骤_目标域_s种子"识别任务（与行号无关），每领一次都重读 tasks.txt。
#       所以重排不影响已完成（.done）或正在跑（.lock）的任务，不用停任何进程。
# 只改 scripts/tasks.txt，先备份为 tasks.txt.bak_1113。
set -e
P="$(cd "${1:?用法: bash prioritize_1113.sh <实验包根目录>}" && pwd)"
T="$P/scripts/tasks.txt"
[ -f "$T" ] || { echo "找不到 $T，请确认实验包根目录"; exit 1; }
cp "$T" "$T.bak_1113"
PYTHONIOENCODING=utf-8 ${PY:-python} - "$T" <<'PYEOF'
import sys, re
p = sys.argv[1]
lines = open(p, encoding='utf-8').read().splitlines()
def key(l):
    f = l.split('#')[0].split()
    return tuple(f[:5]) if len(f) >= 5 else None
# 优先顺序（方法 种子 源域 步骤 目标域）
PRIO = [
  # 1 主表 AntiUAV3D：对手方法用室外仿真训练（审稿人最集中的要求）
  'monocd 0 out sim both', 'gupnet 0 out sim both', 'monodgp 0 out sim both', 'monodetr 0 out sim both', 'monocop 0 out sim both',
  # 2 主表 AntiUAV3D：+ST / +PL（快的在前）
  'monocd 0 out kd anti', 'monocd 0 out st anti', 'gupnet 0 out kd anti', 'gupnet 0 out st anti', 'monodgp 0 out kd anti', 'monodgp 0 out st anti',
  # 3 主表 MAV6D：还缺的格子（截稿前来得及的）
  'gupnet 0 in st mav', 'gupnet 0 in kd mav', 'monodgp 0 in kd mav', 'monodgp 0 in st mav', 'monodetr 0 in sim both', 'monocop 0 in sim both',
  # 4 AntiUAV3D 上 MonoDETR / MonoCoP 的 +ST / +PL
  'monodetr 0 out kd anti', 'monocop 0 out kd anti', 'monodetr 0 out st anti', 'monocop 0 out st anti',
]
# 截稿前跑不完或不需要：加 "#SKIP1113 " 注释掉（要恢复就删掉这个前缀）
SKIP = ['monodetr 0 in kd mav', 'monocop 0 in kd mav', 'monodetr 0 in st mav', 'monocop 0 in st mav',
        'monocd 0 mix sim both', 'gupnet 0 mix sim both', 'monodgp 0 mix sim both', 'monodetr 0 mix sim both', 'monocop 0 mix sim both']
prio = [tuple(s.split()) for s in PRIO]; skip = {tuple(s.split()) for s in SKIP}
tasks, order = {}, []
for l in lines:
    if l.lstrip().startswith('#'):
        continue
    k = key(l)
    if k is not None:
        tasks[k] = l; order.append(k)
out = [l for l in lines if key(l) is None or l.lstrip().startswith('#')]
out = [l for l in out if not re.match(r'^\s*#\s*====', l)]  # 旧分档标题去掉，避免误导
out.append('# ==== 11-13 重排（prioritize_1113.sh）：1 AntiUAV3D 室外源 sim → 2 AntiUAV3D +ST/+PL → 3 MAV6D 还缺的 → 4 其余；#SKIP1113 = 截稿前不跑 ====')
used = set()
for k in prio:
    if k in tasks: out.append(tasks[k]); used.add(k)
for k in order:
    if k not in used and k not in skip: out.append(tasks[k]); used.add(k)
for k in order:
    if k in skip and k not in used: out.append('#SKIP1113 ' + tasks[k]); used.add(k)
missing = [ ' '.join(k) for k in prio if k not in tasks]
open(p, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('已重排 %d 个任务，注释掉 %d 个；优先表里没找到的行：%s' % (len(order), len([k for k in order if k in skip]), missing or '无'))
PYEOF
echo; echo "== 新的前 25 个任务（已完成的会被工人自动跳过）："
grep -v '^\s*#' "$T" | grep -v '^\s*$' | head -25
echo; echo "== 各任务当前状态："
for k in $(grep -v '^\s*#' "$T" | awk 'NF>=5{print $1"_"$3"_"$4"_"$5"_s"$2}'); do
  s="未开始"; [ -d "$P/runs/queue/$k.lock" ] && s="进行中"; [ -f "$P/runs/queue/$k.fail" ] && s="失败"; [ -f "$P/runs/queue/$k.done" ] && s="完成"
  printf '%-32s %s\n' "$k" "$s"
done
echo; echo "完成。不用重启工人：卡一空就会按新顺序领任务。要恢复原顺序：cp $T.bak_1113 $T"
