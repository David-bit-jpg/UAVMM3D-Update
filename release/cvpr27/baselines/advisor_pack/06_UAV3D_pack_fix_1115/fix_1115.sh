#!/usr/bin/env bash
# 11-15 补丁：服务器空闲，把能在本包里跑的补种子任务排进去（先 fix_1114 / 1114b 已运行过）。正在跑的不动。
# 顺序：1 正文能用的 MAV6D kdc（MonoCD、MonoDGP）→ 2 AntiUAV3D kd 快的三个 → 3 室外 sim→st 补种子 → 4 最慢的 MonoDETR/MonoCoP（约 2 天，补充材料）。
# 用法：bash fix_1115.sh [实验包根目录]      只检查不改动：DRYRUN=1 bash fix_1115.sh
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录"; exit 1; }
cd "$P"; PY=${PY:-python}; Q="$P/runs/queue"; export PYTHONIOENCODING=utf-8
grep -q "st|kd|kdc)" scripts/run.sh || { echo "!! 请先运行 fix_1114.sh"; exit 1; }
echo "== 加入任务（已有的不重复加，排在队首）"
if [ -z "$DRYRUN" ]; then cp scripts/tasks.txt scripts/tasks.txt.bak_1115
$PY - scripts/tasks.txt <<'PYEOF'
import sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
NEW = []
# 1 正文能用（快）：MAV6D kdc 的 MonoCD、MonoDGP 补种子
for m, s in [('monocd',0),('monocd',2),('monodgp',1),('monodgp',2)]:
    NEW.append('%-9s %d in  kdc mav    # 11-15 优先1 正文' % (m, s))
# 2 补充材料、快的：AntiUAV3D kd 补种子（GUPNet / MonoCD / MonoDGP）
for m in ['gupnet','monocd','monodgp']:
    for s in (1, 2): NEW.append('%-9s %d out kd  anti   # 11-15 优先2' % (m, s))
# 3 室外 sim 补种子 -> 同种子 st
for m in ['monocd','gupnet','monodgp']:
    for s in (1, 2): NEW.append('%-9s %d out sim both   # 11-15 优先3' % (m, s))
for m in ['monocd','gupnet','monodgp']:
    for s in (1, 2): NEW.append('%-9s %d out st  anti   # 11-15 优先3（等同种子 sim）' % (m, s))
# 4 最慢（约 2 天）：MonoDETR / MonoCoP 的 kdc 与 kd 补种子
for m in ['monodetr','monocop']:
    for s in (1, 2): NEW.append('%-9s %d in  kdc mav    # 11-15 优先4 慢' % (m, s))
for m in ['monodetr','monocop']:
    for s in (1, 2): NEW.append('%-9s %d out kd  anti   # 11-15 优先4 慢' % (m, s))
def key(l):
    s = l.strip()
    if s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
have = {key(l) for l in lines}
add = [l for l in NEW if key(l) not in have]
i = next((k for k, l in enumerate(lines) if key(l)), len(lines))
lines[i:i] = add
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(lines) + '\n'); print('   新加 %d 个' % len(add))
PYEOF
fi
echo "== 在空闲显卡上开工"
IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{printf "%s ", $1}')
if [ -n "$DRYRUN" ]; then echo "   [只检查] 空闲显卡：${IDLE:-无}"; elif [ -n "$IDLE" ]; then GPUS="$IDLE" bash scripts/run_all.sh; else echo "   暂无空闲显卡"; fi
sleep 20; echo "== 新任务状态"
for k in $(grep '11-15' scripts/tasks.txt | grep -v '^\s*#' | awk 'NF>=5{print $1"_"$3"_"$4"_"$5"_s"$2}'); do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"; printf '   %-32s %s\n' "$k" "$s"; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "完成。GUPNet / MonoCD 约 6–8 小时、MonoDGP 约 1 天、MonoDETR / MonoCoP 约 2 天出结果；随时 bash scripts/collect_results.sh 发回。"
