#!/usr/bin/env bash
# 11-17 补丁 b（补充材料，不急，不暂停任何任务）：MAV6D 上对手方法的 +PST 改用"只取通过一致性检查的帧"的伪标签表（2,126 帧）。
#   和 RealDrone3D 现在的 249 帧表同一规则（都 = 有伪框 ∩ 通过检查），两个目标域就统一了。
#   新步骤 kdg = kdc 的伪标签表只保留通过检查的帧（同框、同朝向，只少帧） → runs/<方法>_in_kdg_mav_s<种子>/
#   伪标签只含 seq / frame / center / lwh / euler_xyz，不含任何真实标签。旧结果全部保留。
# 先运行过 fix_1117.sh 才能用本脚本。新任务排在 11-17 那批（kdu）之后、其余旧任务之前；正在跑的任务一律不动。
# 用法：bash fix_1117b.sh [实验包根目录]      只检查不改动：DRYRUN=1 bash fix_1117b.sh
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; CF="$P/Open3DUAVDet/output/camnorm/geo1004/cmpfull"; Q="$P/runs/queue"; D="$P/data/cmpfull"
say() { echo "== $1"; }
[ -n "$DRYRUN" ] && echo "（只检查模式：不改任何文件）"
export PYTHONIOENCODING=utf-8
grep -q "\[UAV 1117\]" scripts/run.sh || { echo "!! 请先运行 fix_1117.sh"; exit 1; }
NEWTASKS="gupnet 0 in kdg mav
gupnet 1 in kdg mav
gupnet 2 in kdg mav
monocd 0 in kdg mav
monodgp 0 in kdg mav
monodetr 0 in kdg mav
monocop 0 in kdg mav
monocd 1 in kdg mav
monocd 2 in kdg mav
monodgp 1 in kdg mav
monodgp 2 in kdg mav
monodetr 1 in kdg mav
monodetr 2 in kdg mav
monocop 1 in kdg mav
monocop 2 in kdg mav"
NEWIDS=$(echo "$NEWTASKS" | awk '{print $1"_"$3"_"$4"_"$5"_s"$2}')

say "A. MAV6D 只取通过检查的帧的伪标签表（2,126 帧）"
NEW="$HERE/data/pcmd_teacher_pseudo_real_train_clean_gated.pkl"
[ -f "$NEW" ] || { echo "!! 补丁包里缺 data/pcmd_teacher_pseudo_real_train_clean_gated.pkl"; exit 1; }
$PY - "$NEW" "$D/real_train/labels.pkl" "$D/pcmd_teacher_pseudo_real_train_clean.pkl" <<'PYEOF'
import os, pickle, sys
import numpy as np
t = pickle.load(open(sys.argv[1], 'rb')); lab = {(l['seq'], l['frame']) for l in pickle.load(open(sys.argv[2], 'rb'))}
assert all(set(r) == {'seq', 'frame', 'center', 'lwh', 'euler_xyz'} for r in t), '字段不对'
assert all((r['seq'], r['frame']) in lab for r in t), '有帧不在 real_train/labels.pkl 里'
assert len(t) == 2126, '帧数 %d，应为 2126' % len(t)
print('   %d 帧，全部在 real_train 里' % len(t))
if os.path.exists(sys.argv[3]):
    c = {(r['seq'], r['frame']): r for r in pickle.load(open(sys.argv[3], 'rb'))}
    ok = all((r['seq'], r['frame']) in c and np.allclose(r['center'], c[(r['seq'], r['frame'])]['center']) for r in t)
    print('   核对：是服务器上 %d 帧干净版表的子集（同帧同框）：%s' % (len(c), '是' if ok else '【不是】'))
PYEOF
[ -n "$DRYRUN" ] || cp "$NEW" "$D/pcmd_teacher_pseudo_real_train_clean_gated.pkl"

say "B. scripts/run.sh：新增步骤 kdg（只用于 mav）；score.py 加显示名"
F="$P/scripts/run.sh"
if grep -q "\[UAV 1117b\]" "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1117b"
  $PY - "$F" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s, n = re.subn(r'(?m)^(  st\|kd[a-z|]*kdeu)\)$', r'\1|kdg)', s)
assert n == 2, 'run.sh 里 "  st|...|kdeu)" 的行数是 %d，不是 2，没改' % n
lines = s.split('\n'); hit = [i for i, l in enumerate(lines) if 'build_pcmd_labels.py" "$DATA" "$O/pseudo.pkl" $TGT' in l]
assert len(hit) == 1, 'build_pcmd_labels 那一行有 %d 处，没改' % len(hit)
l = lines[hit[0]]; k = l.find('   #'); lines[hit[0]] = l[:k] + '$([ $step = kdg ] && echo clean_gated)' + l[k:] + '   # [UAV 1117b] kdg = 干净版只取通过检查的帧'
s = '\n'.join(lines)
old = 'case $step in kdu|kdnu|kdeu)'
assert s.count(old) == 1, '找不到 fix_1117 加的检查行，没改'
s = s.replace(old, '[ $step = kdg ] && { [ "$tgt" = mav ] || { echo "kdg 只用于 mav"; exit 1; }; [ "$m" != gupnet ] && WORKERS=${WORKERS:-6}; true; }   # [UAV 1117b]\n' + old, 1)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s); print('   已改')
PYEOF
else echo "   [只检查] 会改 $F"; fi
bash -n "$F" || { echo "!! run.sh 语法错误，已停止。恢复：cp $F.bak_1117b $F"; exit 1; }
F="$P/scripts/score.py"
if grep -q '"kdg"' "$F"; then echo "   score.py 已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1117b"
  $PY - "$F" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s2, n1 = re.subn(r'("kdeu": "[^"]*")\}', r'\1, "kdg": "+ PCMD 教师伪标签（干净版，只取通过检查的帧，2,126 帧）"}', s, count=1)
s2, n2 = re.subn(r"'kdeu'\]\.index\(step\)", "'kdeu', 'kdg'].index(step)", s2, count=1)
if n1 != 1 or n2 != 1: print('!! score.py 和预期不一样（%d,%d），没改' % (n1, n2)); sys.exit(1)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s2); print('   score.py 已改')
PYEOF
fi
$PY -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$F" || { echo "!! score.py 语法错误。恢复：cp $F.bak_1117b $F"; exit 1; }
if [ -z "$DRYRUN" ]; then
  T=$(mktemp -d); $PY "$CF/build_pcmd_labels.py" "$D" "$T/p.pkl" real_train clean_gated | sed 's/^/   /' || { rm -rf "$T"; echo "!! 拼伪标签失败"; exit 1; }; rm -rf "$T"; echo "   检查通过"
fi

say "C. scripts/tasks.txt：15 个新任务排在 11-17 那批之后（不暂停任何任务）"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; else
cp scripts/tasks.txt scripts/tasks.txt.bak_1117b
NEWTASKS="$NEWTASKS" $PY - scripts/tasks.txt <<'PYEOF'
import os, sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
def key(l):
    s = l.strip()
    if not s or s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
new = [tuple(l.split()) for l in os.environ['NEWTASKS'].strip().split('\n')]
NEW = ['# ==== 11-17b 补充材料：MAV6D +PST 只取通过检查的帧（2,126 帧）===='] + ['%-9s %s %-3s %-4s %s   # 11-17b 补充材料' % (k[0], k[1], k[2], k[3], k[4]) for k in new]
rest = [l for l in lines if key(l) not in new and not l.startswith('# ==== 11-17b')]
# 插在 11-17 kdu 那批（及被暂停的任务）之后：找"其余任务"标题；没有就放在最后一个 kdu/kdnu/kdeu 任务之后
i = next((k for k, l in enumerate(rest) if l.startswith('# ==== 其余任务')), None)
if i is None:
    ks = [k for k, l in enumerate(rest) if key(l) and key(l)[3] in ('kdu', 'kdnu', 'kdeu')]
    i = (ks[-1] + 1) if ks else next((k for k, l in enumerate(rest) if key(l)), len(rest))
out = rest[:i] + NEW + rest[i:]
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(out) + '\n')
print('   新任务 %d 个；共 %d 个任务' % (len(new), sum(1 for l in out if key(l))))
PYEOF
fi
for t in $NEWIDS; do [ -f "$Q/$t.fail" ] && [ -z "$DRYRUN" ] && rm -rf "$Q/$t.lock" "$Q/$t.fail" && echo "   清掉旧的失败记号：$t"; done

say "D. 没有工人的空闲显卡上开工（有工人的卡跑完当前任务会自动接）"
plist() { if [ -n "$PS_LIST_CMD" ]; then $PS_LIST_CMD; else ps -u "$(id -u)" -o pid= -o args=; fi; }
PSL="$(plist)"; NOW=""
while IFS=', ' read -r g mem; do
  [ -n "$g" ] || continue
  has=0; while read -r pid args; do [[ "$args" =~ worker[[:space:]]+${g}$ ]] && has=1; done <<< "$PSL"
  [ $has = 0 ] && [ "$mem" -lt 1000 ] && NOW="$NOW $g"
done <<< "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 没有工人且空闲的显卡：${NOW:-无}"
elif [ -n "${NOW// /}" ]; then GPUS="${NOW# }" bash scripts/run_all.sh; sleep 20
else echo "   每张卡都有工人"; fi

say "E. 后台自动打包（这批每完成一个就打包一次，全部完成后停止）"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; elif pgrep -f "autocollect_1117b" > /dev/null 2>&1; then echo "   已在运行"; else
  IDS="$(echo $NEWIDS)"
  nohup bash -c "autocollect_1117b=1; cd '$P'; last=-1; while true; do n=0; for t in $IDS; do [ -f runs/queue/\$t.done ] && n=\$((n+1)); done;     if [ \$n != \$last ]; then [ \$last -ge 0 ] && bash scripts/collect_results.sh > runs/logs/autocollect_1117b.log 2>&1 && echo \"[\$(date '+%F %T')] fix_1117b：完成 \$n 个，已打包 results_to_send_*.zip\" >> runs/STATUS.txt; last=\$n; fi;     [ \$n -ge $(echo $NEWIDS | wc -w) ] && exit 0; sleep 600; done" > /dev/null 2>&1 &
  echo "   已启动"
fi

say "新任务状态"
for k in $NEWIDS; do s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"; printf '   %-30s %s\n' "$k" "$s"; done
echo; echo "完成。这批排在 11-17 那批之后，结果会随 fix_1117 的后台自动打包一起出现在 results_to_send_*.zip 里；也可以随时 bash scripts/collect_results.sh。"
