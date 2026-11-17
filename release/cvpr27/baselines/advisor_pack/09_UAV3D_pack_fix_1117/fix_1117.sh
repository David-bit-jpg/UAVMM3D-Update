#!/usr/bin/env bash
# 11-17 补丁【最高优先】：RealDrone3D（AntiUAV3D）上对手方法的 +PST 改用"不筛帧"的伪标签表（909 帧），和 MAV6D 的规则一致。
#   旧表（11-12）只留"红外、LiDAR 两组朝向一致"的 249 帧；MAV6D 的表（11-14）不做这一筛，用全部有伪框的帧。两边规则不同，正文主表要统一。
#   新步骤 kdu  = kd 的伪标签换成不筛帧版（909 帧）           → runs/<方法>_out_kdu_anti_s<种子>/
#          kdnu = 纯彩色教师版，不筛帧（909 帧，补充材料用）  → runs/<方法>_out_kdnu_anti_s<种子>/
#          kdeu = 无传感器 E1 版，不筛帧（909 帧，补充材料用）→ runs/<方法>_out_kdeu_anti_s<种子>/
#   伪标签只含 seq / frame / center / lwh / euler_xyz，不含任何真实标签。旧的 kd / kdn / kde 结果全部保留。
# 脚本会：
#   A. 拷入 3 个新伪标签表并核对（每帧都在 data/cmpfull/outdoor_train/labels.pkl 里）。
#   B. scripts/run.sh 新增步骤 kdu / kdnu / kdeu（只用于 anti）；除 GUPNet 外数据加载进程 2 → 6（MonoCD 在服务器上 77% 时间在等数据；只影响速度，不影响训练内容）。
#      scripts/score.py 加显示名。改动前备份为 *.bak_1117。
#   C. scripts/tasks.txt：新任务放在最前面（按论文急需程度）；被暂停的任务紧跟其后；其余任务原顺序不变，一个不少。已完成的任务不动。
#   D. 暂停正在跑的旧任务，把显卡全部让给新任务：只停本账号下、命令行里含该任务运行目录 runs/<任务>/ 或 "scripts/run.sh <该任务>" 的进程。
#      被暂停的任务排在新任务后面，之后自动重新开始（有存档的方法从存档接着跑）。队列外单独跑的程序（如 11-16 的 GUPNet+DANN）一律不碰。
#   E. 没有工人的显卡上开工。F. 后台每 10 分钟检查一次，新任务每完成一个就自动打包一次 results_to_send_*.zip。
# 用法：bash fix_1117.sh [实验包根目录]      只检查不改动：DRYRUN=1 bash fix_1117.sh
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; CF="$P/Open3DUAVDet/output/camnorm/geo1004/cmpfull"; Q="$P/runs/queue"; D="$P/data/cmpfull"
say() { echo "== $1"; }
[ -n "$DRYRUN" ] && echo "（只检查模式：不改任何文件、不停任何进程）"
export PYTHONIOENCODING=utf-8
grep -q "\[UAV 1114\]" "$CF/build_pcmd_labels.py" && grep -q "\[UAV 1114\]" scripts/run.sh || { echo "!! 请先运行 fix_1114.sh"; exit 1; }
mkdir -p "$Q"

# 新任务（从上往下 = 优先级）。按服务器 11-12 实测用时排（GUPNet 3.6 h、MonoDETR 3.3 h、MonoDGP 5.2 h、MonoCoP 8.7 h、MonoCD 6.8 h，+12% 帧）：8 张卡第一批 = MonoCoP/MonoCD 种子 0、MonoDGP 三个种子、GUPNet 种子 0/1、MonoDETR 种子 0；约 4 h 后空出的卡接 MonoDETR 种子 1/2、GUPNet 种子 2。
NEWTASKS="monocop 0 out kdu anti
monocd 0 out kdu anti
monodgp 0 out kdu anti
monodgp 1 out kdu anti
monodgp 2 out kdu anti
gupnet 0 out kdu anti
gupnet 1 out kdu anti
monodetr 0 out kdu anti
monodetr 1 out kdu anti
monodetr 2 out kdu anti
gupnet 2 out kdu anti
monocd 1 out kdu anti
monocd 2 out kdu anti
monocop 1 out kdu anti
monocop 2 out kdu anti
gupnet 0 out kdnu anti
gupnet 0 out kdeu anti
monodgp 0 out kdnu anti
monodgp 0 out kdeu anti"
NEWIDS=$(echo "$NEWTASKS" | awk '{print $1"_"$3"_"$4"_"$5"_s"$2}')

say "A. 3 个不筛帧的伪标签表（RealDrone3D 训练图，各 909 帧）"
for f in outdoor_train_ungated outdoor_train_rgbonly_ungated outdoor_train_nosensor_ungated; do
  NEW="$HERE/data/pcmd_teacher_pseudo_$f.pkl"
  [ -f "$NEW" ] || { echo "!! 补丁包里缺 data/pcmd_teacher_pseudo_$f.pkl"; exit 1; }
  [ -f "$D/outdoor_train/labels.pkl" ] || { echo "!! 缺 $D/outdoor_train/labels.pkl"; exit 1; }
  $PY - "$NEW" "$D/outdoor_train/labels.pkl" "$f" <<'PYEOF'
import pickle, sys
t = pickle.load(open(sys.argv[1], 'rb')); lab = {(l['seq'], l['frame']) for l in pickle.load(open(sys.argv[2], 'rb'))}
assert all(set(r) == {'seq', 'frame', 'center', 'lwh', 'euler_xyz'} for r in t), '字段不对'
miss = [r for r in t if (r['seq'], r['frame']) not in lab]
assert not miss, '%d 帧不在 labels.pkl 里' % len(miss)
assert len(t) == 909, '帧数 %d，应为 909' % len(t)
print('   %-32s %5d 帧，全部在 outdoor_train 里' % (sys.argv[3], len(t)))
PYEOF
  [ -n "$DRYRUN" ] || cp "$NEW" "$D/pcmd_teacher_pseudo_$f.pkl"
done
if [ -f "$D/pcmd_teacher_pseudo_outdoor_train.pkl" ]; then
  $PY - "$HERE/data/pcmd_teacher_pseudo_outdoor_train_ungated.pkl" "$D/pcmd_teacher_pseudo_outdoor_train.pkl" <<'PYEOF'
import pickle, sys
import numpy as np
u = {(r['seq'], r['frame']): r for r in pickle.load(open(sys.argv[1], 'rb'))}; g = pickle.load(open(sys.argv[2], 'rb'))
same = all((r['seq'], r['frame']) in u and np.allclose(r['center'], u[(r['seq'], r['frame'])]['center']) for r in g)
print('   核对：服务器上旧的 %d 帧表%s新表的子集（同帧同中心）' % (len(g), '是' if same else '【不是】'))
PYEOF
fi

say "B. scripts/run.sh：新增步骤 kdu / kdnu / kdeu（只用于 anti）"
F="$P/scripts/run.sh"
if grep -q "\[UAV 1117\]" "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1117"
  $PY - "$F" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s, n = re.subn(r'(?m)^(  st\|kd[a-z|]*)\)$', r'\1|kdu|kdnu|kdeu)', s)
assert n == 2, 'run.sh 里 "  st|kd...)" 的行数是 %d，不是 2，没改' % n
lines = s.split('\n'); hit = [i for i, l in enumerate(lines) if 'build_pcmd_labels.py" "$DATA" "$O/pseudo.pkl" $TGT' in l]
assert len(hit) == 1, 'run.sh 里 build_pcmd_labels 那一行有 %d 处，没改' % len(hit)
l = lines[hit[0]]; k = l.find('   #'); cmd, com = (l[:k], l[k:]) if k >= 0 else (l, '')
lines[hit[0]] = cmd + '$([ $step = kdu ] && echo ungated)$([ $step = kdnu ] && echo rgbonly_ungated)$([ $step = kdeu ] && echo nosensor_ungated)' + com + '   # [UAV 1117] kdu/kdnu/kdeu = 不筛帧版'
s = '\n'.join(lines)
old = 'case $step in\n  sim)'
assert s.count(old) >= 1, 'run.sh 里找不到 "case $step in / sim)"，没改'   # 第一处 = 参数检查
s = s.replace(old, 'case $step in kdu|kdnu|kdeu) [ "$tgt" = anti ] || { echo "kdu / kdnu / kdeu 只用于 anti"; exit 1; }; [ "$m" != gupnet ] && WORKERS=${WORKERS:-6} ;; esac   # [UAV 1117] 数据加载进程 2 -> 6（GUPNet 没有这个参数），只影响速度\n' + old, 1)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s); print('   已改')
PYEOF
else echo "   [只检查] 会改 $F"; fi
bash -n "$F" || { echo "!! run.sh 语法错误，已停止。恢复：cp $F.bak_1117 $F"; exit 1; }
grep -q '^m=\|^m=\$1\|read m\|m="\$1"\|m=\$1' "$F" || grep -q '\$m' "$F" || { echo "!! run.sh 里没有方法变量 \$m，已停止"; exit 1; }

F="$P/scripts/score.py"
if grep -q '"kdu"' "$F"; then echo "   score.py 已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1117"
  $PY - "$F" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s2, n1 = re.subn(r'("kde": "[^"]*")\}', r'\1, "kdu": "+ PCMD 教师伪标签（不筛帧，909 帧）", "kdnu": "+ PCMD 伪标签（纯彩色教师，不筛帧）", "kdeu": "+ PCMD 伪标签（无传感器 E1，不筛帧）"}', s, count=1)
s2, n2 = re.subn(r"'kde'\]\.index\(step\)", "'kde', 'kdu', 'kdnu', 'kdeu'].index(step)", s2, count=1)
if n1 != 1 or n2 != 1: print('!! score.py 和预期不一样（%d,%d），没改' % (n1, n2)); sys.exit(1)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s2); print('   score.py 已改')
PYEOF
fi
$PY -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$F" || { echo "!! score.py 语法错误。恢复：cp $F.bak_1117 $F"; exit 1; }

if [ -z "$DRYRUN" ]; then
  say "   检查：拼伪标签"
  T=$(mktemp -d); bad=0
  for v in ungated rgbonly_ungated nosensor_ungated; do $PY "$CF/build_pcmd_labels.py" "$D" "$T/p.pkl" outdoor_train $v | sed 's/^/   /' || bad=1; done
  rm -rf "$T"
  grep -q "|kdu|kdnu|kdeu)" "$P/scripts/run.sh" || bad=1
  [ $bad = 0 ] || { echo "!! 检查没通过，已停止。把本屏输出发给我们"; exit 1; }
  echo "   通过"
fi

say "D0. 现在正在跑的旧任务"
plist() { if [ -n "$PS_LIST_CMD" ]; then $PS_LIST_CMD; else ps -u "$(id -u)" -o pid= -o args=; fi; }
PSL="$(plist)"
pids_of() {  # $1 = 任务 id（方法_源域_步骤_目标域_s种子）
  local t=$1 m s rest src step tgt pat1 pat2 pid args
  m=${t%%_*}; s=${t##*_s}; rest=${t#*_}; src=${rest%%_*}; rest=${rest#*_}; step=${rest%%_*}; rest=${rest#*_}; tgt=${rest%_s*}
  pat1="runs/${t}(/|[[:space:]]|\$)"; pat2="scripts/run\.sh ${m} ${s} ${src} ${step} ${tgt}([[:space:]]|\$)"
  while read -r pid args; do
    [ -n "$pid" ] && [ "$pid" != "$$" ] || continue
    if [[ "$args" =~ $pat1 ]] || [[ "$args" =~ $pat2 ]]; then echo "$pid"; fi
  done <<< "$PSL"
}
RUNNING=""
for L in "$Q"/*.lock; do
  [ -d "$L" ] || continue; t=$(basename "$L" .lock)
  [ -f "$Q/$t.done" ] || [ -f "$Q/$t.fail" ] && continue
  echo "$NEWIDS" | grep -qx "$t" && continue
  p=$(pids_of $t | tr '\n' ' ')
  printf '   %-30s 进程：%s\n' "$t" "${p:-无（陈旧的 .lock）}"
  RUNNING="$RUNNING $t"
done
[ -n "$RUNNING" ] || echo "   没有"

say "C. scripts/tasks.txt：新任务排最前，被暂停的旧任务紧跟其后，其余顺序不变"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; else
cp scripts/tasks.txt scripts/tasks.txt.bak_1117
NEWTASKS="$NEWTASKS" RUNNING="$RUNNING" $PY - scripts/tasks.txt <<'PYEOF'
import os, sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
def key(l):
    s = l.strip()
    if not s or s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
tid = lambda k: '%s_%s_%s_%s_s%s' % (k[0], k[2], k[3], k[4], k[1])
new = [tuple(l.split()) for l in os.environ['NEWTASKS'].strip().split('\n')]
run = set(os.environ['RUNNING'].split())
NOTE = {'kdu': '11-17 不筛帧伪标签（909 帧），正文主表', 'kdnu': '11-17 纯彩色教师、不筛帧，补充材料', 'kdeu': '11-17 无传感器 E1、不筛帧，补充材料'}
NEW = ['%-9s %s %-3s %-4s %s   # %s' % (k[0], k[1], k[2], k[3], k[4], NOTE[k[3]]) for k in new]
paused = [l for l in lines if key(l) and tid(key(l)) in run and key(l) not in new]
have = {tid(key(l)) for l in paused}
for t in sorted(run - have):   # 正在跑但队列文件里没有这一行的任务：补一行，保证暂停后还会被领回来
    a = t.split('_'); seed = a[-1][1:]; m, src, step, tgt = a[0], a[1], a[2], '_'.join(a[3:-1])
    if (m, seed, src, step, tgt) not in new: paused.append('%-9s %s %-3s %-4s %s   # 11-17 暂停（原来不在队列文件里）' % (m, seed, src, step, tgt))
rest = [l for l in lines if key(l) not in new and (key(l) is None or tid(key(l)) not in run) and not l.startswith('# ==== 11-17') and not l.startswith('# ==== 其余任务')]
i = next((k for k, l in enumerate(rest) if key(l)), len(rest))
out = (rest[:i] + ['# ==== 11-17 最高优先：RealDrone3D +PST 不筛帧（909 帧）===='] + NEW
       + (['# ==== 11-17：被暂停的旧任务，新任务之后自动重新开始 ===='] + paused if paused else []) + ['# ==== 其余任务（原顺序）===='] + rest[i:])
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(out) + '\n')
print('   新任务 %d 个排最前；暂停后紧跟其后的 %d 个；共 %d 个任务' % (len(NEW), len(paused), sum(1 for l in out if key(l))))
PYEOF
fi
# 新任务如果以前失败过，清掉失败记号重新排
for t in $NEWIDS; do [ -f "$Q/$t.fail" ] && [ -z "$DRYRUN" ] && rm -rf "$Q/$t.lock" "$Q/$t.fail" && echo "   清掉旧的失败记号：$t"; done

say "D. 暂停正在跑的旧任务（只停这些任务自己的进程）"
ALL=""; KT=""
for t in $RUNNING; do p=$(pids_of $t | tr '\n' ' '); ALL="$ALL $p"; [ -n "$p" ] && KT="$KT $t"; done
if [ -n "$DRYRUN" ]; then echo "   [只检查] 会停：${RUNNING:-无}"; else
  if [ -n "${ALL// /}" ]; then
    kill -TERM $ALL 2>/dev/null || true
    for i in $(seq 30); do left=""; for p in $ALL; do kill -0 $p 2>/dev/null && left="$left $p"; done; [ -z "$left" ] && break; sleep 1; done
    [ -n "$left" ] && { echo "   30 秒没退出，强制结束：$left"; kill -KILL $left 2>/dev/null || true; }
    for i in $(seq 60); do w=0; for t in $KT; do [ -d "$Q/$t.lock" ] && [ ! -f "$Q/$t.fail" ] && [ ! -f "$Q/$t.done" ] && w=1; done; [ $w = 0 ] && break; sleep 1; done
  fi
  for t in $RUNNING; do [ -f "$Q/$t.done" ] && continue; rm -rf "$Q/$t.lock" "$Q/$t.fail"; echo "[$(date '+%F %T')] fix_1117：暂停 $t（排在 11-17 新任务之后）" >> runs/STATUS.txt; done
  echo "   已暂停：${RUNNING:-无}"
fi

say "E. 没有工人的显卡上开工"
sleep ${SETTLE:-20}
PSL="$(plist)"; NOW=""
while IFS=', ' read -r g mem; do
  [ -n "$g" ] || continue
  has=0; while read -r pid args; do [[ "$args" =~ worker[[:space:]]+${g}$ ]] && has=1; done <<< "$PSL"
  [ $has = 0 ] && [ "$mem" -lt 1000 ] && NOW="$NOW $g"
done <<< "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 没有工人且空闲的显卡：${NOW:-无}"
elif [ -n "${NOW// /}" ]; then GPUS="${NOW# }" bash scripts/run_all.sh; sleep 20
else echo "   每张卡都有工人：腾出来的卡会自动领新任务"; fi

say "F. 后台自动打包（新任务每完成一个就打包一次，全部完成后停止）"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; elif pgrep -f "autocollect_1117" > /dev/null 2>&1; then echo "   已在运行"; else
  IDS="$(echo $NEWIDS)"
  nohup bash -c "autocollect_1117=1; cd '$P'; last=-1; while true; do n=0; for t in $IDS; do [ -f runs/queue/\$t.done ] && n=\$((n+1)); done; \
    if [ \$n != \$last ]; then [ \$last -ge 0 ] && bash scripts/collect_results.sh > runs/logs/autocollect_1117.log 2>&1 && echo \"[\$(date '+%F %T')] fix_1117：新任务完成 \$n 个，已打包 results_to_send_*.zip\" >> runs/STATUS.txt; last=\$n; fi; \
    [ \$n -ge $(echo $NEWIDS | wc -w) ] && exit 0; sleep 600; done" > /dev/null 2>&1 &
  echo "   已启动：每完成一个新任务，实验包根目录下就多一个最新的 results_to_send_*.zip"
fi

say "新任务状态"
for k in $NEWIDS; do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"
  printf '   %-30s %s\n' "$k" "$s"
done
echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo; echo "完成。最先出结果的是 MonoDETR（约 4 小时）。随时可以把实验包根目录下最新的 results_to_send_*.zip 发回。"
