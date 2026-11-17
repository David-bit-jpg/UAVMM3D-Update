#!/usr/bin/env bash
# 11-15 补丁 b：用"不含红外 / LiDAR 信息"的伪标签训对手检测器，证明我们伪标签里的红外 / LiDAR 知识才是提升来源。
# 先运行过 fix_1114.sh（以及 1114b / 1115）才能用本脚本。
#   新步骤 kdn = 伪标签换成"纯彩色教师"版（同一流程，只是教师训练时不用红外 / LiDAR）   → runs/<方法>_<源域>_kdn_<目标域>_s0/
#   新步骤 kde = 伪标签换成"无传感器 E1"版（整条流程都不用红外 / LiDAR）                  → runs/<方法>_<源域>_kde_<目标域>_s0/
#   两个目标域都跑：MAV6D（in → mav）和 AntiUAV3D（out → anti）。伪标签只含 seq / frame / center / lwh / euler_xyz，不含任何真实标签。
# 改动：
#   A. 拷入 4 个伪标签表，核对字段和每一帧都在 data/cmpfull/<real_train|outdoor_train>/labels.pkl 里。
#   B. scripts/run.sh：新增步骤 kdn / kde（MAV6D 和 AntiUAV3D 都可用）；score.py 加显示名。
#   C. scripts/tasks.txt：8 个新任务排在最前面（GUPNet 4 个 → MonoDGP 4 个）。
#   D. 腾显卡：正在跑的 4 个两天的补充材料任务（MonoDETR / MonoCoP 的 kdc 种子 1、2）暂停，
#      排到队尾，之后从存档接着跑（MonoDETR / MonoCoP 每轮存档，最多损失 1 轮）。它们所在显卡的工人会自动领新任务。
#      只停本账号下、命令行里含该任务运行目录 runs/<任务>/ 或 "scripts/run.sh <方法> <种子> in kdc mav" 的进程，其他任务一律不动。
#   E. 若有显卡上没有工人（工人死了）且空闲，就在那张卡上开工。
# 用法：bash fix_1115b.sh [实验包根目录]      只检查不改动：DRYRUN=1 bash fix_1115b.sh
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; CF="$P/Open3DUAVDet/output/camnorm/geo1004/cmpfull"; Q="$P/runs/queue"; D="$P/data/cmpfull"
say() { echo "== $1"; }
[ -n "$DRYRUN" ] && echo "（只检查模式：不改任何文件、不停任何进程）"
export PYTHONIOENCODING=utf-8
grep -q "st|kd|kdc" scripts/run.sh && grep -q "\[UAV 1114\]" "$CF/build_pcmd_labels.py" || { echo "!! 请先运行 fix_1114.sh"; exit 1; }
STOP_TASKS="monodetr_in_kdc_mav_s1 monodetr_in_kdc_mav_s2 monocop_in_kdc_mav_s1 monocop_in_kdc_mav_s2"

say "A. 4 个不含传感器信息的伪标签表"
for f in real_train_rgbonly real_train_nosensor outdoor_train_rgbonly outdoor_train_nosensor; do
  NEW="$HERE/data/pcmd_teacher_pseudo_$f.pkl"; split=${f%_*}
  [ -f "$NEW" ] || { echo "!! 补丁包里缺 data/pcmd_teacher_pseudo_$f.pkl"; exit 1; }
  [ -f "$D/$split/labels.pkl" ] || { echo "!! 缺 $D/$split/labels.pkl"; exit 1; }
  $PY - "$NEW" "$D/$split/labels.pkl" "$f" <<'PYEOF'
import pickle, sys
t = pickle.load(open(sys.argv[1], 'rb')); lab = {(l['seq'], l['frame']) for l in pickle.load(open(sys.argv[2], 'rb'))}
assert all(set(r) == {'seq', 'frame', 'center', 'lwh', 'euler_xyz'} for r in t), '字段不对'
miss = [r for r in t if (r['seq'], r['frame']) not in lab]
assert not miss, '%d 帧不在 labels.pkl 里' % len(miss)
print('   %-24s %5d 帧，全部在 %s 里' % (sys.argv[3], len(t), sys.argv[2].replace('\\', '/').split('/')[-2]))
PYEOF
  [ -n "$DRYRUN" ] || cp "$NEW" "$D/pcmd_teacher_pseudo_$f.pkl"
done

say "B. scripts/run.sh：新增步骤 kdn（纯彩色教师）/ kde（无传感器 E1），MAV6D 和 AntiUAV3D 都可用"
F="$P/scripts/run.sh"
if grep -q "\[UAV 1115b\]" "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1115b"
  $PY - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
assert s.count('  st|kd|kdc)\n') == 2, 'run.sh 里 "st|kd|kdc)" 的数目不是 2，没改'
s = s.replace('  st|kd|kdc)\n', '  st|kd|kdc|kdn|kde)\n')
old = '$([ $step = kdc ] && echo clean)   # [UAV 1114] kdc = 干净版伪标签\n'
assert s.count(old) == 1, 'run.sh 里找不到 fix_1114 改过的 build_pcmd_labels 那一行，没改'
s = s.replace(old, '$([ $step = kdc ] && echo clean)$([ $step = kdn ] && echo rgbonly)$([ $step = kde ] && echo nosensor)   # [UAV 1114] kdc = 干净版伪标签；[UAV 1115b] kdn = 纯彩色教师，kde = 无传感器 E1\n')
old = '{ echo "kdc（干净版伪标签）只用于 MAV6D：目标域写 mav"; exit 1; }   # [UAV 1114]\n'
assert s.count(old) == 1, 'run.sh 里找不到 fix_1114 的 kdc 检查行，没改'
s = s.replace(old, old + '{ [ $step = kdn ] || [ $step = kde ]; } && [ "$tgt" != mav ] && [ "$tgt" != anti ] && { echo "kdn / kde 的目标域只能是 mav / anti"; exit 1; }   # [UAV 1115b]\n')
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s); print('   已改')
PYEOF
else echo "   [只检查] 会改 $F"; fi
bash -n "$F" || { echo "!! run.sh 语法错误，已停止。恢复：cp $F.bak_1115b $F"; exit 1; }

F="$P/scripts/score.py"
if grep -q '"kdn"' "$F"; then echo "   score.py 已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1115b"
  $PY - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s2 = s.replace('"kdc": "+ PCMD 教师伪标签（干净版）"}', '"kdc": "+ PCMD 教师伪标签（干净版）", "kdn": "+ PCMD 伪标签（纯彩色教师）", "kde": "+ PCMD 伪标签（无传感器 E1）"}')
s2 = s2.replace("'kd', 'kdfu', 'kdc'].index(step)", "'kd', 'kdfu', 'kdc', 'kdn', 'kde'].index(step)")
if s2 == s: print('   score.py 和预期不一样，跳过（不影响训练，只是表里显示 kdn / kde）')
else: open(p, 'w', encoding='utf-8', newline=chr(10)).write(s2); print('   score.py 已改')
PYEOF
fi
$PY -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$F" || { echo "!! score.py 语法错误。恢复：cp $F.bak_1115b $F"; exit 1; }

if [ -z "$DRYRUN" ]; then
  say "   检查：拼伪标签"
  T=$(mktemp -d); bad=0
  for x in "real_train rgbonly" "real_train nosensor" "outdoor_train rgbonly" "outdoor_train nosensor"; do
    $PY "$CF/build_pcmd_labels.py" "$D" "$T/p.pkl" $x | sed 's/^/   /' || bad=1
  done
  rm -rf "$T"
  grep -q "st|kd|kdc|kdn|kde)" "$P/scripts/run.sh" || bad=1
  [ $bad = 0 ] || { echo "!! 检查没通过，已停止。把本屏输出发给我们"; exit 1; }
  echo "   通过"
fi

say "C. scripts/tasks.txt：8 个新任务排在最前面；4 个要暂停的补充材料任务移到最后"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; else
cp scripts/tasks.txt scripts/tasks.txt.bak_1115b
$PY - scripts/tasks.txt <<'PYEOF'
import sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
NEW = []
for m, h in [('gupnet', '6 h'), ('monodgp', '1 天')]:
    NEW += ['%-9s 0 in  kdn mav    # 11-15b 纯彩色教师伪标签，约 %s' % (m, h), '%-9s 0 in  kde mav    # 11-15b 无传感器 E1 伪标签，约 %s' % (m, h),
            '%-9s 0 out kdn anti   # 11-15b 纯彩色教师伪标签，约 %s' % (m, h), '%-9s 0 out kde anti   # 11-15b 无传感器 E1 伪标签，约 %s' % (m, h)]
MOVE = [('monodetr', '1', 'in', 'kdc', 'mav'), ('monodetr', '2', 'in', 'kdc', 'mav'), ('monocop', '1', 'in', 'kdc', 'mav'), ('monocop', '2', 'in', 'kdc', 'mav')]
def key(l):
    s = l.strip()
    if s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
newk = [key(l) for l in NEW]
moved = [l for l in lines if key(l) in MOVE]
rest = [l for l in lines if key(l) not in newk and key(l) not in MOVE and not l.startswith('# ==== 11-15b')]   # 重复运行时先拿掉，再放回
i = next((k for k, l in enumerate(rest) if key(l)), len(rest))
out = rest[:i] + NEW + rest[i:] + ['# ==== 11-15b：暂停的补充材料任务（MonoDETR / MonoCoP kdc 种子 1、2），之后从存档接着跑 ===='] * bool(moved) + moved
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(out) + '\n')
print('   队首新加 8 个；移到队尾 %d 个；共 %d 个任务' % (len(moved), sum(1 for l in out if key(l))))
PYEOF
fi

say "D. 暂停 4 个补充材料任务（只停本账号下属于这 4 个任务的进程）"
plist() { if [ -n "$PS_LIST_CMD" ]; then $PS_LIST_CMD; else ps -u "$(id -u)" -o pid= -o args=; fi; }   # PS_LIST_CMD 只在本地测试时用
PSL="$(plist)"
pids_of() {  # $1 = 任务 id，如 monodetr_in_kdc_mav_s1
  local t=$1 m s pat1 pat2 pid args; m=${t%%_*}; s=${t##*_s}
  pat1="runs/${t}(/|[[:space:]]|\$)"; pat2="scripts/run\.sh ${m} ${s} in kdc mav([[:space:]]|\$)"
  while read -r pid args; do
    [ -n "$pid" ] && [ "$pid" != "$$" ] || continue
    if [[ "$args" =~ $pat1 ]] || [[ "$args" =~ $pat2 ]]; then echo "$pid"; fi
  done <<< "$PSL"
}
ALL=""; HIT=""; KT=""
for t in $STOP_TASKS; do
  p=$(pids_of $t | tr '\n' ' ')
  { [ -n "$p" ] || [ -d "$Q/$t.lock" ]; } && [ ! -f "$Q/$t.done" ] && HIT="$HIT $t"
  st="没在跑"; [ -d "$Q/$t.lock" ] && st="有 .lock"; [ -f "$Q/$t.done" ] && st="已完成"
  printf '   %-26s %-8s 进程：%s\n' "$t" "$st" "${p:-无}"
  [ -n "$p" ] && echo "$PSL" | awk -v L=" $p " 'index(L, " " $1 " ") {print "      " substr($0, 1, 160)}'
  ALL="$ALL $p"; [ -n "$p" ] && KT="$KT $t"
done
if [ -n "$DRYRUN" ]; then echo "   [只检查] 会停以上进程，清掉这 4 个任务的 .lock / .fail"; else
  if [ -n "${ALL// /}" ]; then
    kill -TERM $ALL 2>/dev/null || true
    for i in $(seq 30); do left=""; for p in $ALL; do kill -0 $p 2>/dev/null && left="$left $p"; done; [ -z "$left" ] && break; sleep 1; done
    [ -n "$left" ] && { echo "   30 秒没退出，强制结束：$left"; kill -KILL $left 2>/dev/null || true; }
    # 工人发现任务进程退出后会写 .fail 再去领新任务；等它写完再清，免得清完又被写上
    for i in $(seq 60); do w=0; for t in $KT; do [ -d "$Q/$t.lock" ] && [ ! -f "$Q/$t.fail" ] && [ ! -f "$Q/$t.done" ] && w=1; done; [ $w = 0 ] && break; sleep 1; done
  fi
  for t in $STOP_TASKS; do [ -f "$Q/$t.done" ] && continue; rm -rf "$Q/$t.lock" "$Q/$t.fail"; done
  for t in $HIT; do echo "[$(date '+%F %T')] fix_1115b：暂停 $t（排到队尾，之后从存档接着跑）" >> runs/STATUS.txt; done
  echo "   已暂停并清掉 .lock / .fail"
fi

say "E. 没有工人的空闲显卡上开工"
sleep ${SETTLE:-20}
PSL="$(plist)"; NOW=""
while IFS=', ' read -r g mem; do
  [ -n "$g" ] || continue
  has=0; while read -r pid args; do [[ "$args" =~ worker[[:space:]]+${g}$ ]] && has=1; done <<< "$PSL"
  [ $has = 0 ] && [ "$mem" -lt 1000 ] && NOW="$NOW $g"
done <<< "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
if [ -n "$DRYRUN" ]; then echo "   [只检查] 没有工人且空闲的显卡：${NOW:-无}"
elif [ -n "${NOW// /}" ]; then GPUS="${NOW# }" bash scripts/run_all.sh; sleep 20
else echo "   每张卡都有工人在跑：腾出来的卡已自动领新任务"; fi

say "新任务状态"
for k in $(grep -v '^\s*#' scripts/tasks.txt | awk 'NF>=5 && ($4=="kdn" || $4=="kde"){print $1"_"$3"_"$4"_"$5"_s"$2}') $STOP_TASKS; do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"
  printf '   %-32s %s\n' "$k" "$s"
done
echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo; echo "完成。GUPNet 约 6 小时、MonoDGP 约 1 天出结果；随时 bash scripts/collect_results.sh 把 results_to_send_*.zip 发回。"
