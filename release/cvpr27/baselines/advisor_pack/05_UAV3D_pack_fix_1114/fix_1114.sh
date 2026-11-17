#!/usr/bin/env bash
# 11-14 一键补丁：MAV6D 上新增"干净版 PCMD 教师伪标签"（步骤名 kdc），四个检测器各跑一次，排在队首，在空闲显卡上开工。
# 已完成、正在跑的任务一律不动（不杀进程、不碰 .lock / .done）；旧的 kd 结果保留，kdc 写进新的运行目录 runs/<方法>_in_kdc_mav_s0/。
# 用法：bash fix_1114.sh [实验包根目录]
#   缺省目录：/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack
#   只检查不改动：DRYRUN=1 bash fix_1114.sh
#
# 为什么：11-10 包里的 MAV6D 伪标签（pcmd_teacher_pseudo_real_train.pkl，4,088 帧）来自开发期的流程；
#   论文正式版是 11-13 定下的干净流程，它导出的伪标签是 5,227 帧（data/pcmd_teacher_pseudo_real_train_clean.pkl）。
#   伪标签不含任何真实标签，只含 seq / frame / center / lwh / euler_xyz，格式与旧文件相同。
#   AntiUAV3D 的伪标签（11-12 更新的 249 帧）已经是干净流程的，不用重跑。
#
# 改动：
#   A. 拷入干净版伪标签，核对每一帧都在 data/cmpfull/real_train 里。
#   B. build_pcmd_labels.py：可选第 4 个参数 <标签版本>，读 pcmd_teacher_pseudo_<划分>_<版本>.pkl；不给参数时行为不变。
#   C. scripts/run.sh：新增步骤 kdc（= kd，只是伪标签换成干净版），只用于 MAV6D（目标域 mav）。
#   D. scripts/score.py：分数表里 kdc 显示为"+ PCMD 教师伪标签（干净版）"。
#   E. 失败任务重新排队；按论文急需程度重排 scripts/tasks.txt，四个 kdc 任务排在主表缺的任务之后、其余任务之前。
#   F. 在空闲显卡上开工。
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; CF="$P/Open3DUAVDet/output/camnorm/geo1004/cmpfull"; Q="$P/runs/queue"; D="$P/data/cmpfull"
say() { echo "== $1"; }
[ -n "$DRYRUN" ] && echo "（只检查模式：不改任何文件）"
export PYTHONIOENCODING=utf-8

say "A. 干净版 MAV6D 伪标签"
NEW="$HERE/data/pcmd_teacher_pseudo_real_train_clean.pkl"
[ -f "$NEW" ] || { echo "!! 补丁包里缺 data/pcmd_teacher_pseudo_real_train_clean.pkl"; exit 1; }
$PY - "$NEW" "$D/real_train/labels.pkl" <<'PYEOF'
import pickle, sys
t = pickle.load(open(sys.argv[1], 'rb')); lab = {(l['seq'], l['frame']) for l in pickle.load(open(sys.argv[2], 'rb'))}
miss = [r for r in t if (r['seq'], r['frame']) not in lab]
assert all(set(r) == {'seq', 'frame', 'center', 'lwh', 'euler_xyz'} for r in t), '字段不对'
assert not miss, '%d 帧不在 real_train 里' % len(miss)
print('   %d 帧，全部在 real_train 里' % len(t))
PYEOF
[ -n "$DRYRUN" ] || cp "$NEW" "$D/pcmd_teacher_pseudo_real_train_clean.pkl"

say "B. build_pcmd_labels.py：支持标签版本参数"
F="$CF/build_pcmd_labels.py"; [ -f "$F" ] || { echo "!! 缺 $F"; exit 1; }
if grep -q "\[UAV 1114\]" "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1114"
  $PY - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
old = "tab = pickle.load(open(os.path.join(data, 'pcmd_teacher_pseudo_%s.pkl' % split), 'rb'))"
new = ("tag = sys.argv[4] if len(sys.argv) > 4 else ''   # [UAV 1114] 可选标签版本：clean -> pcmd_teacher_pseudo_<split>_clean.pkl\n"
       "tab = pickle.load(open(os.path.join(data, 'pcmd_teacher_pseudo_%s%s.pkl' % (split, '_' + tag if tag else '')), 'rb'))")
assert s.count(old) == 1, 'build_pcmd_labels.py 和预期不一样，没改'
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s.replace(old, new)); print('   已改')
PYEOF
else echo "   [只检查] 会改 $F"; fi

say "C. scripts/run.sh：新增步骤 kdc（只用于 MAV6D）"
F="$P/scripts/run.sh"
if grep -q "\[UAV 1114\]" "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1114"
  $PY - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
assert s.count('  st|kd)\n') == 2, 'run.sh 里 "st|kd)" 的数目不是 2，没改'
s = s.replace('  st|kd)\n', '  st|kd|kdc)\n')
old = '$PY "$H/build_pcmd_labels.py" "$DATA" "$O/pseudo.pkl" $TGT\n'
assert s.count(old) == 1, 'run.sh 里找不到 build_pcmd_labels 那一行，没改'
s = s.replace(old, '$PY "$H/build_pcmd_labels.py" "$DATA" "$O/pseudo.pkl" $TGT $([ $step = kdc ] && echo clean)   # [UAV 1114] kdc = 干净版伪标签\n')
old = 'case $step in\n  sim)'
assert s.count(old) == 2, 'run.sh 结构和预期不一样，没改'
s = s.replace(old, '[ $step = kdc ] && [ "$tgt" != mav ] && { echo "kdc（干净版伪标签）只用于 MAV6D：目标域写 mav"; exit 1; }   # [UAV 1114]\ncase $step in\n  sim)', 1)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s); print('   已改')
PYEOF
else echo "   [只检查] 会改 $F"; fi
bash -n "$F" || { echo "!! run.sh 语法错误，已停止。恢复：cp $F.bak_1114 $F"; exit 1; }

say "D. scripts/score.py：kdc 的显示名"
F="$P/scripts/score.py"
if grep -q '"kdc"' "$F"; then echo "   已改过"; elif [ -z "$DRYRUN" ]; then
  cp -n "$F" "$F.bak_1114"
  $PY - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
s2 = s.replace('"kdfu": "+ PCMD 教师伪标签（焦距统一）"}', '"kdfu": "+ PCMD 教师伪标签（焦距统一）", "kdc": "+ PCMD 教师伪标签（干净版）"}')
s2 = s2.replace("['sim', 'simfu', 'st', 'stfu', 'kd', 'kdfu'].index(step)", "['sim', 'simfu', 'st', 'stfu', 'kd', 'kdfu', 'kdc'].index(step)")
if s2 == s: print('   score.py 和预期不一样，跳过（不影响训练，只是表里显示 kdc）')
else: open(p, 'w', encoding='utf-8', newline=chr(10)).write(s2); print('   已改')
PYEOF
fi

say "检查改动"
bad=0
for f in "$CF/build_pcmd_labels.py" "$P/scripts/score.py"; do $PY -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" || { echo "!! 语法错误：$f"; bad=1; }; done
if [ -z "$DRYRUN" ]; then
  T=$(mktemp -d); $PY "$CF/build_pcmd_labels.py" "$D" "$T/pseudo.pkl" real_train clean || bad=1
  $PY -c "import pickle,sys; n=len(pickle.load(open(sys.argv[1],'rb'))); assert n==5227, n; print('   干净版伪标签拼好：%d 帧' % n)" "$T/pseudo.pkl" || bad=1
  $PY "$CF/build_pcmd_labels.py" "$D" "$T/old.pkl" real_train > /dev/null || { echo "!! 不带版本参数的旧用法坏了"; bad=1; }
  rm -rf "$T"
  grep -q "st|kd|kdc)" "$P/scripts/run.sh" || { echo "!! run.sh 未改成功"; bad=1; }
fi
[ $bad = 0 ] || { echo "!! 检查没通过，已停止。把本屏输出发给我们"; exit 1; }
echo "   通过"

say "E. 失败任务重新排队（只清 .fail 和它的 .lock；正在跑的任务不动）"
k=0; shopt -s nullglob
for f in "$Q"/*.fail; do id=$(basename "$f" .fail); if [ -n "$DRYRUN" ]; then echo "   [只检查] 会重新排队：$id"; else rm -rf "$Q/$id.fail" "$Q/$id.lock"; echo "   重新排队：$id"; fi; k=$((k+1)); done
echo "   共 $k 个"

say "   重排 scripts/tasks.txt（任务一个不少，只改顺序；新增 4 个 kdc 任务）"
if [ -z "$DRYRUN" ]; then
cp "$P/scripts/tasks.txt" "$P/scripts/tasks.txt.bak_1114"
$PY - "$P/scripts/tasks.txt" <<'PYEOF'
import sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
def key(l):
    s = l.strip()
    if s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
NEW = {'gupnet 0 in kdc mav': 'gupnet    0 in  kdc mav    #   6.3 h（11-14 干净版伪标签）',
       'monodgp 0 in kdc mav': 'monodgp   0 in  kdc mav    #  29.3 h（11-14 干净版伪标签）',
       'monodetr 0 in kdc mav': 'monodetr  0 in  kdc mav    #  51.1 h（11-14 干净版伪标签）',
       'monocop 0 in kdc mav': 'monocop   0 in  kdc mav    #  51.1 h（11-14 干净版伪标签）'}
PRIO = ['monocd 0 out sim both', 'monocd 0 out kd anti', 'monocd 0 out st anti',      # 主表 AntiUAV3D 的 MonoCD 一行
        'monodgp 0 in st mav', 'monocop 0 in st mav',                                  # 主表 MAV6D 的 +ST
        'monocop 0 in kdc mav', 'monodetr 0 in kdc mav', 'monodgp 0 in kdc mav', 'gupnet 0 in kdc mav']   # 长的先开
tasks, order = {}, []
for l in lines:
    k = key(l)
    if k is None or k in tasks: continue
    tasks[k] = l; order.append(k)
for s, l in NEW.items():
    k = tuple(s.split())
    if k not in tasks: tasks[k] = l; order.append(k)
head = [l for l in lines if l.lstrip().startswith('#') and not l.lstrip().startswith('# ====')]
out = head + ['# ==== 11-14 fix_1114.sh：主表缺的 → MAV6D 干净版伪标签（kdc）→ 其余；全部任务都保留 ====']
used = set()
for s in PRIO:
    k = tuple(s.split())
    if k in tasks: out.append(tasks[k]); used.add(k)
out += [tasks[k] for k in order if k not in used]
open(p, 'w', encoding='utf-8', newline=chr(10)).write('\n'.join(out) + '\n')
print('   共 %d 个任务' % len(order))
PYEOF
fi

say "F. 在空闲显卡上开工"
IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{printf "%s ", $1}')
if [ -n "$DRYRUN" ]; then echo "   [只检查] 空闲显卡：${IDLE:-无}"
elif [ -n "$IDLE" ]; then GPUS="$IDLE" bash scripts/run_all.sh; else echo "   暂无空闲显卡：正在跑的工人跑完当前任务后会自动接新任务（kdc 已排在前面）"; fi

say "各任务状态"
for k in $(grep -v '^\s*#' scripts/tasks.txt | awk 'NF>=5{print $1"_"$3"_"$4"_"$5"_s"$2}'); do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"
  printf '   %-32s %s\n' "$k" "$s"
done
echo; echo "完成。之后看进度：cat runs/STATUS.txt；随时可以 bash scripts/collect_results.sh，把 results_to_send_*.zip 发回来。"
