#!/usr/bin/env bash
# 11-13 一键修复并继续：修好所有失败任务的原因 → 失败任务重新排队 → 按论文急需程度排序 → 在空闲显卡上开工。
# 正在跑的任务一律不动（不杀进程、不碰它们的 .lock），它们跑完后工人会接着领新顺序里的任务。
# 用法：bash fix_all_1113.sh [实验包根目录]
#   缺省目录：/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack
#   只检查不改动：DRYRUN=1 bash fix_all_1113.sh
#
# 修复内容：
#   A. MonoCD 全部失败（"Pin memory thread exited unexpectedly"）：数据加载改为单进程、关闭 pin_memory。
#      这只影响读数据的速度，不影响训练结果。
#   B. 焦距统一版（antifu）全部失败：四个方法的 predict.py 只接受固定几个数据划分名，不认 outdoor_test_fu / outdoor_train_fu。
#      GUPNet 的 kdfu 其实已训完 140 轮，只是在预测这一步报错。现在把这两个名字加进白名单。
#      已训好的模型会保留，重跑时直接从预测开始。
#   C. 焦距统一的数据如果不全，自动先跑一次 scripts/focal_unify.py。
set -e
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -f "$P/scripts/tasks.txt" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/run.sh）"; exit 1; }
cd "$P"; PY=${PY:-python}; CF="$P/Open3DUAVDet/output/camnorm/geo1004/cmpfull"; Q="$P/runs/queue"
say() { echo "== $1"; }
run() { if [ -n "$DRYRUN" ]; then echo "   [只检查] $*"; else eval "$@"; fi; }

say "A. MonoCD：数据加载改为单进程、不用 pin_memory"
for f in "$CF/monocd/train.py" "$CF/monocd/predict.py"; do
  [ -f "$f" ] || { echo "!! 缺 $f"; exit 1; }
  run "cp -n '$f' '$f.bak_1113'"
  run "sed -i \"s/add_argument('--workers', type=int, default=2)/add_argument('--workers', type=int, default=0)/\" '$f'"
done
run "sed -i 's/pin_memory=True/pin_memory=False/' '$CF/monocd/train.py'"

say "B. 四个方法的 predict.py：允许焦距统一版的数据划分（去掉 --split 的固定白名单，其余不动）"
for m in gupnet monodgp monodetr monocop; do
  f="$CF/$m/predict.py"; [ -f "$f" ] || { echo "!! 缺 $f"; exit 1; }
  if ! grep -E "add_argument.*--split" "$f" | grep -q "choices"; then echo "   $m：--split 没有白名单，已支持"; continue; fi
  run "cp -n '$f' '$f.bak_1113'"
  run "sed -i -E '/add_argument.*--split/ s/, *choices=\[[^]]*\]//' '$f'"
  echo "   $m：已去掉 --split 的白名单"
done
say "检查改动"
bad=0
for f in "$CF/monocd/train.py" "$CF/monocd/predict.py" "$CF"/{gupnet,monodgp,monodetr,monocop}/predict.py; do
  $PY -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" || { echo "!! 语法错误：$f"; bad=1; }
done
if [ -z "$DRYRUN" ]; then
  grep -q "pin_memory=False" "$CF/monocd/train.py" || { echo "!! monocd/train.py 未改成功"; bad=1; }
  grep -q "default=0" "$CF/monocd/train.py" || { echo "!! monocd/train.py workers 未改成功"; bad=1; }
  for m in gupnet monodgp monodetr monocop; do grep -E "add_argument.*--split" "$CF/$m/predict.py" | grep -q "choices" && { echo "!! $m/predict.py 未改成功"; bad=1; }; done
fi
[ $bad = 0 ] || { echo "!! 检查没通过，已停止。把本屏输出发给我们"; exit 1; }
echo "   通过"

say "C. 焦距统一的数据"
D="$P/data/cmpfull"
if [ -f "$D/outdoor_test_fu/labels.pkl" ] && [ -f "$D/outdoor_train_fu/labels.pkl" ] && [ -f "$D/pcmd_teacher_pseudo_outdoor_train_fu.pkl" ]; then
  echo "   已齐全"
else
  echo "   不全，运行 scripts/focal_unify.py（几分钟）"; run "$PY scripts/focal_unify.py"
fi

say "D. MonoCD 冒烟测试（5 步，单独输出目录，不进 runs/）"
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1; exit}')
if [ -n "$DRYRUN" ]; then echo "   [只检查] 跳过"; elif [ -z "$G" ]; then echo "   暂无空闲显卡，跳过冒烟测试（不影响后面的步骤）"; else
  T="$P/runs_smoke_1113/monocd"; rm -rf "$T"; mkdir -p "$T"
  set +e; CUDA_VISIBLE_DEVICES=$G bash "$CF/monocd/train.sh" --labels "$D/sim_outdoor_train/labels.pkl" --out "$T" --seed 0 --epochs 1 --batch 2 --max-iters 5 > "$T/smoke.log" 2>&1; rc=$?; set -e
  if [ $rc = 0 ] && ! grep -q "Traceback" "$T/smoke.log"; then echo "   MonoCD 冒烟通过（卡 $G）"; else echo "!! MonoCD 冒烟失败，最后几行："; tail -15 "$T/smoke.log"; echo "!! 已停止，把本屏输出发给我们"; exit 1; fi
fi

say "E. 失败任务重新排队（只清 .fail 和它的 .lock；正在跑的任务不动）"
k=0; shopt -s nullglob
for f in "$Q"/*.fail; do id=$(basename "$f" .fail); run "rm -rf '$Q/$id.fail' '$Q/$id.lock'"; echo "   重新排队：$id"; k=$((k+1)); done
echo "   共 $k 个"

say "F. 按论文急需程度重排 scripts/tasks.txt（任务一个不少，只改顺序）"
run "cp '$P/scripts/tasks.txt' '$P/scripts/tasks.txt.bak_1113fix'"
[ -n "$DRYRUN" ] || PYTHONIOENCODING=utf-8 $PY - "$P/scripts/tasks.txt" <<'PYEOF'
import sys
p = sys.argv[1]; lines = open(p, encoding='utf-8').read().splitlines()
def key(l):
    s = l.strip()
    if s.startswith('#SKIP1113 '): s = s[len('#SKIP1113 '):]   # 上一个脚本跳过的行：恢复（用户要求全部继续）
    elif s.startswith('#'): return None
    f = s.split('#')[0].split(); return tuple(f[:5]) if len(f) >= 5 else None
PRIO = ['monocd 0 out sim both', 'monocd 0 out kd anti', 'monocd 0 out st anti',            # 主表 AntiUAV3D 缺的 MonoCD 一行
        'gupnet 0 out sim both', 'monodgp 0 out sim both', 'monodetr 0 out sim both', 'monocop 0 out sim both',
        'gupnet 0 out kd anti', 'gupnet 0 out st anti', 'monodgp 0 out kd anti', 'monodgp 0 out st anti',
        'monodetr 0 out kd anti', 'monodetr 0 out st anti', 'monocop 0 out kd anti', 'monocop 0 out st anti',
        'monodgp 0 in st mav', 'monocop 0 in st mav', 'gupnet 0 in st mav', 'gupnet 0 in kd mav', 'monodgp 0 in kd mav',
        'monodetr 0 in sim both', 'monocop 0 in sim both', 'monodetr 0 in kd mav', 'monocop 0 in kd mav', 'monodetr 0 in st mav']
tasks, order = {}, []
for l in lines:
    k = key(l)
    if k is None or k in tasks: continue
    tasks[k] = l.strip()[len('#SKIP1113 '):] if l.strip().startswith('#SKIP1113 ') else l; order.append(k)
head = [l for l in lines if l.lstrip().startswith('#') and not l.strip().startswith('#SKIP1113 ') and not l.lstrip().startswith('# ====')]
out = head + ['# ==== 11-13 fix_all_1113.sh：先补主表缺的，再跑其余；全部任务都保留 ====']
used = set()
for s in PRIO:
    k = tuple(s.split())
    if k in tasks: out.append(tasks[k]); used.add(k)
out += [tasks[k] for k in order if k not in used]
open(p, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('   重排 %d 个任务' % len(order))
PYEOF

say "G. 在空闲显卡上开工"
IDLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{printf "%s ", $1}')
if [ -n "$DRYRUN" ]; then echo "   [只检查] 空闲显卡：${IDLE:-无}"
elif [ -n "$IDLE" ]; then GPUS="$IDLE" bash scripts/run_all.sh; else echo "   暂无空闲显卡：正在跑的工人跑完当前任务后会自动接新任务"; fi

say "各任务状态"
for k in $(grep -v '^\s*#' scripts/tasks.txt | awk 'NF>=5{print $1"_"$3"_"$4"_"$5"_s"$2}'); do
  s="排队"; [ -d "$Q/$k.lock" ] && s="进行中"; [ -f "$Q/$k.fail" ] && s="失败"; [ -f "$Q/$k.done" ] && s="完成"
  printf '   %-32s %s\n' "$k" "$s"
done
echo; echo "完成。之后看进度：cat runs/STATUS.txt；全部跑完后：bash scripts/collect_results.sh，把 results_to_send_*.zip 发回来。"
