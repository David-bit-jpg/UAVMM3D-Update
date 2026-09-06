#!/usr/bin/env bash
# 阶段 5b：半监督迁移（自训练）。保留式微调（RM0）在 5% 档明显变差（0.200 vs 0.165），说明仿真特征必须被改写；
# 换成用真实【无标签】帧：拿 5% 档微调好的权重给其余 95% 训练帧打伪标签（tools/mav6d_pseudo_label.py），
# 再从该权重继续训（预算帧真标签 + 伪标签帧），评同一测试集。对两个起点各做一次：
#   BASE1 = MT/S1/S1MT 里 5% 档位置最好的（我们的预训练线），BASE2 = C_p05（从零，验证增益是否独立于预训练）。
# 前置：run_night_v2.sh 已 ALL DONE，且 mav6d_det_dataset.py 已打 LABEL_DIR/SPLIT_DIR 补丁（无训练进程时打）。
# 用法：bash tools/run_night_phase5b.sh [epochs=3] [score=0.3] [base=S1MT]
set -u
EP=${1:-3}
SCORE=${2:-0.3}
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/night_mtkd.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
grep -q "split_dir_name" /e/Open3DUAVDet/uavdet3d/datasets/mav6d/mav6d_det_dataset.py || { say "数据集还没打 LABEL_DIR 补丁，退出"; exit 1; }

best=$("$PY" - << 'PYEOF'
import json, os
d = 'E:/Open3DUAVDet/output/bench_json'
best = None
for arm in ('MT', 'S1', 'S1MT'):
    p = os.path.join(d, arm + '_p05.json')
    if os.path.exists(p):
        v = json.load(open(p))['pos_median']
        if best is None or v < best[1]:
            best = (arm, v)
print(best[0] if best else '')
PYEOF
)
[ -n "${3:-}" ] && best=$3        # 第 3 个参数可指定起点臂（例如 S1MT：10% 档最好、也是完整方法）
[ -n "$best" ] || { say "阶段 5b：没有 MT/S1/S1MT 的 p05 结果，退出"; exit 1; }
say "== 阶段 5b 自训练：起点 ${best}_p05 与 C_p05，伪标签置信度 >= $SCORE，续训 $EP 轮"

st() {  # base_tag
  local base=$1 tag=ST_$1 ck
  ck=$(last_ckpt $OUT/mav6d/centerdet/$base)
  [ -n "$ck" ] || { say "没有 $base 的 ckpt"; return; }
  if [ ! -f "E:/MAV6D/phantom4/split_pseudo_$base/stats.json" ]; then
    say "打伪标签 $base"
    "$PY" mav6d_pseudo_label.py --ckpt "$ck" --tag "$base" --suffix "pseudo_$base" --interval 20 --score "$SCORE" --workers 2 > "$LOG/pseudo_$base.log" 2>&1
    say "伪标签 $base: $(grep -o '"n_pseudo": [0-9]*\|"pseudo_pos_median": [0-9.]*\|"pseudo_ang_median": [0-9.]*' $LOG/pseudo_$base.log | tr '\n' ' ')"
  fi
  [ -n "$(last_ckpt $OUT/mav6d/centerdet/$tag)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$EP" --extra_tag "$tag" --pretrained_model "$ck" \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.LABEL_DIR "labels_pseudo_$base" DATA_CONFIG.SPLIT_DIR "split_pseudo_$base" \
            DATA_CONFIG.SAMPLED_INTERVAL.train 1 > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
ev() {
  local tag=$1
  [ -f "$JS/$tag.json" ] && return
  local ck; ck=$(last_ckpt $OUT/mav6d/centerdet/$tag)
  [ -n "$ck" ] || { say "eval $tag: 无 ckpt"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
}
# 伪标签推理串行（各一个进程），两次训练并行
st ${best}_p05 &
sleep 5
until [ -f "E:/MAV6D/phantom4/split_pseudo_${best}_p05/stats.json" ]; do sleep 30; done
st C_p05 &
wait
ev ST_${best}_p05; ev ST_C_p05
"$PY" - "$best" << 'PYEOF' 2>&1 | tee -a "$LOG/night_mtkd.log"
import json, os, sys
d = 'E:/Open3DUAVDet/output/bench_json'
best = sys.argv[1]
def r(t):
    p = os.path.join(d, t + '.json')
    return json.load(open(p)) if os.path.exists(p) else None
print('自训练（5% 真标签 + 其余训练帧伪标签，续训）：位置中位 m / 角度中位 ° / acc@0.2')
for base in (best + '_p05', 'C_p05'):
    a, b = r(base), r('ST_' + base)
    f = lambda x: ('%.3f / %.1f / %.3f' % (x['pos_median'], x['ang_median'], x['acc_0.2'])) if x else '—'
    print('  %-10s %s  ->  ST  %s' % (base, f(a), f(b)))
PYEOF
say "PHASE5B DONE"
