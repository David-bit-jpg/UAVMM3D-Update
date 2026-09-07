#!/usr/bin/env bash
# 阶段 5c：自训练的「从零」对照重做。
# 原 C_p05 是 2026-09-05 训的，早于 mav6d.yaml 加入按域归一化（NORM_MEAN/STD），评测时要 --no-norm；
# 阶段 5b 用现行（归一化）流程给它打伪标签，14k 帧里只检出 1.3k 且中位误差 0.83 m——ST_C_p05 无效。
# 这里在现行流程下从零重训 5% 档（Cn_p05，同协议 40 轮、间隔 20），评测，再做自训练 ST_Cn_p05。
# 与 run_night_phase5b.sh 错开：Cn_p05 训练与 ST_S1MT_p05 并行（共 2 个作业），其后步骤等 5b 的 PHASE5B DONE。
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
ev() {
  local tag=$1
  [ -f "$JS/$tag.json" ] && return
  local ck; ck=$(last_ckpt $OUT/mav6d/centerdet/$tag)
  [ -n "$ck" ] || { say "eval $tag: 无 ckpt"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
}

if [ -z "$(last_ckpt $OUT/mav6d/centerdet/Cn_p05)" ]; then
  say "START Cn_p05（现行归一化流程下的从零 5% 对照）"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs 40 --extra_tag Cn_p05 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train 20 > "$LOG/Cn_p05.log" 2>&1
  say "DONE  Cn_p05 rc=$?"
fi
until grep -q "PHASE5B DONE" "$LOG/night_mtkd.log"; do sleep 60; done
ev Cn_p05
CK=$(last_ckpt $OUT/mav6d/centerdet/Cn_p05)
[ -n "$CK" ] || { say "Cn_p05 没有 ckpt，退出"; exit 1; }
if [ ! -f "E:/MAV6D/phantom4/split_pseudo_Cn_p05/stats.json" ]; then
  say "打伪标签 Cn_p05"
  "$PY" mav6d_pseudo_label.py --ckpt "$CK" --tag Cn_p05 --suffix pseudo_Cn_p05 --interval 20 --score "$SCORE" --workers 2 > "$LOG/pseudo_Cn_p05.log" 2>&1
  say "伪标签 Cn_p05: $(grep -o '"n_pseudo": [0-9]*\|"pseudo_pos_median": [0-9.]*\|"pseudo_ang_median": [0-9.]*' $LOG/pseudo_Cn_p05.log | tr '\n' ' ')"
fi
if [ -z "$(last_ckpt $OUT/mav6d/centerdet/ST_Cn_p05)" ]; then
  say "START ST_Cn_p05"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$EP" --extra_tag ST_Cn_p05 --pretrained_model "$CK" \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.LABEL_DIR labels_pseudo_Cn_p05 DATA_CONFIG.SPLIT_DIR split_pseudo_Cn_p05 \
            DATA_CONFIG.SAMPLED_INTERVAL.train 1 > "$LOG/ST_Cn_p05.log" 2>&1
  say "DONE  ST_Cn_p05 rc=$?"
fi
ev ST_Cn_p05
"$PY" - << 'PYEOF' 2>&1 | tee -a "$LOG/night_mtkd.log"
import json, os
d = 'E:/Open3DUAVDet/output/bench_json'
def r(t):
    p = os.path.join(d, t + '.json')
    return json.load(open(p)) if os.path.exists(p) else None
f = lambda x: ('%.3f / %.1f / %.3f' % (x['pos_median'], x['ang_median'], x['acc_0.2'])) if x else '—'
print('自训练汇总（位置中位 m / 角度中位 ° / acc@0.2）：')
for base in ('S1MT_p05', 'Cn_p05'):
    print('  %-10s %s  ->  ST  %s' % (base, f(r(base)), f(r('ST_' + base))))
PYEOF
say "PHASE5C DONE"
