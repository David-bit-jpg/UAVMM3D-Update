#!/usr/bin/env bash
# 阶段 5（接 run_night_mtkd.sh 的 ALL DONE 之后自动跑）：把「保留式微调」（centerdet_retain）用在
# MT / S1 / S1MT 三个预训练里 5% 档位置误差最好的那个上，1%/5%/10% 各跑一遍（R<ARM>_p01/p05/p10），
# 评测，更新曲线与汇总表。RM0（M0 的保留式微调）已在主驱动里。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MAVR=cfgs/models/uavdet_3d/mav6d/centerdet_retain.yaml
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/night_mtkd.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
xfer() {  # tag src_ckpt epochs interval
  local tag=$1 src=$2 ep=$3 iv=$4
  local dir=$OUT/mav6d/centerdet_retain/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $MAVR --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --pretrained_model "$src" --pretrained_skip hm --pretrained_src_max_dis 40 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" MODEL.DISTILL.TEACHER_CKPT "$src" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
ev() {
  local tag=$1
  [ -f "$JS/$tag.json" ] && return
  local ck; ck=$(last_ckpt $OUT/mav6d/centerdet_retain/$tag)
  [ -n "$ck" ] || { say "eval $tag: 无 ckpt"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
}

until grep -q "ALL DONE" "$LOG/night_mtkd.log" 2>/dev/null; do sleep 120; done
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
[ -n "$best" ] || { say "阶段 5：没有 MT/S1/S1MT 的 p05 结果，退出"; exit 1; }
case $best in
  MT)   CK=$(last_ckpt $OUT/mmcache/student_mt_mix/MT_mix);;
  S1)   CK=$(last_ckpt $OUT/mmcache/student_kd_mix/S1_mix);;
  S1MT) CK=$(last_ckpt $OUT/mmcache/student_kdmt_mix/S1MT_mix);;
esac
say "== 阶段 5：5% 档最好的预训练 = $best（$CK），跑保留式微调 R${best}_p01/p05/p10"
xfer R${best}_p01 "$CK" 80 100 & xfer R${best}_p05 "$CK" 40 20 & wait
ev R${best}_p01; ev R${best}_p05
xfer R${best}_p10 "$CK" 30 10 & wait
ev R${best}_p10
ARMS="C:scratch (real only),B:sim near15 pretrain,M0:mix pretrain,MT:mix multi-task,S1:mix KD,S1MT:mix KD+MT,RM0:M0 + retain FT,R${best}:${best} + retain FT"
"$PY" plot_lowdata_curve.py --arms "$ARMS" --fracs p01,p05,p10 --out ../docs/results/lowdata_night.png > "$LOG/plot_night.log" 2>&1
"$PY" plot_lowdata_curve.py --arms "$ARMS" --fracs p01,p05,p10 --metric ang_median --out ../docs/results/lowdata_night_ang.png >> "$LOG/plot_night.log" 2>&1
(cd .. && "$PY" tools/summarize_bench.py --arms C,B,M0,MT,S1,S1MT,RM0,R${best} --fracs p01,p05,p10 --zero ZM0_mix,ZMT_mix,ZS1_mix,ZS1MT_mix --out output/bench_json/summary_night.md) 2>&1 | tee -a "$LOG/night_mtkd.log"
say "PHASE5 DONE"
