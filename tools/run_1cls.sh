#!/usr/bin/env bash
# 训练不分机型（热力图单通道 'drone'）vs 原来分 mavic2/phantom4 的对照。用户 2026-09-14 要求。
# 两组都用当前流程（按域归一化），协议与 I0b / C 完全一致，只换 centerdet_1cls.yaml。
#   I1_* = 新室内仿真 I0b 预训练 + 不分机型微调
#   C1_* = 从零 + 不分机型
# 每次最多 2 个训练作业。
set -u
PY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
cd /e/Open3DUAVDet/tools || exit 1
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
CFG=cfgs/models/uavdet_3d/mav6d/centerdet_1cls.yaml
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run_1cls.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

# 等 ADS 评测跑完（GPU 上别叠 3 个进程）
until [ -f /e/Open3DUAVDet/output/ads_metrics/I0b_p10/indoor/report.txt ]; do sleep 60; done
sleep 30
say "ADS 评测已完成，开始不分机型实验"

I0B=$(last_ckpt $OUT/mmcache/student_rgb_indoor/I0b)
[ -n "$I0B" ] || { say "找不到 I0b 预训练权重"; exit 1; }
say "I0b = $I0B"

run() {  # tag epochs interval pretrained(可空)
  local tag=$1 ep=$2 iv=$3 pre=${4:-}
  [ -n "$(last_ckpt $OUT/mav6d/centerdet_1cls/$tag)" ] && { say "SKIP $tag"; return; }
  say "START $tag"
  local extra=()
  [ -n "$pre" ] && extra=(--pretrained_model "$pre" --pretrained_skip hm --pretrained_src_max_dis 40)
  "$PY" train.py --cfg_file $CFG --batch_size 8 --workers 2 --fix_random_seed \
      --max_ckpt_save_num 2 --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      "${extra[@]}" \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" \
      > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}

run I1_p01 80 100 "$I0B" &
run C1_p01 80 100 &
wait
run I1_p05 40 20 "$I0B" &
run C1_p05 40 20 &
wait
run I1_p10 30 10 "$I0B" &
run C1_p10 30 10 &
wait

for tag in I1_p01 I1_p05 I1_p10 C1_p01 C1_p05 C1_p10; do
  [ -f "$JS/$tag.json" ] && continue
  ck=$(last_ckpt $OUT/mav6d/centerdet_1cls/$tag); [ -n "$ck" ] || { say "缺 $tag"; continue; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差' $LOG/eval_$tag.log | head -1)"
done

say "===== 分机型 vs 不分机型 ====="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG/run_1cls.log"
import json, os
JS = r'E:/Open3DUAVDet/output/bench_json'
def g(t, k):
    p = os.path.join(JS, t + '.json')
    return json.load(open(p, encoding='utf-8')).get(k) if os.path.exists(p) else None
arms = [('C',  '纯真实·分机型(旧代)'), ('C1', '纯真实·不分机型'),
        ('I0b', '迁移·分机型'),       ('I1', '迁移·不分机型')]
for k, lab in (('pos_median', '位置误差中位 m'), ('z_median', '深度误差中位 m'),
               ('ang_median', '角度误差中位 deg'), ('acc_0.2', '位置<0.2m 占比')):
    print()
    print('== %s ==' % lab)
    print('%-20s %9s %9s %9s' % ('', '1%', '5%', '10%'))
    for a, n in arms:
        vals = [g('%s_%s' % (a, f), k) for f in ('p01', 'p05', 'p10')]
        print('%-20s %s' % (n, ''.join('%9.3f' % v if isinstance(v, (int, float)) else '%9s' % '-' for v in vals)))
PYEOF
say "ALL DONE"
