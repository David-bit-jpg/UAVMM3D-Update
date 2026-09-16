#!/usr/bin/env bash
# 新室内仿真 RGB -> MAV6D 迁移（用户 2026-09-13 批准的 GPU 训练）。
# 与 C / B / M0 / MT / S1 / S1MT 严格同协议：batch 8、--pretrained_skip hm、
# --pretrained_src_max_dis 40、p01/p05/p10 = SAMPLED_INTERVAL 100/20/10、epochs 80/40/30。
# 每次最多 2 个训练作业（Windows 上 >=3 个 spawn dataloader 会锁死）。
set -u
PY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
cd /e/Open3DUAVDet/tools || exit 1

SUB=cfgs/subsets/indoor8
CACHE=E:/mmcache/indoor8
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MM=cfgs/models/uavdet_3d/mmcache
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
mkdir -p "$LOG" "$JS"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/indoor_transfer.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

# ---------- 0. 等子集清单 ----------
until [ -s "$SUB/near_train.txt" ] && [ -s "$SUB/near_test.txt" ]; do sleep 60; done
say "子集就绪: train $(grep -vc '^#' $SUB/near_train.txt) 帧, test $(grep -vc '^#' $SUB/near_test.txt) 帧"

# ---------- 1. 建缓存 ----------
for sp in train test; do
  if [ ! -f "$CACHE/$sp/rgb.npy" ]; then
    say "建缓存 $sp"
    "$PY" build_mm_cache.py --list "$SUB/near_$sp.txt" --split "$sp" \
        --root D:/data_collect --out "$CACHE" --every 3 --max-label-range 40 --workers 8 \
        > "$LOG/indoor_cache_$sp.log" 2>&1
    say "建缓存 $sp rc=$? -> $(ls -la $CACHE/$sp/rgb.npy 2>/dev/null | awk '{print $5}')"
  fi
done
[ -f "$CACHE/train/rgb.npy" ] || { say "缓存没建出来，停"; exit 1; }

# ---------- 2. 可见度分数（MIN_RGB_VIS 过滤要用） ----------
if [ ! -f "$CACHE/train/vis_score.npy" ] || [ ! -f "$CACHE/test/vis_score.npy" ]; then
  say "算可见度"
  "$PY" mm_vis_score.py --cache "$CACHE" --splits train test > "$LOG/indoor_vis.log" 2>&1
  say "  rc=$?"
fi

# ---------- 3. 本域归一化统计量 ----------
if [ ! -f "$CACHE/norm.txt" ]; then
  say "算归一化统计量"
  "$PY" - <<'PYEOF' > "$CACHE/norm.txt" 2>"$LOG/indoor_norm.log"
import numpy as np
# 必须逐帧用 float64 累加：一次 np.mean 在 2.6 亿个 float32 上会丢精度，
# 采样越多越离谱（600 帧时三通道会塌成同一个值 0.1896，真值约 0.35/0.40/0.45）。
a = np.load(r'E:/mmcache/indoor8/train/rgb.npy', mmap_mode='r')
idx = np.linspace(0, len(a) - 1, min(600, len(a))).astype(int)
s1 = np.zeros(3, np.float64); s2 = np.zeros(3, np.float64); n = 0
for i in idx:
    f = a[i].astype(np.float64) / 255.0
    v = f.reshape(-1, 3)
    s1 += v.sum(0); s2 += (v * v).sum(0); n += v.shape[0]
mean = s1 / n
std = np.sqrt(np.maximum(s2 / n - mean * mean, 0))
print('NORM_MEAN: [%.4f, %.4f, %.4f]' % tuple(mean))
print('NORM_STD: [%.4f, %.4f, %.4f]' % tuple(std))
print('frames %d' % len(a))
PYEOF
fi
cat "$CACHE/norm.txt" | tee -a "$LOG/indoor_transfer.log"
NM=$(grep NORM_MEAN "$CACHE/norm.txt" | cut -d' ' -f2-)
NS=$(grep NORM_STD  "$CACHE/norm.txt" | cut -d' ' -f2-)

# ---------- 4. 配置 ----------
DSCFG=cfgs/dataset_configs/uavdet_3d/mmcache_indoor.yaml
cat > "$DSCFG" <<EOF
# 新室内仿真缓存（tools/build_mm_cache.py，源 = D:/data_collect 四个场景）。
# 与 mmcache.yaml 只差 DATA_PATH 和本域归一化统计量。
_BASE_CONFIG_: cfgs/dataset_configs/uavdet_3d/mmcache.yaml
DATA_PATH: '$CACHE'
NORM_MEAN: $NM
NORM_STD: $NS
EOF
MDCFG=$MM/student_rgb_indoor.yaml
cat > "$MDCFG" <<EOF
# 纯 RGB 检测器，训在新室内仿真数据上 —— 迁 MAV6D 的 I0 臂权重来源。
# 网络结构与 student_rgb.yaml 完全一致，只换数据。
_BASE_CONFIG_: cfgs/models/uavdet_3d/mmcache/student_rgb.yaml
DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/uavdet_3d/mmcache_indoor.yaml
    MAX_OBJ_PER_SAMPLE: 7
    MODALITIES: ['rgb']
EOF
say "配置已写: $DSCFG / $MDCFG"

# ---------- 5. 预训练 I0 ----------
I0_CK=$(last_ckpt $OUT/mmcache/student_rgb_indoor/I0b)
if [ -z "$I0_CK" ]; then
  say "START I0 预训练（12 轮）"
  "$PY" train.py --cfg_file "$MDCFG" --batch_size 8 --workers 2 --fix_random_seed \
      --max_ckpt_save_num 2 --epochs 12 --logger_iter_interval 100 --extra_tag I0b \
      > "$LOG/I0.log" 2>&1
  say "DONE  I0 rc=$?"
  I0_CK=$(last_ckpt $OUT/mmcache/student_rgb_indoor/I0b)
fi
[ -n "$I0_CK" ] || { say "I0 没产出 ckpt，看 $LOG/I0.log"; exit 1; }
say "I0=$I0_CK"

# ---------- 6. 迁移 ----------
xfer() {  # tag epochs interval
  local tag=$1 ep=$2 iv=$3
  [ -n "$(last_ckpt $OUT/mav6d/centerdet/$tag)" ] && { say "SKIP $tag"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed \
      --max_ckpt_save_num 2 --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --pretrained_model "$I0_CK" --pretrained_skip hm --pretrained_src_max_dis 40 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" \
      > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
xfer I0b_p01 80 100 &
xfer I0b_p05 40 20 &
wait
xfer I0b_p10 30 10 &
wait

# ---------- 7. 评测 ----------
for tag in I0b_p01 I0b_p05 I0b_p10; do
  [ -f "$JS/$tag.json" ] && continue
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差' $LOG/eval_$tag.log | head -1)"
done
"$PY" eval_on_mav6d.py --ckpt "$I0_CK" --tag ZI0b --decode-max-dis 40 --json "$JS/ZI0b.json" \
    > "$LOG/eval_ZI0b.log" 2>&1
say "零样本 I0: $(grep -E '位置误差' $LOG/eval_ZI0b.log | head -1)"

# ---------- 8. 对比 ----------
say "===== 对比（MAV6D 位置误差中位，米）====="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG/indoor_transfer.log"
import json, os
JS = r'E:/Open3DUAVDet/output/bench_json'
arms = [('C', '从零(真实)'), ('M0', '贴图增广'), ('MT', '多任务'),
        ('S1', '蒸馏'), ('S1MT', '蒸馏+多任务'), ('I0', '新室内(归一化错)'), ('I0b', '新室内(修正)')]
def get(tag, k):
    p = os.path.join(JS, tag + '.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8')).get(k)
for metric, unit in (('pos_median', '位置中位 m'), ('ang_median', '角度中位 deg'), ('acc_0.2', 'acc@0.2')):
    print()
    print('== %s ==' % unit)
    print('%-8s %-14s %9s %9s %9s' % ('臂', '说明', 'p01', 'p05', 'p10'))
    for a, name in arms:
        vals = []
        for f in ('p01', 'p05', 'p10'):
            v = get('%s_%s' % (a, f), metric)
            vals.append('%9.3f' % v if isinstance(v, (int, float)) else '%9s' % '-')
        print('%-8s %-14s %s' % (a, name, ''.join(vals)))
print()
print('== 零样本（不微调）位置中位 m ==')
for t in ('A_sim_only', 'ZM0_mix', 'ZI0', 'ZI0b'):
    v = get(t, 'pos_median')
    print('  %-12s %s' % (t, ('%.3f' % v) if isinstance(v, (int, float)) else '-'))
PYEOF
say "ALL DONE"
