#!/usr/bin/env bash
# 用户 2026-09-07：「只用生成的、量大一点的仿真数据，去测在 MAV6D 上的表现」= 零样本，不用任何真实帧。
# 两个数据量点，配方完全相同（纯 RGB CenterDet，12 轮，batch 8），只差训练集大小：
#   Z_aug3k   3000 张交叉贴样本（已生成，E:/data_collect/aug_paste_v1/samples）
#   Z_aug12k  12000 张（再并行生成 9000 张，源帧池与背景池不变，seed 不同）
# 每个缓存的 test split = 100 张原版白天帧（域内 sanity，不进训练）。
# 阶段：A 合并 3k 缓存 -> 训 -> 零样本；同时 B 并行生成 9000 -> 打包 12k -> 训 -> 零样本 -> 汇总。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
AUG=E:/data_collect/aug_paste_v1
SRC=E:/mmcache/paste_src
MM=cfgs/models/uavdet_3d/mmcache
mkdir -p "$LOG" "$JS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/augonly.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

train_eval() {  # tag data_path
  local tag=$1 dp=$2
  local dir=$OUT/mmcache/student_rgb_augonly/$tag
  if [ -z "$(last_ckpt $dir)" ]; then
    say "START $tag  (DATA_PATH=$dp)"
    "$PY" train.py --cfg_file $MM/student_rgb_augonly.yaml --batch_size 8 --workers 2 --fix_random_seed \
        --max_ckpt_save_num 2 --epochs 12 --logger_iter_interval 100 --extra_tag "$tag" \
        --set DATA_CONFIG.DATA_PATH "$dp" > "$LOG/$tag.log" 2>&1
    say "DONE  $tag rc=$?  域内: $(grep -A2 'recall@2m' $LOG/$tag.log | tail -3 | tr -s ' \n' ' ')"
  fi
  local ck; ck=$(last_ckpt $dir)
  [ -n "$ck" ] || { say "$tag 没有 ckpt，跳过评测"; return; }
  [ -f "$JS/Z${tag}.json" ] && return
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "Z$tag" --decode-max-dis 40 --json "$JS/Z$tag.json" > "$LOG/eval_Z$tag.log" 2>&1
  say "零样本 Z$tag: $(grep -E '位置误差 ' $LOG/eval_Z$tag.log | head -1)"
}

# ---------- B（后台，纯 CPU）：再生成 9000 张 ----------
gen() {  # seed outdir n
  [ -d "$2" ] && [ "$(ls $2/*.npz 2>/dev/null | wc -l)" -ge "$3" ] && { say "SKIP 生成 $2（已有）"; return; }
  "$PY" mm_paste_aug.py --cache "$SRC" --erased "${SRC}_erased" --plates "${SRC}_plates" --n "$3" --no-sheets \
      --range-ref docs/results/mav6d_size_px_range_m.npy --tone 0.6 --seed "$1" --out "$2" \
      > "$LOG/gen_$(basename $2).log" 2>&1
  say "生成完成 $2: $(ls $2/*.npz 2>/dev/null | wc -l) 个"
}
for i in 1 2 3; do gen $((1000 + i)) "$AUG/samples_b$i" 3000 & done

# ---------- A：3000 张 ----------
if [ ! -f "E:/data_collect/aug_only_3k/READY" ]; then
  say "合并 aug_only_3k（train = 3000 生成帧，test = 100 原版帧）"
  "$PY" merge_mmcaches.py --out E:/data_collect/aug_only_3k --input "$AUG/cache:0:0:aug" --input "$SRC:5:100:orig" \
      --test-from orig --test-n 100 2>&1 | tail -3 | tee -a "$LOG/augonly.log"
fi
[ -f "E:/data_collect/aug_only_3k/READY" ] || { say "aug_only_3k 合并失败"; exit 1; }
train_eval aug3k E:/data_collect/aug_only_3k

wait        # 等 9000 张生成完
say "生成总数：$(ls $AUG/samples/*.npz $AUG/samples_b1/*.npz $AUG/samples_b2/*.npz $AUG/samples_b3/*.npz 2>/dev/null | wc -l)"

# ---------- 打包 12k ----------
if [ ! -f "$AUG/cache12k/train/index.pkl" ]; then
  say "打包 12000 张 -> $AUG/cache12k"
  "$PY" pack_paste_cache.py --samples "$AUG/samples" --samples "$AUG/samples_b1" --samples "$AUG/samples_b2" --samples "$AUG/samples_b3" \
      --out "$AUG/cache12k" --note "run_augonly_scale.sh: 4 批 x 3000，seed 2026/1001/1002/1003，源帧池与背景池同 aug_paste_v1" \
      2>&1 | tail -3 | tee -a "$LOG/augonly.log"
fi
[ -f "$AUG/cache12k/train/index.pkl" ] || { say "打包失败"; exit 1; }
if [ ! -f "E:/data_collect/aug_only_12k/READY" ]; then
  say "合并 aug_only_12k"
  "$PY" merge_mmcaches.py --out E:/data_collect/aug_only_12k --input "$AUG/cache12k:0:0:aug" --input "$SRC:5:100:orig" \
      --test-from orig --test-n 100 2>&1 | tail -3 | tee -a "$LOG/augonly.log"
fi
train_eval aug12k E:/data_collect/aug_only_12k

# ---------- 汇总 ----------
"$PY" - << 'PYEOF' 2>&1 | tee -a "$LOG/augonly.log"
import json, os
D = 'E:/Open3DUAVDet/output/bench_json'
def r(t):
    p = os.path.join(D, t + '.json')
    return json.load(open(p)) if os.path.exists(p) else None
print('只用生成的仿真数据训练 -> MAV6D 零样本（不用任何真实帧）：')
print('%-12s %10s %8s %9s %9s %8s' % ('训练集', '位置中位', '角度中位', 'acc@0.5m', 'acc@1m', '检出帧'))
for tag, lab in (('Zaug3k', '3000 生成'), ('Zaug12k', '12000 生成'),
                 ('ZM0_mix', '3000 生成+1436 原版'), ('A_sim_only', '第一版 near15 原版')):
    d = r(tag)
    if d:
        print('%-12s %9.2fm %7.1f° %9.4f %9.4f %5d/%d' % (lab, d['pos_median'], d['ang_median'],
              d.get('acc_0.5', float('nan')), d.get('acc_1', float('nan')), d['n_valid'], d['n_total']))
PYEOF
say "AUGONLY DONE"
