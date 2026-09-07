#!/usr/bin/env bash
# 接 run_augonly_scale.sh（那次因为 --range-ref 相对路径写错重启过一次；重启时旧的 merge 还在跑，
# 两个 merge 同时写同一个 memmap 报 Errno 22。现在：旧 merge 继续跑，生成 9000 张的三个进程也在跑，
# 本脚本只做后续步骤，不再启动 merge aug_only_3k）。
#   1 等 aug_only_3k/READY -> 训 aug3k（纯 RGB 12 轮）-> MAV6D 零样本
#   2 等三批各 3000 张生成完 -> 打包 12000 -> 合并 aug_only_12k -> 训 aug12k -> 零样本
#   3 汇总
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
  [ -n "$ck" ] || { say "$tag 没有 ckpt（看 $LOG/$tag.log）"; return 1; }
  [ -f "$JS/Z$tag.json" ] && return 0
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "Z$tag" --decode-max-dis 40 --json "$JS/Z$tag.json" > "$LOG/eval_Z$tag.log" 2>&1
  say "零样本 Z$tag: $(grep -E '位置误差 ' $LOG/eval_Z$tag.log | head -1)"
}

say "== 续跑：等 aug_only_3k 合并完成"
until [ -f "E:/data_collect/aug_only_3k/READY" ]; do sleep 60; done
say "aug_only_3k 就绪"
train_eval aug3k E:/data_collect/aug_only_3k

say "等三批生成（每批 3000）"
until [ "$(ls $AUG/samples_b1/*.npz $AUG/samples_b2/*.npz $AUG/samples_b3/*.npz 2>/dev/null | wc -l)" -ge 8900 ]; do sleep 120; done
say "生成完成：b1 $(ls $AUG/samples_b1/*.npz 2>/dev/null | wc -l) / b2 $(ls $AUG/samples_b2/*.npz 2>/dev/null | wc -l) / b3 $(ls $AUG/samples_b3/*.npz 2>/dev/null | wc -l)"

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
[ -f "E:/data_collect/aug_only_12k/READY" ] || { say "aug_only_12k 合并失败"; exit 1; }
train_eval aug12k E:/data_collect/aug_only_12k

"$PY" - << 'PYEOF' 2>&1 | tee -a "$LOG/augonly.log"
import json, os
D = 'E:/Open3DUAVDet/output/bench_json'
def r(t):
    p = os.path.join(D, t + '.json')
    return json.load(open(p)) if os.path.exists(p) else None
print('只用仿真数据训练 -> MAV6D 零样本（不用任何真实帧）：')
print('%-22s %9s %9s %9s %9s %10s' % ('训练集', '位置中位', '角度中位', 'acc@0.5m', 'acc@1m', '检出帧'))
for tag, lab in (('Zaug3k', '3000 生成'), ('Zaug12k', '12000 生成'),
                 ('ZM0_mix', '3000 生成+1436 原版'), ('ZS1_mix', '同上+蒸馏'),
                 ('A_sim_only', '第一版 near15 原版')):
    d = r(tag)
    if d:
        print('%-22s %8.2fm %8.1f° %9.4f %9.4f %6d/%d' % (lab, d['pos_median'], d['ang_median'],
              d.get('acc_0.5', float('nan')), d.get('acc_1', float('nan')), d['n_valid'], d['n_total']))
PYEOF
say "AUGONLY DONE"
