#!/bin/bash
# 用户 2026-09-15 22:3x：D 盘原始数据改完后，用 tiny 纯仿真数据集跑一下看效果（E 盘继续改）。
# tiny 缓存直接从【已就地修正】的 D:/data_collect 建（与 indoor8jpg 同列表同参数，只是每 30 帧取 1）。
# 对照：TINY_fix = 修正后标签；TINY_old = 同一缓存的逆变换旧标签。其余完全相同，纯仿真训练，只评 MAV6D 真实 test（带 ADS）。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/tiny_nosex
JS=/e/Open3DUAVDet/output/camnorm/simfix/json
CACHE=E:/mmcache/tiny_nosex
M=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_indoor8_mz
mkdir -p $LOG $JS
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG/driver.log; }
alive() { powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { 'y' } else { 'n' }" | tr -d '\r'; }

until grep -q "after_fix_D done" /e/Open3DUAVDet/output/label_fix/after_fix_D.log 2>/dev/null; do sleep 60; done
say "D 盘修正与索引替换已完成：$(grep -c '与 index_nose 对比' /e/Open3DUAVDet/output/label_fix/after_fix_D.log) 个抽查序列"

for sp in train test; do
  if [ ! -f "$CACHE/$sp/index.pkl" ]; then
    say "建 tiny 缓存 $sp（从已修正 D 盘，每 30 帧取 1）"
    $PY build_mm_cache.py --list cfgs/subsets/indoor8/near_$sp.txt --split $sp --root D:/data_collect --out $CACHE \
        --every 30 --max-label-range 40 --workers 8 --rgb-only --intrinsic auto --store jpeg > $LOG/cache_$sp.log 2>&1
    say "  rc=$? $(grep -E '^完成' $LOG/cache_$sp.log)"
  fi
done
[ -f "$CACHE/test/vis_score.npy" ] || $PY mm_vis_score.py --cache $CACHE --splits train test > $LOG/vis.log 2>&1
$PY annot_audit/make_labelx_index.py --cache $CACHE > $LOG/labelx_index.log 2>&1
say "对照索引：$(tr '\n' ' ' < $LOG/labelx_index.log)"

while [ "$(alive 15456)" = "y" ] || [ "$(alive 3684)" = "y" ]; do sleep 60; done
say "S2/S3 训练已结束，开始 tiny 对照训练"

train_arm() {  # $1 tag  $2 index file
  $PY train.py --cfg_file cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml --batch_size 8 --workers 2 --fix_random_seed --seed 0 \
     --cudnn_benchmark --use_amp --epochs 8 --extra_tag $1 --max_ckpt_save_num 1 --logger_iter_interval 200 \
     --val_split test --val_interval 1 --val_every 2 --val_match greedy --skip_test_eval \
     --set DATA_CONFIG.DATA_PATH $CACHE DATA_CONFIG.INDEX_FILE $2 DATA_CONFIG.TRAIN_REPEAT 4 > $LOG/$1.train.log 2>&1
  say "$1 train exit $?"
}
( train_arm TINY_fix index.pkl ) &
P1=$!
sleep 60
( train_arm TINY_old index_labelx.pkl ) &
P2=$!
wait $P1 $P2
for t in TINY_fix TINY_old; do
  ck=$M/$t/ckpt/best.pth
  [ -f "$ck" ] || { say "$t 没有 best.pth"; continue; }
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $ck --tag Z_$t --split test --workers 0 \
      --json $JS/Z_${t}_test.json --ads > $LOG/Z_${t}_test.log 2>&1
  say "Z_$t 真实 test 评测 exit $?"
done
$PY report_simfix.py > $LOG/report.txt 2>&1
say "TINY ALL DONE"
