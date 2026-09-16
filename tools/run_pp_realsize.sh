#!/bin/bash
# 采集其余场景的同时，用已采完的 PowerPlant（真机尺寸 + 机头 +x）训练纯仿真模型 P1，只评 MAV6D 真实 test。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/pp_realsize
JS=$LOG/json
CACHE=E:/mmcache/pp_realsize
mkdir -p $LOG $JS
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG/driver.log; }
for sp in train test; do
  if [ ! -f "$CACHE/$sp/index.pkl" ]; then
    say "建缓存 $sp"
    $PY build_mm_cache.py --list cfgs/subsets/pp_realsize/near_$sp.txt --split $sp --root D:/data_collect --out $CACHE \
        --every 3 --max-label-range 40 --workers 8 --rgb-only --intrinsic auto --store jpeg > $LOG/cache_$sp.log 2>&1
    say "  rc=$? $(grep -E '^完成' $LOG/cache_$sp.log)"
  fi
done
[ -f "$CACHE/test/vis_score.npy" ] || { $PY mm_vis_score.py --cache $CACHE --splits train test > $LOG/vis.log 2>&1; say "vis rc=$?"; }
say "训练 P1"
$PY train.py --cfg_file cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml --batch_size 8 --workers 2 --fix_random_seed --seed 0 \
   --cudnn_benchmark --use_amp --epochs 24 --extra_tag P1 --max_ckpt_save_num 1 --logger_iter_interval 200 \
   --val_split test --val_interval 5 --val_every 2 --val_match greedy --skip_test_eval \
   --set DATA_CONFIG.TRAIN_REPEAT 4 > $LOG/P1.train.log 2>&1
say "P1 train exit $?"
CK=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_pp_realsize/P1/ckpt/best.pth
$PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $CK --tag Z_P1 --split test --workers 0 --json $JS/Z_P1_test.json --ads > $LOG/Z_P1_test.log 2>&1
say "Z_P1 真实 test 评测 exit $?"
say "ALL DONE"
