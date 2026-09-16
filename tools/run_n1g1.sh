#!/bin/bash
# 2026-09-16 两臂并行：
#   N1 = P1b + 回归头加 bias + 真值自洽标准化           （隔离「先验混在特征通路里」这一项）
#   G1 = ImageNet 骨干 + size2d 几何解深度 + bias + 标准化（把今天验证过的三件事叠起来）
# 评测配置必须与训练结构一致，且带【训练域】的标准化常数（解码要用它）。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/n1g1; JS=$LOG/json
mkdir -p $LOG $JS
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG/driver.log; }
run() {   # $1=训练cfg $2=tag $3=stem $4=评测cfg
  $PY train.py --cfg_file $1 --batch_size 8 --workers 2 --fix_random_seed --seed 0 \
     --cudnn_benchmark --use_amp --epochs 24 --extra_tag $2 --max_ckpt_save_num 1 --logger_iter_interval 200 \
     --val_split test --val_interval 5 --val_every 2 --val_match greedy --skip_test_eval \
     --set DATA_CONFIG.TRAIN_REPEAT 4 > $LOG/$2.train.log 2>&1
  say "$2 train exit $?"
  $PY eval_camnorm.py --cfg $4 --ckpt /e/Open3DUAVDet/output/models/uavdet_3d/camnorm/$3/$2/ckpt/best.pth \
     --tag Z_$2 --split test --workers 0 --json $JS/Z_$2_test.json --ads > $LOG/Z_$2_test.log 2>&1
  say "Z_$2 真实 test 评测 exit $?"
}
say "启动 N1 与 G1"
run cfgs/models/uavdet_3d/camnorm/sim_pp_std.yaml N1 sim_pp_std cfgs/models/uavdet_3d/camnorm/mav6d_std_pp.yaml &
A=$!
run cfgs/models/uavdet_3d/camnorm/sim_pp_g1.yaml  G1 sim_pp_g1  cfgs/models/uavdet_3d/camnorm/mav6d_g1.yaml &
B=$!
wait $A $B
say "ALL DONE"
