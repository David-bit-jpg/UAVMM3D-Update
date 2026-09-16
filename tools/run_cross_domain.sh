#!/usr/bin/env bash
# 训练域 x 测试域交叉评测：等 S1 预训练与 MAV6D 零样本评测完成后，在仿真测试集上评 S1（整幅 / 2.5 倍长焦），出表。
set -u
PY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
cd /e/Open3DUAVDet/tools || exit 1
H=../output/models/uavdet_3d/camnorm/sim_indoor8_mz/S1/val_history.json
until [ -f ../output/camnorm/json/Z_S1_test.json ]; do sleep 30; done
CK=../output/models/uavdet_3d/camnorm/sim_indoor8_mz/S1/ckpt/best.pth
for z in 1 2.5; do
  "$PY" eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml --ckpt "$CK" --tag S1__simtest_z$z \
      --split test --match greedy --workers 0 --json ../output/camnorm/json_simtest/S1__simtest_z$z.json --ads \
      --ads-out E:/Open3DUAVDet/output/camnorm/ads_simtest --set DATA_CONFIG.VAL_ZOOMS "[$z]" \
      > ../output/camnorm/logs/simtest_S1_z$z.log 2>&1
  echo "S1 z=$z rc=$?"
done
"$PY" table_cross_domain.py
