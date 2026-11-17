#!/usr/bin/env bash
# 11-16 补丁：GUPNet + DANN（对抗特征对齐）在 RealDrone3D（室外，打分包里叫 outdoor / anti）上的一行。【优先】
# 从服务器已训好的 GUPNet 室外仿真模型 runs/gupnet_out_sim_s0 出发，和 MAV6D 上已有的 GUPNet + DANN 完全同一做法：
#   域判别器接在颈部特征图上，经梯度反转层；每步 仿真批 = 真值检测损失 + 域损失，真实批 = 只有域损失（真实训练图不带任何标签）；
#   40 轮 × 每轮 3712 个仿真样本，批 8，恒定学习率 3.125e-5（= 仿真训练学习率 × 0.1），BN 冻结，种子 0，取最后一轮，不挑存档。
# 不改 run.sh、不改任务队列，独立运行；断了重跑同一命令会从最新存档接着训；显存不够自动把批减半（会写进日志）。
# 输出：runs/gupnet_out_dann_anti_s0/（pred_outdoor_test.pkl + train.log），之后照常 bash scripts/collect_results.sh 打包发回即可。
# 用法（在实验包根目录）：
#   bash fix_1116.sh [实验包根目录]                 自动挑一张最空的显卡，后台运行
#   CUDA_VISIBLE_DEVICES=3 bash fix_1116.sh ...     指定显卡
#   DRYRUN=1 bash fix_1116.sh ...                   只检查，不训练
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
P="${1:-/userhome/public/zoulk/S2MUAV3D_pack/UAV3D_baselines_pack}"; P="$(cd "$P" && pwd)"
[ -f "$P/scripts/env.sh" ] || { echo "!! $P 不是实验包根目录（找不到 scripts/env.sh）"; exit 1; }
cd "$P"; source scripts/env.sh
T="$CF/gupnet/code/tools"; SRC="$RUNS/gupnet_out_sim_s0/checkpoints/checkpoint_final.pth"; O="$RUNS/gupnet_out_dann_anti_s0"
say() { echo "== $1"; }

say "检查"
[ -f "$SRC" ] || { echo "!! 找不到室外仿真 GUPNet 模型 $SRC（需要先有 gupnet 0 out sim both 的结果）"; exit 1; }
for f in "$DATA/sim_outdoor_train/labels.pkl" "$DATA/outdoor_train/labels.pkl" "$DATA/outdoor_test/labels.pkl" "$CF/gupnet/predict.py"; do
  [ -f "$f" ] || { echo "!! 缺少 $f"; exit 1; }; done
$PY -c "
import torch
try: ck=torch.load('$SRC', map_location='cpu', weights_only=False)
except TypeError: ck=torch.load('$SRC', map_location='cpu')
assert 'mean_size' in ck, 'checkpoint 里没有 mean_size'; print('   源模型可用，mean_size =', ck['mean_size'])"
if [ -f "$T/uda_train.py" ] && ! cmp -s "$T/uda_train.py" "$HERE/uda_train.py"; then cp "$T/uda_train.py" "$T/uda_train.py.bak_1116"; fi
cp "$HERE/uda_train.py" "$T/uda_train.py"
echo "   通过（uda_train.py 已放到 $T）"
[ -n "$DRYRUN" ] && { echo "[只检查] 结束"; exit 0; }

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
fi
say "在显卡 $CUDA_VISIBLE_DEVICES 上后台运行，日志：$O/train.log"
mkdir -p "$O"
nohup bash -c '
  set -e; source scripts/env.sh; O='"$O"'; SRC='"$SRC"'; T='"$T"'
  export GUPNET_NO_PRETRAIN=1
  if [ ! -f "$O/checkpoints/checkpoint_final.pth" ]; then
    B=8
    for try in 0 1 2 3; do
      echo "[$(date "+%F %T")] DANN 训练（批 $B，第 $((try+1)) 次）" >> "$O/train.log"
      set +e
      $PY "$T/uda_train.py" --method dann --init "$SRC" --src-labels "$DATA/sim_outdoor_train/labels.pkl" \
        --tgt-labels "$DATA/outdoor_train/labels.pkl" --out "$O" --seed 0 --epochs 40 --epoch-size 3712 \
        --batch-size $B --workers 0 --resume auto >> "$O/train.log" 2>&1
      rc=$?; set -e
      [ $rc = 0 ] && [ -f "$O/checkpoints/checkpoint_final.pth" ] && break
      if [ $rc = 3 ] && [ $B -gt 1 ]; then B=$((B / 2)); echo "显存不够，批大小减到 $B，从存档续训（与 MAV6D 那次的配方不同，要在论文里注明）" >> "$O/train.log"; continue; fi
      echo "训练失败 rc=$rc，看 $O/train.log" >> "$O/train.log"; exit 1
    done
  fi
  [ -f "$O/pred_outdoor_test.pkl" ] || (cd "$CF/gupnet" && $PY predict.py --ckpt "$O/checkpoints/checkpoint_final.pth" --split outdoor_test \
      --out "$O/pred_outdoor_test.pkl" --thresh 0 >> "$O/train.log" 2>&1)
  bash scripts/collect_results.sh >> "$O/train.log" 2>&1 || true
  echo "[$(date "+%F %T")] 完成：GUPNet + DANN（室外）。结果已自动打包成实验包根目录下最新的 results_to_send_*.zip，发回即可。" >> "$O/train.log"
' > "$O/run_1116.out" 2>&1 &
sleep 30
tail -5 "$O/train.log" 2>/dev/null || true
echo
echo "已在后台开始。看进度：tail -f $O/train.log"
echo "预计单卡约 4–5 小时。跑完会自动打包成 results_to_send_*.zip，发回即可。"
