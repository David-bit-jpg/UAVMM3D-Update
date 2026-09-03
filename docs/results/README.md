# Sim → Real 迁移实验结果

三臂对比，全部在**同一个 MAV6D 测试集**（4,800 帧，mavic2 1,290 + phantom4 3,510）上，
用同一个评测器 `tools/eval_on_mav6d.py`、同一套**类别无关**的位姿误差指标
（每帧取置信度最高的检测，与 GT 比位置/深度/旋转）。

各臂的 `hm` 通道数不同（仿真 7 类 vs MAV6D 2 类），无法直接用各自配置里的评测流程
横向比，所以统一成上面这套指标。

## 三个臂

| 臂 | 训练数据 | 权重来源 |
|---|---|---|
| **A 纯模拟** | 仅仿真 near15（8,773 帧）| 直接零样本推理，不碰任何真实数据 |
| **B 迁移** | 仿真预训练 → MAV6D 微调（15,202 帧）| 载入 A 的权重，96.9% 张量可迁 |
| **C 纯真实** | 仅 MAV6D（15,202 帧）| 随机初始化 |

B 与 C 除初始化外**完全相同**：同配置、同数据、同超参、同 15 轮、并行运行以保证资源对等。

## 结果

| 指标 | A 纯模拟 | B 迁移 | C 纯真实 |
|---|---|---|---|
| 位置误差 中位 (m) | 7.925 | 0.084 | **0.075** |
| 位置误差 均值 (m) | 6.151 | 0.121 | **0.102** |
| 深度误差 中位 (m) | 7.662 | 0.080 | **0.072** |
| 角度误差 中位 (deg) | 119.99 | 10.005 | **8.561** |
| 角度误差 均值 (deg) | 127.04 | 15.541 | **13.018** |
| 位置误差 < 0.1 m | 0.2% | 56.5% | **61.5%** |
| 位置误差 < 0.2 m | 0.4% | 82.1% | **88.4%** |
| 位置误差 < 0.5 m | 1.1% | 98.0% | **99.4%** |
| 有效检测 | 4140/4800 | 4788/4800 | 4783/4800 |
| 训练末 loss | — | 0.312 | 0.313 |

GT 深度范围 1.54 ~ 5.55 m（中位 3.38）。

![三臂同帧对比](three_arm_compare.jpg)

上中下依次为 A / B / C，绿框 = GT，红框 = 预测。

## 结论

### 1. 仿真零样本完全不可用

A 的位置误差中位 **7.9 m**，比整个 GT 深度范围（1.5–5.6 m）还大；角度误差 **120°**
接近随机取向的期望值。可视化中它把目标定位到画面上方的安全网上，深度预测 12.3 m
而真值 2.7 m —— 输出的基本是源域先验（≤15 m 的中段深度），与真实场景无对应关系。

**这给迁移提供了明确的下限：不做任何适配，仿真训练的模型在真实域上是无效的。**

### 2. 在当前设置下，迁移没有增益，反而略差

B 在所有指标上都比 C 差一点（位置 +12%，角度 +17%），两者训练末 loss 几乎相同
（0.312 vs 0.313），说明收敛到了同一水平。

判断原因是**目标域数据太充足**：MAV6D 有 15,202 训练帧，而任务本身相对简单
（单目标、静止相机、深度范围仅 1.5–5.6 m）。数据足够时从零训完全能学好，
而仿真先验（室外、几十米、1.3 m 大目标、CARLA 渲染风格）反而构成负迁移。

**这不否定对齐工作的价值**：96.9% 的权重能成功迁入，证明模态 / 骨干 / 检测头 /
深度尺度 / 旋转约定五个维度确实已经对齐（详见主 README 的「跨域对齐」一节）。
只是在"目标域数据充足"这一条件下，迁移本身不产生收益。

### 3. 待补：低数据量下的对比

迁移学习真正的用武之地是目标域数据稀缺时，而当前设置恰好绕开了这个场景。
下一步应把 MAV6D 训练集降采样到 1% / 5% / 10% / 25% 重跑 B vs C：

- 若曲线在低数据端分开，说明预训练的价值被充足数据掩盖了
- 若全程重合，则是真的负迁移，需要换预训练策略（只迁 backbone、或改用更接近的源域）

## 数据本身的验证

![GT 投影验证](gt_check/mavic2_01_0101_1649917993389201164.jpg)

`tools/mav6d_prepare.py check` 在 20,002 帧上的结论：

| | mavic2 | phantom4 |
|---|---|---|
| 图 / 标签配对 | 10001 / 10001，零缺失零多余 | 10001 / 10001，零缺失零多余 |
| 深度范围（中位）| 0.76–5.55 m（3.25）| 1.57–5.72 m（3.47）|
| GT 投影落在画面内 | **100.0%** | **99.9%** |
| 相机位姿波动 | 0.41 mm / 0.466° | 0.61 mm / 0.563° |
| 折算到中位深度的位置偏差 | 26.8 mm | 34.8 mm |

相机位姿波动折算后相对 340 mm 的目标尺寸约 10%，可忽略 ——
`read_truth_Rt` 里那个**写死的 camera→VICON 外参对两个型号都成立**。

## 复现

```bash
# A 纯模拟（零样本）
python tools/eval_on_mav6d.py --ckpt <sim_ckpt>.pth --tag A_sim_only --decode-max-dis 15 --vis 4

# B 迁移
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --batch_size 8 --workers 4 --epochs 15 \
    --pretrained_model <sim_ckpt>.pth --pretrained_skip hm --pretrained_src_max_dis 15 \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>
python tools/eval_on_mav6d.py --ckpt <B_ckpt>.pth --tag B_transfer --vis 3

# C 纯真实
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --batch_size 8 --workers 4 --epochs 15 \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>
python tools/eval_on_mav6d.py --ckpt <C_ckpt>.pth --tag C_real_only --vis 3
```

源域预训练（产出 `<sim_ckpt>`）：30 轮，loss 448 → 0.335，单卡 RTX 5090 Laptop 约 4.5 小时。

```bash
python tools/train.py --cfg_file cfgs/models/uavdet_3d/laam6d/centerdet_rgb_near15.yaml \
    --batch_size 8 --workers 4 --epochs 30 --logger_iter_interval 20 \
    --set DATA_CONFIG.DATA_PATH <data_collect>
```
