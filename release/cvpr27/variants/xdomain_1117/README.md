# 跨域补充实验（11-17，用户批准，补充材料）

两组实验：
- **实验 A：换目标域、不再适配**（只推理）。把已经适配到一个真实目标域的检测器，直接拿去另一个真实目标域的测试集上测。
- **实验 B：换源域的完整 PST 链**。室外仿真 SimDrone3D 作源域、室内真实 MAV6D 作目标域（代号 out2mav）。它和已有的 srcx/in2antifs（室内仿真 → 室外真实 RealDrone3D，测试 UDS 15.82）正好是一对。

数据集名：MAV6D = 室内真实（代码目标 `mav`）；RealDrone3D = 室外真实（代码目标 `antifs`）；SimDrone3D 室内 / 室外 = 代码源域 `in` / `out`。

真值使用：两组实验都只由现成的打分脚本 `srcx/lib/score_x.py` 读测试真值，每个格子只打一次。没有任何真值参与挑选。

## 实验 A：换目标域、不再适配

### 模型（全部现成，来自 `review1113/ourdet_baseproto/`，没有新训练）

| 格子 | 检测器 | 训练时的源域（解码统计量来源） | 测试集 |
|---|---|---|---|
| A1 | MAV6D 适配的 +PST 行（in2mav_pst，第 5 轮存档） | 室内仿真 | RealDrone3D 测试集（2,624 帧） |
| A2 | RealDrone3D 适配的 +PST 行（out2antifs_pst，第 5 轮存档） | 室外仿真 | MAV6D 测试集 |
| A3（参照） | 室内仿真纯仿真检测器，种子 0（release_20260920/ramp_s0.pth） | 室内仿真 | RealDrone3D 测试集 |
| A4（参照） | 室外仿真纯仿真检测器，种子 0（sim_out_r34_geo_ramp_out/DET_s0 第 5 轮） | 室外仿真 | MAV6D 测试集 |

存档目录：`D:/S2MUAV3D/ourdet_baseproto_models/models/uavdet_3d/camnorm/ourdet_baseproto/<行>/DET_s0/ckpt/`。

### 解码与打分

- 推理脚本原样用 `review1113/ourdet_baseproto/lib/ourdet_infer.py`：单模型、不集成、不翻转、不用 s̄，预测尺寸解码（`SIZE_SOURCE pred` + `SIZE2D_MODE width`，深度由预测的 2D 宽和模型自己预测的 3D 尺寸解出），每帧取分数最高的 1 个框。与 ourdet_baseproto 完全相同。
- 唯一的改动是环境变量 `SX_SRC / SX_TGT`：解码配置 = `srcx/real_<训练源域>2<测试目标域>.yaml`。它和 ourdet_infer.py 为对应组合建配置的方式一样：
  - 测试目标域自己的真实数据配置：缓存、内参、输入归一化。RealDrone3D 用 `outreal_fs_r34_geo_full_tr.yaml`，归一化均值 [0.4968, 0.5059, 0.5084]、方差 [0.1854, 0.1816, 0.1970]。MAV6D 用 `mav6d_r34_geo_full_tr.yaml`，归一化均值 [0.4360, 0.4418, 0.4374]、方差 [0.2239, 0.2244, 0.2200]。这两组数都由各自无标签训练图算出，与本目标域原来的测试一致。
  - **来自训练源域的常数**：解码器的深度 / 尺寸标准化统计量（DEPTH_MEAN/STD、SIZE_MEAN/STD），因为网络就是按它们训练的：
    - 室内源域：深度 5.9527 ± 3.7845 m，尺寸均值 [0.6609, 0.6330, 0.2555]，方差 [0.5517, 0.5008, 0.1951]（A1、A3，配置 `srcx/real_in2antifs.yaml`，早已存在）。
    - 室外源域：深度 28.3561 ± 20.9749 m，尺寸均值 [0.8036, 0.7784, 0.3668]，方差 [0.4473, 0.4304, 0.2376]（A2、A4，配置 `srcx/real_out2mav.yaml`，由 `srcx/lib/mkcfg.py` 用 SX_SRC=out SX_TGT=mav 新写出）。
  - 没有从测试真值重新算任何东西。也没有用 s̄。
- 打分：`srcx/lib/score_x.py`。MAV6D 用 real_test 规则（绝对阈值 0.1/0.2/0.4/0.8 m），RealDrone3D 用 antifs_test 规则 + 封存真值。与 ourdet_baseproto 相同。
- 脚本：`run_A.sh`。预测与打分 json：`preds/<格子>_test.pkl(.score_<split>.json)`；日志 `logs/`。

### 结果（测试集，各打分一次）

| 测试集 | 检测器（适配到哪个目标域） | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|---|
| RealDrone3D | A1 MAV6D 适配的 +PST（室内源域） | 0.00 | 0.638 | 11.604 | 16.66 | 6.59 | 9.69 |
| RealDrone3D | A3 室内仿真纯仿真（种子 0） | 0.00 | 0.640 | 11.391 | 25.98 | 2.10 | 7.10 |
| RealDrone3D | 参照：室外仿真纯仿真（种子 0，ourdet_baseproto 行 1） | 2.93 | 0.470 | 36.975 | 26.31 | 71.95 | 17.22 |
| RealDrone3D | 参照：RealDrone3D 适配的 +PST（ourdet_baseproto） | 14.77 | 0.128 | 15.279 | 23.21 | 72.83 | 30.01 |
| MAV6D | A2 RealDrone3D 适配的 +PST（室外源域） | 0.00 | 3.288 | 20.970 | 26.44 | 7.25 | 7.64 |
| MAV6D | A4 室外仿真纯仿真（种子 0） | 0.00 | 2.214 | 16.452 | 23.91 | 14.10 | 9.05 |
| MAV6D | 参照：室内仿真纯仿真（种子 0，ourdet_baseproto 行 1） | 13.09 | 0.666 | 5.920 | 20.42 | 69.79 | 32.59 |
| MAV6D | 参照：MAV6D 适配的 +PST（ourdet_baseproto） | 23.84 | 0.485 | 2.943 | 20.89 | 95.88 | 47.10 |

帧数：RealDrone3D 测试 2,624 帧、MAV6D 测试 4,800 帧，四个格子每帧都有 1 个框（无空帧）。置信度中位数 A1 0.283 / A3 0.279 / A2 0.310 / A4 0.346。
参照行是 ourdet_baseproto 里同一协议的现成分数（README 结果表），本次没有重算。

读法：适配到一个真实目标域的 +PST 检测器换到另一个目标域后基本失效（AP 都是 0，UDS 7.6–9.7）。
它比同源的纯仿真检测器只在朝向上好一些（A1 对 A3：AOE 26.0→16.7°，mDR 2.1→6.6，UDS 7.1→9.7）。
在 MAV6D 上反而比纯仿真更差（A2 对 A4：UDS 9.05→7.64，ATE 2.21→3.29 m）。
在 MAV6D 上的主要问题是深度：室外训练的网络（解码深度均值 28 m）放到 1.6–5.6 m 的室内场景，ATE 有 2–3 m。在 RealDrone3D 上则是检出率：mDR 只有 2–7%。
所以 PST 的增益来自对目标域本身的适配，不能不经再适配就搬到另一个目标域。

## 实验 B：室外仿真 → MAV6D 的完整 PST 链（out2mav）

- 用干净链 `srcx/run_x.sh out mav` 原样运行，没有改 srcx/ 下任何代码。`sx.py` 本来就同时有 `SOURCES['out']` 和 `TARGETS['mav']`，库里没有按源域 / 目标域分支的代码，所以不需要复制库。
- 输出：`srcx/out2mav/`（新目录）、大文件 `D:/S2MUAV3D/srcx_work/out2mav/`、自训练检测器 `output/models/uavdet_3d/camnorm/srcx/st_out2mav/`、伪标签索引 `E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_out2mav.pkl`（新文件名）。配置 `tools/cfgs/.../srcx/{real,st}_out2mav.yaml`（新文件）。
- 包装脚本 `run_B.sh`：run_x.sh 每一步若输出已存在就跳过，所以被停掉后重跑即续跑；最多 6 次，间隔 5 分钟。GPU 步骤都经过 `gproto/gpu_slot.sh`（链内自带）。
- 沿用的跨源域规则（STATUS.md 偏差 1–4、6、6b，全部不变）：
  - 精炼器第 1 段按本组合重训，背景库来自 MAV6D 无标签训练图。
  - 精炼器仿真样本与 w_sim 用室外源域纯仿真检测器种子 0。
  - 米制阈值按距离比例：仿真匹配取距离的 25%，伪标签一致取距离的 14.3%（室内 3.5 m 处 0.5 m）。
  - 朝向裁窗放大上限：真实 MAV6D 帧用 4.0（目标域相机 / 距离范围），仿真验证裁窗用源域自己的 12.5。
  - 宽度门限按输入像素判断（mav 系数 3.75）。
  - 真实检测器配置 = MAV6D 数据 + 室外源域解码统计量（见上）。
  - s̄ 回退：第 1 段检测器集成若在无标签验证图上没有分数 ≥ 0.7 的框，取最有把握的前 10%；自训练检测器集成若没有，就沿用第 1 段的 s̄。
- 链内的验证集打分只是自检，不参与任何选择（与其它 srcx 组合相同）。测试集打分在链尾只做一次。
- 运行记录：`srcx/out2mav/logs/run_x.log`，包装记录 `logs/run.log`。

## 汇总表

| 实验 | 格子 | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|---|
| A | MAV6D 适配的 +PST → RealDrone3D 测试 | 0.00 | 0.638 | 11.604 | 16.66 | 6.59 | 9.69 |
| A | 室内仿真纯仿真（种子 0）→ RealDrone3D 测试 | 0.00 | 0.640 | 11.391 | 25.98 | 2.10 | 7.10 |
| A | RealDrone3D 适配的 +PST → MAV6D 测试 | 0.00 | 3.288 | 20.970 | 26.44 | 7.25 | 7.64 |
| A | 室外仿真纯仿真（种子 0）→ MAV6D 测试 | 0.00 | 2.214 | 16.452 | 23.91 | 14.10 | 9.05 |
| B | 室外仿真 → MAV6D，只用仿真（3 种子集成 + 翻转 + s̄） | 运行中 | | | | | |
| B | 室外仿真 → MAV6D，完整 PST 链 | 运行中 | | | | | |
| 对照 | 室内仿真 → RealDrone3D，完整 PST 链（srcx/in2antifs，已有） | 0.00 | 0.596 | 6.764 | 13.40 | 17.57 | 15.82 |
| 对照 | 室内仿真 → RealDrone3D，只用仿真（srcx/in2antifs，已有） | 0.00 | 0.546 | 7.723 | 25.19 | 4.88 | 10.46 |
| 对照 | 室内仿真 → MAV6D，只用仿真（srcx/in2mav，已有） | 26.87 | 0.283 | 1.737 | 20.32 | 69.77 | 49.51 |
| 对照 | 室内仿真 → MAV6D，完整 PST 链（srcx/in2mav，已有） | 40.93 | 0.306 | 1.754 | 14.48 | 96.17 | 60.81 |
