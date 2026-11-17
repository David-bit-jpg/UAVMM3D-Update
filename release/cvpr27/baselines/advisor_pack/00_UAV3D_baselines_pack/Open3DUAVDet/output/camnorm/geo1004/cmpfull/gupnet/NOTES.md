# GUPNet（ICCV 2021）→ 无人机统一数据：最小改动说明

来源：`E:/ext_repos/gupnet` 原样复制到 `code/`（删了 `__pycache__`），原仓库没改。改过的地方都标了 `[UAV]`。
KITTI 的代码（`kitti.py`、`kitti_utils.py`、`train_val.py`、`tester_helper.py`、`config.yaml`）都留着没删，方便对照。

## 入口
- 训练：`bash train.sh --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl [伪标签.pkl ...] --out <dir> --seed 444 --epochs 140 [--batch-size B]`
  （实际跑的是 `code/tools/train_uav.py`；`--labels` 可以给多个，图像根目录取记录里的 `root` 字段，没有就用 pkl 所在目录）
- 预测：`python predict.py --ckpt <dir>/checkpoints/checkpoint_final.pth --split real_test --out pred.pkl`
  输出与 `labels.pkl` 同序，每帧是 `dict(seq, frame, boxes9d (n,9) = center, lwh, euler_xyz, scores (n,))`，按分数降序最多 10 个。
- 自检：`tests/roundtrip.py`（编码→解码往返，`--flip` 检查翻转），`tests/flip_R_check.py`，`tests/check_pred.py`（检查输出格式和投影）。

## 改了什么
| 文件 | 改动 | 原因 |
|---|---|---|
| `lib/datasets/uav.py`（新，由 `kitti.py` 改写） | 读 labels.pkl；单类 drone；图缩放到 960×540 后在下方补零到 960×544，K 只缩放；`resolution` 1280×384 改成 960×544；2D 框 = 8 角点投影后裁到 960×540；3D 投影中心 = 几何中心（去掉原来的 `pos − h/2`）；depth = center_z；size 顺序 (h,w,l) = (lwh[2],lwh[1],lwh[0])；heading 目标从 1 个角扩成 3 个角 | SPEC 第 1、3、4 条 |
| 〃 过滤 | 保留 truncation ≤ 0.5（截断率 = 投影框落在有效区外的面积比例）和 heatmap 中心必须在图内；**去掉** KITTI 的 2D 高 ≥ 25 px、occlusion、pos_z < 2 m 三条 | 在 960 宽下，sim_train 里 44.6% 的框 2D 高 < 25 px，1.3% 的框 z < 2 m（真实测试是 1.6–5.6 m）；数据里没有遮挡信息 |
| `lib/datasets/uav_utils.py`（新） | `UAVCalib`（P2=[K\|0]，ry2alpha/alpha2ry/img_to_rect 的公式照抄原实现；flip 直接写成 cu' = W − cu，原来是用 8 个点重新解 P2，结果一样）；朝向编解码；翻转 | 见下面「朝向」 |
| `lib/models/gupnet.py` | heatmap 输出通道从写死的 3 改成 cls_num(=1)；heading 头从 24 维改成 72 维（3 个角 ×（12 bin + 12 残差））；uint8 掩码改成 bool；`GUPNET_NO_PRETRAIN=1` 时不加载 ImageNet 权重（只给冒烟用） | |
| `lib/backbones/dla.py` | 可以用环境变量 `GUPNET_DLA34_WEIGHTS` 指向本地的官方 `dla34-ba72cf86.pth` | 本机不联网下载 |
| `lib/losses/loss_function.py` | heading 损失 = 3 个角各算一次原来的 multi-bin（CE + 选中 bin 的残差 L1），同权相加；HTL 原来写死的 T=140 改成 max_epoch；`.cuda()` 改成跟随输入所在设备 | SPEC 第 2 条 |
| `lib/helpers/decode_helper.py` | 新增 `decode_detections_uav`：3 个角各用原来的 `get_heading_angle`；yaw = alpha2ry(alpha, 2D 中心 u)；位置 = img_to_rect(x3d, y3d, depth)（去掉原来的 `+h/2`）；输出 9D | |
| `lib/helpers/trainer_helper.py` | 去掉训练中途的 KITTI 评测（原来只写 txt，不影响训练和选模型）；结束时存 `checkpoint_final`；mask 转成 bool；`max_iters` 只给冒烟用；日志里加显存峰值和耗时 | |
| `tools/train_uav.py`、`experiments/config_uav.yaml`、`train.sh`、`predict.py` | 新入口。超参数全部是官方默认：DLA34 + DLAUp，Adam lr 1.25e-3，wd 1e-5，batch 32，余弦 warmup 5 轮，在 90/120 轮各 ×0.1，140 轮，flip 0.5，crop 0.5，scale 0.4，shift 0.1，use_3d_center | |

## 朝向（按 cmpfull/angles.py 的约定）
- 每个框用 `ypr = [yaw, pitch, roll]`，满足 `R = Ry(yaw)·Rx(pitch)·Rz(roll)·R0`；yaw 就是 KITTI 的 ry。没有用 euler_xyz 做监督。
- **yaw**：照 GUPNet 原逻辑，网络预测观察角 `alpha = ry2alpha(yaw, u)`，u 是 2D 框中心横坐标（原代码用的就是 box2d 中心），解码时 `yaw = alpha2ry(alpha, u)`。
- **pitch、roll**：直接回归绝对角，编码和原来一样（12 bin + 残差，bin 覆盖 2π）。仿真里这两个角都在 ±40° 以内，只会用到中间 3–5 个 bin。
- 原来的 alpha 换算只对 ry 有意义，所以只用在 yaw 上（SPEC 的写法）。没有用完整 R 做"观察角"的推广，比如 Ry(−θ)·R。
- **翻转增广**：原来是 `ry → π − ry`、`x → −x`。对完整 R，保持机头和朝上的水平镜像是 R' = S·R·D，换成角度就是 `(yaw, pitch, roll) → (π − yaw, −pitch, roll)`。`tests/flip_R_check.py` 用 500 个随机 R 核对，最大误差 1.5e-15；`roundtrip.py --flip` 拿翻转后解码出的角点和独立镜像的真值角点比，误差 1e-6 m。
- GUPNet 里 ry 只用来编码 heading，没有用 ry 去转角点算深度或关键点，所以没有其他地方需要换成完整 R。
- 输出时用 `angles.euler_xyz_from_ypr` 换回 euler_xyz。

## 深度几何（原逻辑保留，在这里写清楚）
GUPNet 的几何深度是 `depth_geo = h3d / h2d · f_u`，h3d 是 size_3d 的第 0 维，这里取 lwh[2]（机体竖直轴），h2d 是 2D 框高。不确定度 `σ_geo` 从 h3d 的不确定度投影过来，最终深度 = depth_geo + 一个学出来的正残差 `1/σ(·) − 1`。这些都没改。
- KITTI 的车是竖直放的，相机也是水平的，所以 2D 高 ≈ f·h/z。无人机姿态和视角都是任意的，旋转后的框投影出来的 2D 高等于 8 个角点投影的纵向范围，大致对应 `|R[1,:]|·lwh`（框在相机 y 方向的展开），不等于 lwh[2]。
- sim_train 实测：`f·lwh[2] / h2d / z`（几何深度 / 真实深度）的中位数 0.61，5%–95% 分位 0.34–0.89，所以几何深度系统性**偏小**。换成相机 y 方向展开的话，中位数 0.93（0.59–1.35）。
- 原方法的残差只能为正，正好能补"偏小"的情况，约 5% 比值 > 1 的样本补不回来。这是原方法用在任意姿态物体上的固有局限。我们没改：既没有偷偷换成相机 y 方向展开（那算增强），也没有加别的东西。

## 轮数选择
- 官方：KITTI train 3,712 张 × 140 轮，batch 32，约 16,240 次迭代、52 万个样本。
- 本实现**保留官方按轮的全部日程**（warmup 5 轮、衰减 90/120 轮、HTL 的 T 和 5 轮统计窗口、140 轮），但每轮只随机取 `epoch_size = 3712` 个样本（每轮不放回抽样，轮与轮之间重新抽）。这样迭代数和样本数都和官方一样，相当于把 sim_train 19,490 张过了约 26.7 遍。
- 不建议直接跑 140 个完整轮：样本数是官方的 5.25 倍，HTL 的 5 轮窗口和衰减点在样本意义上也会被拉长 5 倍。
- 也可以用 `--epoch-size 0 --epochs 27`（整轮，迭代数相近），但那样 HTL 的 5 轮窗口占到训练的 18%（官方是 3.6%），更不忠实。

## 耗时与显存（冒烟实测，随机初始化）
- batch 4、960×544、DataLoader 2 个 worker：显存峰值 3.34 GB，约 0.19 s/迭代（每秒约 21 张；冒烟时 GPU 基本空闲）。
- batch 8 在显存上限 0.2（4.77 GB）下 OOM。
- 外推（显存按约 0.76 GB/张线性估）：batch 32 ≈ 24 GB，单卡共享时跑不了；batch 16 ≈ 12.5 GB（要 `GPU_FRAC≈0.6`）；batch 8 ≈ 6.5 GB（`GPU_FRAC≈0.3`）。
- 建议 **batch 16、lr 6.25e-4**（按线性规则，`train_uav.py` 给了 `--batch-size` 会自动缩 lr）。轮的定义不变（3712 样本/轮），迭代数随之变成约 32,480。
- 总耗时 ≈ 52 万样本 / 每秒约 20 张 ≈ **7 小时**（只是估计，取决于 GPU 争用和读 jpg 的速度）。每轮 3712 张约 3 分钟。

## 冒烟测试结果
- 命令：`GUPNET_NO_PRETRAIN=1 GPU_FRAC=0.2 bash train.sh --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl --out smoke/run1 --epochs 10 --epoch-size 80 --batch-size 4 --max-iters 200`
  （共 200 次迭代；故意把轮设小，让 HTL 在第 7 轮把 3D 项打开，所有损失项都能走到。日志在 `smoke_run1.log`）
- 损失（每 5 个 batch 的均值，前 → 后）：seg 16.7 → 3.6，depth 7.6 → 2.7，heading（3 个角之和）10.0 → 5.2，offset3d 0.52 → 0.33，size3d 0.74 → 0.55；size2d 在 8–18 之间波动（随机初始化，2D 尺寸以 1/4 分辨率的像素计）。
- HTL 权重：前 6 轮 3D 项权重为 0；第 7 轮开始非零，这和官方行为一致。但小轮（80 样本/轮）下统计噪声很大，第 9 轮 depth 权重被算成 2.2e12（见下面「保真度问题」第 3 条）。
- 编码→解码往返（把训练目标当成完美输出，走同一个解码函数）：sim_train 300 帧 617 个框，加上 sim_val 50 帧，角点误差最大 1.2e-6 m、投影 1e-4 px，旋转误差 1e-6°；翻转版本角点误差 1.1e-6 m。没有框被过滤掉。
- predict 自检：sim_val 50 帧输出与 labels.pkl 同序，key 是 seq/frame/boxes9d/scores，boxes9d 形状 (10,9)，分数降序。冒烟模型（随机初始化、200 次迭代）预测还不准，这是预期的。

## 保真度上不确定的地方
1. **ImageNet 预训练**：官方会联网下载 `http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth`（DLA-34 ImageNet，约 63 MB）。本机没有这个文件（@@ROOT@@/torch_home 只有 resnet18/34），没有下载。正式训练前需要拿到这个文件，然后 `export GUPNET_DLA34_WEIGHTS=<路径>`；不设的话官方代码会自己去下载（该站可能已失效）。另一个来源是 timm 的 `dla34.in1k`，那是同一份权重转换过来的，但键名需要映射，没做。**不要用随机初始化做正式对比**，那会削弱原方法。
2. 去掉了 KITTI 的 25 px、z < 2 m 两条过滤（理由见上表）。保留它们会丢掉 45% 的样本。
3. **HTL 不稳定**是原代码自带的：`init_diff` 来自最初那个窗口；如果某项损失前几轮几乎不降或在涨，权重的指数会变成负数，权重就会爆掉。官方 3712 样本/轮时统计平滑得多，但正式训练时请看日志里的 `Weights:` 行。代码没改。
4. 几何深度用 lwh[2] 当 h3d（上面写了），对任意姿态有系统偏差，残差只能为正。
5. 2D 框用的是 8 角点投影的外接框，不是物体的真实轮廓。KITTI 的 2D 框是人工标的，更紧。这个差别会让 h2d 偏大，也是几何深度偏小的原因之一。
6. 翻转时 cu' = W − u，沿用官方 `calib.flip` 的约定（严格的像素中心约定应该是 W − 1 − u），差半到一个像素，影响可以忽略。
7. pitch、roll 用 2π 范围的 12 个 bin（同 bin 数），所以绝大部分 bin 用不上，分辨率等于 30°/bin 加残差。这是"同样编码"要求的直接结果。
8. predict 默认不卡分数阈值（官方 tester 用 0.2），只取前 10 个，交给评测。需要的话加 `--thresh 0.2`。
9. 原 Trainer 用 `DataParallel`；单卡下没影响，保留。
10. 没装任何新包。
