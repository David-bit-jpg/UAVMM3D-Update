# MonoCoP（CVPR 2026）→ cmpfull 无人机数据 移植说明

来源：官方仓库 `E:/ext_repos/monocop` 原样复制（原仓库没动）。所有改动行都标了 `[UAV port]`。
规格：`../SPEC.md`。同门参照：`../monodgp/`（MonoCoP 的数据集代码 `kitti_dataset.py`、`decode_helper.py`、`save_helper.py`、
`ms_deform_attn_func.py`、`kitti_utils.py`、`pd.py` 与 MonoDGP 官方仓库**逐字相同**（diff 去掉行尾 CR 后只差 MonoDGP 文件末尾的 `__main__` 调试段），
所以 UAV 数据集类、解码、存档、可变形注意力纯 PyTorch 回退直接照搬 MonoDGP 移植版，两种方法的数据口径完全一致）。

## 入口
- 训练：`bash train.sh --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl [伪标签.pkl[@图像根目录] ...] --out <dir> --seed 444 --epochs 48 --batch-size 2 [--resume] [--max-iters N]`
  （= `python tools/train_uav.py ...`；`GPU_FRAC` 限显存（默认 0.2，见下文需要 0.3）；`TORCH_HOME` 默认 `@@ROOT@@/torch_home`；`--workers` 默认 2；
  `--keep-every 4` 每 4 轮另存只含模型的 `model_epoch_N.pth`，**最后一轮一定写 `model_epoch_<E>.pth`**（队列判断训完用）。）
- 续训：每轮结束写 `<out>/checkpoint.pth`（模型 + 优化器 + 学习率调度器状态 + 轮数），`--resume` 从它接着训。
- 预测：`python predict.py --ckpt <dir>/checkpoint.pth（或 model_epoch_N.pth） --split real_test|sim_val|real_train --out pred.pkl`
  → 与该 split 的 labels.pkl 同序的 list，每帧 `dict(seq, frame, boxes9d (n×9: center, lwh, euler_xyz，我们的定义), scores (n,))`，n ≤ 10。
- 自检：`python predict.py --roundtrip --split sim_val --limit 50`（真值 → 训练编码器 → 解码器 → boxes9d，不用模型）；
  `python tools/selfcheck_proj.py --pred <pred_sim_val.pkl>`（格式 + 投影自检）。
- 原 KITTI 启动脚本改名为 `train_kitti.sh`；KITTI/nuScenes/Waymo 数据集代码没动（只是改成用到时才 import）。

## 改了哪些文件、为什么
| 文件 | 改动 |
|---|---|
| `lib/datasets/uav/uav_dataset.py`（新，照搬 MonoDGP 移植版，只改了一行注释） | 代替 `kitti_dataset.py`。`__getitem__` 逐步保留：光度扰动、翻转 0.5、裁剪/缩放 0.05/平移 0.05 仿射（概率 0.5）、mixup3d（MonoCoP 官方 0.3，读 config，同 K + 同尺寸才混）、投影 3D 中心在图内过滤、l/r/t/b ≥ 0 过滤（clip_2d False）、深度目标 = z·crop_scale、同样的 target 字典。KITTI 语义替换见下节。 |
| `lib/models/monocop/monocop.py` | `num_classes` 1（config）。角度头 24 → 72（ypr 三个角 × 官方 12 bin 分类 + 12 残差）；`loss_angles` 对每个角用官方的同一损失（bin 交叉熵 + 真值 bin 残差 L1），三个角等权相加（`angle_loss_coef`=1 不变）。深度图损失里写死的 `[80, 24, 80, 24]`（1280×384 的 stride-16 特征图大小）→ 实际特征图大小。`.cuda()` → `device=`。**CoP 链（size→angle→depth 的 AttributeNet 串联）、训练时按组随机走链/并行（`groupwise_random_route`）、推理时 UGS（链路深度的 log 方差 < τ=5 用链，否则用并行头）、所有损失/匹配/权重原样。** 角度头变宽后仍放在链的中间（angle_feat 继续喂深度），链结构不变。 |
| `lib/helpers/decode_helper.py`（照搬） | heading 的 `gather` 用实际宽度 72。 |
| `lib/models/monocop/ops/functions/ms_deform_attn_func.py`（照搬） | CUDA 算子没编译 → 回退到仓库自带的 `ms_deform_attn_core_pytorch`（同一算子：双线性 grid_sample、零填充、align_corners=False、加权和）。只影响速度/显存，不影响结果。若以后编好 `MultiScaleDeformableAttention` 会自动用。（MonoCoP 的 `ms_deform_attn.py` 已修好 torch 2.x 版本判断的优先级 bug，不用改。） |
| `lib/models/monocop/region_seg_head.py` | `sklearn` 改成用到时才 import（只有 `cluster_regions` 用，官方 forward 从不调用；本环境没装 sklearn）。 |
| `lib/models/monocop/backbone.py` | **官方骨干不是 torchvision ResNet-50**，而是 RT-DETR 的 ResNet-50-vd（Paddle 转换权重 `ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`，构造时 `torch.hub.load_state_dict_from_url` 自动下载）。改成：只从 `$TORCH_HOME/hub/checkpoints/<同名文件>` 读，**缺失就直接报错，绝不自动下载**；`MONOCOP_NO_PRETRAIN=1` 才随机初始化（仅冒烟用）。见「待办 1」。 |
| `lib/helpers/dataloader_helper.py` | 加 `type: UAV` 分支（无 test loader，选模型由 predict.py 在 sim_val 上做）；KITTI/NUSC/WAYMO 数据集改为懒加载。 |
| `lib/helpers/trainer_helper.py`（MonoCoP 官方版与 MonoDGP 官方版只差引号，所以以 MonoDGP 移植版为底） | 绝对路径 `--out`、损失 jsonl 日志、`max_iters`（冒烟用）、每 4 轮 + 最后一轮另存 `model_epoch_N.pth`。**新增（MonoDGP 移植版没有）：** ① 存档里加 `lr_scheduler_state`，续训时恢复；② 损失非有限（NaN/Inf）时跳过这一步（不 backward、不 step，`zero_grad`），打 WARNING 日志并计数；网络输出 NaN 导致匈牙利匹配（scipy）抛 ValueError 时同样跳过这一批。优化本身没改（无梯度裁剪，同官方）。 |
| `lib/helpers/save_helper.py`（照搬） | `torch.load(..., weights_only=False)`。 |
| `config/monocop_uav.yaml`（新） | `config/monocop.yaml`（三类版本）的拷贝；只改数据集 type/labels、`num_classes: 1`、`writelist: [drone]`、过滤开关、类均值尺寸（不用）。模型/损失/匹配/优化器/增广全部官方值（含 `uncertainty_tau: 5`、`group_num: 11`、`random_mixup3d: 0.3`、`depth_max 60`、LID 80 bin）。 |
| `tools/train_uav.py`、`train.sh`、`predict.py`、`tools/selfcheck_proj.py`（新） | 入口（照搬 MonoDGP 版，只改官方 batch = 16、config 路径；predict 里设 `MONOCOP_NO_PRETRAIN=1`，因为骨干初值会被存档覆盖，预测不需要 ImageNet 文件）。 |

代码里会被走到的 `pdb/ipdb.set_trace()`：MonoCoP 里全部本来就是注释掉的（`waymo_eval.py` 只 `import pdb`，不会走到）；没有交互式输入。

### 标签语义（KITTI → 我们，与 MonoDGP 移植版完全相同）
- **中心**：直接用框几何中心（KITTI 代码本身也先把底面中心换成几何中心再投影）；解码去掉 `+h/2` 底面中心偏移。深度目标 = 几何中心的 z。
- **2D 框**：KITTI 用标注 2D 框（`bbox2d_type: anno`），我们没有 → 用 8 个投影角点的紧框，裁到 960×540 图内。
- **过滤**：保留 truncation ≤ 0.5（= 1 − 裁剪后/未裁剪投影框面积；遮挡无 = 0）。KITTI 的 25 px 框高和 z ≥ 2 m 关掉（`min_box_height: 0`、`min_depth: 0`）：那是 KITTI 评测协议 / 路面场景阈值，开着会丢掉 45% 的 sim_train 无人机和真实测试里的近距离目标（1.6–5.6 m）。65 m 上限保留（无影响）。
- **尺寸**：KITTI 顺序 `[h, w, l]` = `(lwh[2], lwh[1], lwh[0])`。官方 `meanshape: False` ⇒ 回归绝对尺寸；sim_train 单类均值 `[0.251, 0.620, 0.646]` 只在打开 meanshape 时用。
- **角度**：KITTI 式 `ypr`（`../angles.py`，R = Ry(yaw)·Rx(pitch)·Rz(roll)·R0）。**yaw 就是 KITTI 的 ry**，完整走官方处理：`alpha = ry − atan2(u2d − cu, fu)`，12 bin + 残差，解码 `alpha2ry`。pitch、roll 用同样的 12 bin + 残差直接回归绝对角。三个角损失等权相加。predict 用 `angles.euler_xyz_from_ypr` 换回 euler_xyz。
- **翻转增广**：(yaw, pitch, roll) → (π − yaw, −pitch, roll)，即 R' = S·R·D（S = diag(−1,1,1)，D = diag(1,−1,1)），MonoDGP 移植时已在 500 个随机 R 上验证到 1e-15。
- **几何深度（MonoCoP 也有）**：`depth = h3d · fx / h2d + Δ`，h3d = 预测尺寸[0]（无人机高 `lwh[2]`）、h2d = 预测 2D 框高。无人机倾斜时 2D 框高里含有顶/底面和 l、w 的投影，h3d·f/h2d 会系统性偏小，由学到的 Δ（MonoCoP 里由 CoP 链或并行头给出、UGS 选择）吸收。保持方法原样，没换成考虑旋转的投影高度。
- **匹配**：匈牙利代价不变（类别 2、3D 中心 10、2D l/r/t/b 5、GIoU 2），训练 11 组 × 50 query，测试 1 组。

### 输入分辨率
SPEC 要 960×540 → 下方补零到 960×544。**MonoCoP 的 RegionSegHead 与 MonoDGP 完全相同**（stride-64 层 ×2 上采样后加到 stride-32 层），要求输入高是 64 的倍数（544/32 = 17 为奇数会崩）。所以同 MonoDGP 补到 **960×576**（同样的 960×540 内容、同样的缩放 K，多 32 行零；官方 KITTI 输入 1280×384 也是 64 的倍数）。模型代码没为此改动。

## 解码 / 输出
完全照 `decode_helper.py`：sigmoid 类别概率取 top-50，score = prob · exp(−log σ)（深度不确定度），中心 = 预测投影 3D 中心按预测深度反投影，yaw 用 2D 框中心 x 做 `alpha2ry`。推理时深度偏移由 UGS 在 CoP 链和并行头之间逐 query 选（官方 eval forward，τ = 5）。DETR 无 NMS；保留最终分数最高的 10 个，**不设分数阈值**（官方 tester 写 KITTI txt 前过滤 prob < 0.2，留给评测器）。

## 轮数与学习率（推荐 48 轮，batch 2，lr 2.5e-5）
- 官方 KITTI：250 轮 × 3,712 张，**batch 16**（`config/monocop.yaml` 与 `assets/train.log`：Batch Size 16、lr 2e-4）→ 232 it/轮，**5.8 万次迭代，看过 92.8 万个样本**；AdamW lr 2e-4、wd 1e-4，LambdaLR 在 [85,125,165,205] 处 ×0.5，无 warmup；官方两份日志的最好验证轮是 246 / 229（每轮在 KITTI val 上挑）。
- sim_train 19,490 帧：48 轮 × 19,490 = 93.6 万样本 ≈ 官方看过的样本数（与 MonoDGP 同样按样本数对齐，MonoDGP 也是 48 轮）。衰减轮按比例缩放 → **[16, 24, 32, 39]**（`train_uav.py` 自动算）。
- batch 2：按线性规则 lr = 2e-4 × 2/16 = **2.5e-5**（MonoDGP 官方 batch 8，所以它是 5e-5；两者都是同一条线性规则）。轮数仍 48（样本数不变，迭代数 8 倍 = 46.8 万次）。
- 选模型：官方每轮在 val 上挑最好；我们写 `checkpoint.pth`（最新）+ 每 4 轮和最后一轮 `model_epoch_N.pth`，由统一评测器在 **sim_val** 上挑（不用真实标签）。

## 续训说明
官方续训只设 `lr_scheduler.last_epoch = epoch − 1`，而存档前调度器已经 step 过 `epoch` 次，所以官方续训后每个衰减点会**晚一轮**（一个 off-by-one）。我们在存档里多存调度器状态并在续训时恢复（last_epoch = epoch），续训后的学习率曲线与不中断时完全一致。不中断训练时与官方没有任何差别。（MonoDGP 移植版保留了官方写法，存在这个晚一轮的问题，供协调者参考。）

## 代价（RTX 24 GB，纯 PyTorch 可变形注意力）
- batch 2：峰值 `max_memory_allocated` **4.85 GB**、reserved **5.12 GB** → `GPU_FRAC=0.25`（≈ 6.1 GB）够用（冒烟就是 0.25 跑的）；正式训练建议 **0.3** 留余量（mixup/目标数波动、workers 2）。
- 速度：冒烟实测 **1.1 s/it**（batch 2、workers 0，当时显卡被 MonoDGP/MonoCD 等占到 15–20 GB、利用率 70–95%，**严重争用，不代表独占速度**）。
  按这个速度 9,745 it/轮 ≈ 3 h/轮，48 轮 ≈ 6 天；独占或轻度共享时预计与 MonoDGP（同骨架量级，MonoCoP 只多三个小 MLP 链 + 一个并行深度头，3.4 it/s）接近，
  约 **3 it/s ≈ 55 min/轮、48 轮 ≈ 45 h**（估计，未在空闲卡上实测）。
- 预测：50 张 14–16 s（含模型构建，GPU 争用中）。
- 存档 `checkpoint.pth`（含优化器）502 MB；只含模型的快照约 170 MB。

## 冒烟测试（2026-10-31 00:45，GPU_FRAC 0.25，batch 2，workers 0，200 次迭代，`MONOCOP_NO_PRETRAIN=1` 随机初始化骨干，输出 @@ROOT@@/runs/baselines/monocop_smoke）
- 先查 nvidia-smi（≤ 16 GB 才跑）、用 `SMOKE_LOCK` 锁、无 PAUSE；跑完已释放锁。
- 损失（前 20 次均值 → 后 20 次均值）：loss_detr 53.5 → 38.2，ce 0.98 → 0.73，angle 7.47 → 4.39，dim 1.34 → 0.85，depth_map 1.60 → 0.85，bbox 0.93 → 0.48，depth 1.97 → 1.80，center 0.79 → 0.75；region dice ≈ 平（3.84 → 3.83）、giou 1.78 → 2.14（随机初始化骨干早期正常波动）。非有限损失跳过次数 0。
- 续训（CPU 干跑）：`--resume` 正确读回轮数/优化器/调度器状态并接着训。
- `predict.py` sim_val 50 / real_test 50：与 labels.pkl 同序（错位 0），boxes9d (10, 9) float32，scores (10,)，全部有限。
- **50 张 sim_val 投影自检**（`tools/selfcheck_proj.py`）：top-1 预测中心投影 50/50 落在图内，冒烟模型 top-1 与真值的平均角点投影距离 214 px、中心距 1.9 m（模型没训好，数值无意义；约定一致性看下面的往返误差）。

## 编码 → 解码往返误差（CPU，不用模型）
- sim_val 前 50 帧（50 个框，0 个被过滤）：最大角点误差 5.2e-7 m，最大角点重投影误差 4.5e-5 px，最大旋转误差 6.0e-6°。
- sim_train 前 1000 帧（1000 个框）：8.5e-7 m / 7.0e-5 px / 1.0e-5°。
（与 MonoDGP 移植版一致：同一编码/解码代码。）

## 待办 / 与官方的差异
1. **官方骨干预训练权重不在本机**：`ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`（ResNet-50-vd，RT-DETR 作者用 Paddle 转换，大小未核实），
   来源 `https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`，放到 `@@ROOT@@/torch_home/hub/checkpoints/`。
   本机只有 torchvision 的 `resnet50-0676ba61.pth`，它是普通 ResNet-50（7×7 stem、无 avg-pool 下采样捷径），结构不同、**装不进** vd 骨干。
   按指示没有自行下载。**正式训练前需要用户下载该文件**（或协调者决定改用 torchvision ResNet-50 —— 那是偏离官方的改动，不推荐）。
   缺文件时 `train.sh` 会直接报错退出，不会悄悄随机初始化。冒烟测试用 `MONOCOP_NO_PRETRAIN=1` 随机初始化。
2. 补零到 576 而不是 544（结构约束，内容相同），同 MonoDGP。
3. KITTI 25 px / 2 m 训练过滤关掉，同 MonoDGP。
4. batch 2 + 线性 lr，而非官方 batch 16（显存）。Group-DETR 匹配、11 组 query、按组随机走链都是逐图的，batch 只影响优化噪声。
   注意 `groupwise_random_route` 每张图每组独立抽签，与 batch 无关。
5. 深度 LID 80 bin、0–60 m 保持官方；无人机 ≤ 8 m 只用到前约 29 个 bin（官方设定，不改）。
6. 几何深度用无人机高度对倾斜放大的 2D 框高（方法原样，由 Δ 吸收）。

## 给队列的正式训练命令（需先放好待办 1 的骨干权重）
```
GPU_FRAC=0.3 TORCH_HOME=@@ROOT@@/torch_home bash @@ROOT@@/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocop/train.sh --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl --out @@ROOT@@/runs/baselines/monocop_s0 --seed 444 --epochs 48 --batch-size 2 --workers 2
```
（中断后同一命令加 `--resume`；训完标志 `@@ROOT@@/runs/baselines/monocop_s0/model_epoch_48.pth`。）
