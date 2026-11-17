# MonoDETR (ICCV 2023) → cmpfull 无人机数据 移植说明

来源：官方仓库 `E:/ext_repos/monodetr` 整体复制过来再改（原仓库未动）。规格见 `../SPEC.md`。
数据加载、角度 ypr、尺寸、过滤规则、predict 输出格式、存档/续训 **直接照搬已验证的 MonoDGP 移植**（`../monodgp/`），
保证几个方法口径一致；MonoDETR 自己的模型、损失、匹配器、超参数保持官方原样。所有改动行都标了 `[UAV port]`。

## 入口
- 训练：`bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [pseudo.pkl[@img_root] ...] --out <dir> --seed 444 --epochs 37 --batch-size 2 [--resume] [--max-iters N]`
  （= `python tools/train_uav.py ...`；`GPU_FRAC` 限显存，默认 0.2；`TORCH_HOME` 默认 `E:/torch_home`，ImageNet ResNet-50 权重从那里读，不下载；`--workers` 默认 2；`--keep-every` 默认 4。）
- 预测：`python predict.py --ckpt <dir>/checkpoint.pth|model_epoch_N.pth --split real_test|sim_val|real_train --out pred.pkl [--limit 50]`
  → 与 labels.pkl 同序的 list，每帧 `dict(seq, frame, boxes9d (n×9: center, lwh, euler_xyz，我们的定义), scores (n,))`，n ≤ 10。
- 自检：`python predict.py --roundtrip --split sim_val --limit 50`（真值 → 训练编码 → 解码 → boxes9d）。
- 官方 KITTI 启动脚本保留为 `train_kitti.sh`；KITTI 数据集代码未动（只是不再提前 import）。

## 改了哪些文件、为什么
| 文件 | 改动 |
|---|---|
| `lib/datasets/uav/uav_dataset.py`（新，抄自 MonoDGP 移植） | 代替 `kitti_dataset.py`。`__getitem__` 逐步保留官方流程：光度扰动、0.5 水平翻转、0.5 概率 crop/scale 0.05/shift 0.05 仿射、3D 中心在图内过滤、l/r/t/b ≥ 0 过滤（clip_2d False）、深度目标 = z·crop_scale、同样的 target 字典。与 MonoDGP 版的区别只有三点，都是因为 MonoDETR 官方 loader 本来就不同：**不做 mixup3d**（MonoDETR 官方没有；配置里显式 0）、**不产生 obj_region**（MonoDGP 专有的区域分割头）、**画布 960×544 严格按 SPEC**（MonoDGP 因 RegionSegHead 要 576；MonoDETR 没这个限制，544 时各层 68/34/17/9 行，深度预测器按 stride-16 尺寸显式插值）。 |
| `lib/models/monodetr/monodetr.py` | `num_classes` 1（配置）。角度头 24 → 72（yaw/pitch/roll 各用官方 12 bin 分类 + 12 残差）；`loss_angles` 对每个角原样算官方损失（bin 交叉熵 + 真值 bin 残差 L1），三项等权相加（`angle_loss_coef`=1 不变）。深度图损失里写死的 `[80, 24, 80, 24]`（1280×384 的 stride-16 图尺寸）→ 实际图尺寸（960×544 下 60×34）。`.cuda()` → `device=`。 |
| `lib/models/monodetr/backbone.py` | `pretrained=True` → `weights='IMAGENET1K_V1'`（就是同一个 `resnet50-0676ba61.pth`；torchvision 0.26 没有 `pretrained=`）。权重从 `TORCH_HOME=E:/torch_home` 读。`MONODETR_NO_PRETRAIN=1` 仅供离线测试，正式训练不要设。 |
| `lib/models/monodetr/ops/functions/ms_deform_attn_func.py` | CUDA 算子没编译 → 回退到仓库自带的纯 PyTorch 版 `ms_deform_attn_core_pytorch`（同一个算子：双线性 grid_sample、零填充、align_corners=False、加权求和）。只影响速度/显存，不影响结果。若以后编译了 `MultiScaleDeformableAttention` 会自动用。（文件与 MonoDGP 移植版相同。） |
| `lib/models/monodetr/ops/modules/ms_deform_attn.py` | 官方 torch 版本判断在 torch 2.x 上解析错（运算符优先级）→ 改 try/except import。（同 MonoDGP。） |
| `lib/helpers/dataloader_helper.py` | `type: UAV` 分支；不建测试 loader。（同 MonoDGP。） |
| `lib/helpers/decode_helper.py` | 角度 `gather` 用实际宽度 72。（同 MonoDGP。） |
| `lib/helpers/save_helper.py` | `torch.load(..., weights_only=False)`；存档额外存 `lr_scheduler_state`，`load_checkpoint` 可恢复调度器；先写 `.tmp` 再 `os.replace`，中途被杀不会留下坏存档。 |
| `lib/helpers/trainer_helper.py` | 绝对路径 `--out`；损失 jsonl 日志；`max_iters` 冒烟停止；**每轮结束写 `checkpoint.pth`（模型+优化器+调度器+轮数）**；**每 4 轮写 `model_epoch_N.pth`，最后一轮无论能否被 4 整除都写 `model_epoch_<E>.pth`**；**续训**：从 `<out>/checkpoint.pth` 恢复模型/优化器/调度器/轮数。**非有限损失保护**：损失非有限 → 不反传、不更新，写日志 `[UAV port] non-finite loss ... step skipped`；反传后梯度总范数非有限（`clip_grad_norm_(max_norm=inf)`，只用来检查，不缩放梯度）→ 同样跳过并记日志。官方没有梯度裁剪，这里也没有。`prepare_targets` 保持 MonoDETR 原样（没有 MonoDGP 的 depth_map/obj_region 键）。 |
| `configs/monodetr_uav.yaml`（新） | `monodetr.yaml` 的副本，只改数据集类型/标签、`num_classes: 1`、`writelist: [drone]`、`random_mixup3d: 0`、过滤开关、类均值尺寸（不用）。模型/损失/匹配器/优化器/增广全部是官方值。 |
| `tools/train_uav.py`、`train.sh`、`predict.py`（新，抄自 MonoDGP） | 入口。只改了官方 epoch/batch 常数（195/16）、默认配置名、默认 TORCH_HOME。 |

### 交互式调试
全仓库搜过 `pdb` / `ipdb` / `set_trace` / `input(`：官方 MonoDETR 只有被注释掉的 `#ipdb.set_trace()`，没有会被走到的断点；未加任何交互代码。

### 续训的一个官方小 bug（已修）
官方 resume 设 `lr_scheduler.last_epoch = epoch - 1`，而存档是在 `scheduler.step()` 之后写的，所以续训后每个衰减点会晚一轮。这里先恢复调度器状态，并设 `last_epoch = epoch`（与不中断训练完全一致）。测试：20 帧小集训 2 轮（衰减 [1,2]）→ `--resume --epochs 3`，续训后 lr = 2.5e-7 = 2.5e-5·0.1²，正确；`model_epoch_2.pth`、`model_epoch_3.pth` 都写出。

### 标签语义（KITTI → 我们，与 MonoDGP 移植完全相同）
- **中心**：KITTI 的 pos 是底面中心，官方代码自己换成几何中心（`pos + [0,-h/2,0]`）做 3D 中心投影；我们直接用几何中心。解码时去掉底面中心的 `+h/2`。深度目标 = 几何中心的 z。
- **2D 框**：没有标注 2D 框 → 用 8 个投影角点的外接框，裁到 960×540 图内。
- **过滤**：KITTI 丢 `UnKnown`（框高 < 25 px 或截断 > 0.5 或遮挡 > 2）和 z < 2 m。这里保留截断 ≤ 0.5（截断 = 1 − 裁后/裁前投影框面积；无遮挡信息 = 0）。**25 px 和 2 m 关掉**（`min_box_height: 0`、`min_depth: 0`）：这是 KITTI 评测协议/道路场景阈值，开着会丢掉 45% 的 sim_train 无人机（960 宽下框高 < 25 px）和近处的无人机（真实测试 1.6–5.6 m）。65 m 上限保留（无影响）。
- **尺寸**：官方 `[h, w, l]` 顺序，h = `lwh[2]`，w = `lwh[1]`，l = `lwh[0]`。官方 `meanshape: False` ⇒ 直接回归绝对尺寸（MonoDETR 的 3D 尺寸由解码器 `dim_embed` 逐层回归）；sim_train 单类均值 `[0.251, 0.620, 0.646]` 填了但不用。
- **角度**：KITTI 式 `ypr = [yaw, pitch, roll]`（`../angles.py`，R = Ry(yaw)·Rx(pitch)·Rz(roll)·R0）。**yaw 就是 KITTI 的 ry**，完整走官方流程：`alpha = ry − atan2(u2d − cu, fu)`，12 bin + 残差，解码用 `alpha2ry`。pitch、roll（仿真 ±30° 内）用同样的 12 bin + 残差直接回归绝对角。三项损失等权相加。predict 用 `angles.euler_xyz_from_ypr` 换回 euler_xyz。
- **翻转增广**：KITTI 的 `ry → π − ry`；保持机头/朝上的镜像 R' = S·R·D 在本约定下正好是 **(yaw, pitch, roll) → (π − yaw, −pitch, roll)**（MonoDGP 移植里已在 500 个随机 R 上验证到 1e-15，管线里翻转样本解码误差 9e-6°；本数据集代码与之逐行相同）。
- **深度**：MonoDETR 官方的深度 = (回归深度 + 几何深度 + 深度图在 3D 中心处的采样) / 3，代码不动。其中几何深度 `h3d · fx / h2d` 用预测的物体高度（我们的 `lwh[2]`）和预测 2D 框高度；无人机倾斜时 2D 框高度还包含顶/底面和 l、w 的投影，几何深度会系统性偏小，靠另外两项和不确定度加权去吸收——这是方法本身的设计，没改（同 MonoDGP 的处理）。
- **匹配**：匈牙利代价不变（类别 2、3D 中心 10、2D l/r/t/b 5、GIoU 2），训练 11 组 × 50 query，测试 1 组。

## 解码 / 输出
完全按 `decode_helper.py`：sigmoid 类别概率取 top-50，得分 = prob · exp(−log σ)（深度不确定度），中心 = 预测的 3D 中心投影 + 预测深度反投影，yaw 用 `alpha2ry`（2D 框中心 x）。DETR 没有 NMS；保留最终得分最高的 10 个，**不设得分阈值**（官方 tester 写 KITTI txt 前过滤 prob < 0.2，留给统一评测器）。

## 轮数与学习率（推荐 37 轮，batch 2，lr 2.5e-5）
- 官方 `configs/monodetr.yaml`：195 轮 × 3,712 张 = **723,840 个样本**，batch 16 → 232 it/轮，共 45,240 次迭代；AdamW lr 2e-4，wd 1e-4，step 衰减 ×0.1 于 [125, 165]。
- sim_train 19,490 帧：723,840 / 19,490 = 37.1 → **`--epochs 37`**（721,130 个样本，差 0.4%）。衰减轮按比例缩放：125·37/195 = 23.7 → **24**，165·37/195 = 31.3 → **31**（`train_uav.py` 自动算）。
- 显存放不下官方 batch 16 → **batch 2，学习率按线性规则缩放 2e-4 × 2/16 = 2.5e-5**，轮数不变（看过的样本数相同，迭代次数是官方的 8 倍：9,745 it/轮 × 37 = 360,565 次）。
- 模型选择：官方每轮在 KITTI val 上挑最好；这里每 4 轮 + 最后一轮存 `model_epoch_N.pth`，由统一评测器在 **sim_val** 上挑（不用真实标签）。

## 冒烟测试（2026-10-31，GPU_FRAC 0.25，batch 2，workers 0，200 次迭代，ImageNet 预训练骨干）
当时显卡：另外两个正式训练在跑，已用 15.3 GB、利用率 98%。
- 损失（前 20 次 → 后 20 次的均值，`loss_detr` 为含 2 个辅助层的总和）：loss_detr 48.5 → 34.7，ce 1.03 → 0.74，bbox 1.13 → 0.57，angle 7.27 → 4.49，center 0.84 → 0.70，depth_map 1.48 → 1.02，dim 1.18 → 1.05，depth 1.69 → 1.68（平），giou 1.79 → 1.90（平）。没有非有限损失/梯度。
- 显存：`max_memory_allocated` 4.16 GB，`max_reserved` 4.74 GB → **`GPU_FRAC=0.25`（≈6.1 GB）够用**；0.2（4.9 GB）太紧，不推荐。
- 速度：1.2 it/s（workers 0，数据在主进程里加载；单样本加载+增广 0.134 s，即每次迭代约 0.27 s 是数据）。扣掉数据后 GPU 部分约 0.55 s/it（在 98% 被占的卡上）。正式训练 workers 2 时预计 ~1.8 it/s → 9,745 it/轮 ≈ **90 min/轮，37 轮 ≈ 55 h**（当前两个训练同时在跑的情况下；卡空一些会快，MonoDGP 在较空时是 3.4 it/s）。
- 存档：`checkpoint.pth`（含优化器/调度器）448 MB；模型快照约 150 MB。
- predict：sim_val 50 张、real_test 50 张，顺序/seq/frame 与 labels.pkl 一致，boxes9d (10, 9) float32，scores (10,)。
- 续训：见上（20 帧小集 2 轮 → 续到 3 轮，lr 与轮数正确）。

## 编码 → 解码往返误差（无模型，真值喂训练编码器再走 predict 的解码）
- sim_val 50 帧 50 个框：角点最大误差 5.2e-7 m，角点重投影最大误差 4.5e-5 px，旋转最大误差 6.0e-6°（0 个被过滤）。
- sim_train 1000 帧 1000 个框：8.5e-7 m / 7.0e-5 px / 1.0e-5°。

## sim_val 50 张投影自检
- 用 `predict.corners`（boxes9d 的 euler_xyz）算的真值角点与 labels.pkl 定义（`(PROTO*lwh) @ R.T + center`）的投影：最大差 0 px（50 个框）。
- 冒烟模型 top-1 预测的 3D 中心投影 50/50 落在图内；叠图 `smoke/sim_val50_proj.jpg`（绿 = 真值，红 = 冒烟模型 top-1；冒烟模型没训练，框本身不准，只看约定）。

## 正式训练命令（给队列，一行）
```
GPU_FRAC=0.25 bash E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodetr/train.sh --labels D:/cmpfull/sim_train/labels.pkl --out D:/cmpfull_runs/monodetr_s0 --seed 444 --epochs 37 --batch-size 2
```
中断后同一命令加 `--resume` 接着训；`model_epoch_37.pth` 出现即训完。

## 偏离官方之处（汇总）
1. batch 16 → 2，lr 线性缩到 2.5e-5（显存）；轮数 195 → 37、衰减 [125,165] → [24,31]（按看过的样本数对齐）。
2. KITTI 25 px / 2 m 训练过滤关掉（见上）；2D 框用投影外接框（没有标注框）。
3. 角度头 1 个角 → 3 个角（ypr），损失三项等权；翻转用 (π−yaw, −pitch, roll)。
4. 纯 PyTorch 可变形注意力（数值同一算子，只是慢）。
5. 续训修正了官方 last_epoch 差一轮的问题；加了非有限损失/梯度跳过保护（官方没有；正常训练不触发时与官方完全一致）。
