# MonoCD (CVPR'24, 官方代码) → cmpfull 无人机数据：改动记录

源：`E:/ext_repos/monocd`（commit 1a4a6bc，原仓库未改动）完整复制到本目录后修改。所有改动处都带 `[UAV port]` 注释，`grep -rn "UAV port"` 可列全。

## 入口
- 训练：`bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [伪标签.pkl ...] --out OUT --seed 0 --epochs 19`
  （= `python train.py ...`；`--batch` 改 batch 时学习率按线性规则缩放；`--resume` 续训；`--max-iters` 冒烟用；`--no-pretrain` 随机初始化）
- 预测：`python predict.py --ckpt OUT/model_final.pth --split real_test --out pred.pkl`
  输出与 labels.pkl 同序，每帧 `dict(seq, frame, boxes9d (n×9: center, lwh, euler_xyz), scores)`，按分数取前 10。
- 自检：`python uav_tools/roundtrip_check.py`（真值编码→官方解码→boxes9d，原图重投影）；`uav_tools/sim_stats.py`（单类均值）。

## 改了什么、为什么
| 文件 | 改动 |
|---|---|
| `data/datasets/uav.py`（新） | 逐行照抄 `kitti.py::__getitem__` 的目标生成（热图、2D 框、3D 中心偏移/截断外推、edge fusion、10 关键点+可见性+关键点深度掩码、尺寸、深度、MultiBin），只换 KITTI 专有部分：读 labels.pkl；单类；框几何中心；角点/关键点用完整 R；无标注 2D 框→投影框裁到图内；截断率用投影框出图比例算；遮挡无→0；图像缩放到 960×540、**下方**补零到 960×544（pad_size=(0,0)）、K 只缩放。 |
| `model/anno_encoder.py` | 新增 `ypr_to_internal_R`（可微）、`encode_box3d_R`（官方 encode_box3d 同角点顺序，ry 矩阵换成完整 R）、`decode_euler_orientation`（对每个角调用官方 multi-bin 解码；yaw 仍做 alpha+射线角→ry）。 |
| `model/head/detector_loss.py` | 角点损失的真值/预测角点用完整 R；朝向损失 = 3 个角各一个官方 `Real_MultiBin_loss`，同权相加；`torch.arange` 设备修正（torch≥1.5 规则）。 |
| `model/head/detector_infer.py` | UAV 分支输出 `[cls, score, center, lwh, ypr, box2d]`（不做底面中心 +h/2 的 KITTI 平移）；打分仍是官方 heatmap×(1−clamp(深度不确定度,0.01,1))。 |
| `config/defaults.py` | 加 `UAV.*` 选项（不开时行为=KITTI 原样）。 |
| `runs/monocd_uav.yaml`（新） | 官方 `runs/monocd.yaml` 原样复制，改动行标 `# UAV:`。 |
| 兼容性（不改算法） | DCNv2 CUDA 扩展未编译 → `torchvision.ops.deform_conv2d`（同为调制可变形卷积，offset 交错 (dy,dx)、mask=sigmoid；官方偏移卷积零初始化，无预训练 DCN 权重，故布局无关）；`inplace_abn` 未装 → 等价的 BN+LeakyReLU(0.01)（同参数名，只是不省显存）；sklearn/skimage/pycocotools 只用于关掉的分支或可视化 → 改为可选导入；`select_topk` 中 `/` 改整除（torch 1.4 的 int/int 是整除，新版是真除，不改会把峰值坐标算错）；`collections.Iterable`→`collections.abc`。 |
| 环境 | pip 装了纯 Python 包 `yacs==0.1.8`（官方 config 依赖）。 |

### 角度（按 cmpfull/angles.py 约定）
- 监督用 `ypr=[yaw,pitch,roll]`，`R = Ry(yaw)Rx(pitch)Rz(roll)@R0`。MonoFlex 内部物体系（x=长、y=高朝下、z=宽）正好是 angles.py 的"水平机体系"，所以 `R_int = R@R0ᵀ = Ry Rx Rz`，pitch=roll=0 时与官方 `rad_to_matrix(ry)` 完全相同；尺寸内部顺序 (l,h,w)=(lwh0,lwh2,lwh1)。
- yaw = KITTI ry：官方的 alpha=ry−atan2(x,z) 编码、解码 ry=alpha+射线角，原样作用于 yaw。pitch、roll：同样 4 bin（中心 0/±90°/180°，margin 1/6）+ (sin,cos) 残差，回归绝对角。头通道 `ori_cls`/`ori_offset` 由 8/8 → 24/24。
- 翻转增广（官方 ry→π−ry）：`R_int' = M·R_int·S_z`（M 镜像相机 x，S_z 翻物体宽度轴）⇔ `(yaw,pitch,roll)→(π−yaw, −pitch, roll)`，中心 x→−x，cx'=W−1−cx（官方写法）。往返自检见下。
- euler_xyz 只在 predict 输出时用 `angles.euler_xyz_from_ypr` 换回。

### 互补深度（MonoCD 核心）怎么处理及依据
官方互补深度公式（`decode_depth_from_roof_and_bottom`）：`z = (Y·f_v + P[1,3] − v·P[2,3]) / (v − c_v)`，Y=EL−H/2 是物体中心在相机系的高度，v 是关键点给出的中心像素行。公式本身只依赖内参，**保留不动**。
问题在 Y 的来源：官方配置（`PRED_GROUND_PLANE: True`）用地平线热图拟合出地面法向，再固定相机离地高 d=1.65 m（`decode_ground_plane_from_heatmap`），把**底面中心关键点的射线与地面求交**得到 Y（`decode_y3d_from_ground_plane`）。这一步明确假设"物体底面落在地面上"，对飞行中的无人机不成立（射线会打到无人机后方的地面，或在地平线以上无交点），且我们没有地面真值来训练地平线热图。所以**只去掉地平线热图头 + 地面求交这一 Y 来源**（及其 horizon_hm_loss）。
Y 改由官方代码里自带的另一条路径给出：`y3d_offset`/`y3d_uncertainty` 回归头（`decode_y3d`，`y3d_offset_loss` 带不确定度，`PRED_Y3D` 分支，训练/推理代码都是官方现成的，只是官方 KITTI 配置没打开）。为此做的三个设定：
1. `Y3D_REG: ['linear', True]`：官方默认 `['exp', True]` 是 `mean + std·exp(o)`，永远 ≥ 均值，表示不了相机上下两侧都有的无人机；linear 也是官方支持的模式。
2. `Y_MEAN/Y_STD/Y3D_RANGE` 用 sim_train 统计（−0.133 / 0.807 / [−3.08, 2.65]）。
3. `y3d_offset_loss` 权重 1（与直接深度、原 horizon_hm_loss 同权）——这是我选的数，官方没有。
EL 定义成 center_y + h/2，使官方公式里的 EL−H/2 恰好是框中心的相机系 Y。
于是 7 路深度（直接 1 + 关键点 3 + 互补 3）的不确定度加权与官方完全相同。替代方案（若认为"Y 头"不算原方法）：把 `compensated_depth_uncertainty`、`y3d_*` 头和对应损失去掉、`CORNER_LOSS_DEPTH: soft_combine`，即退化为 MonoFlex。
已知性质（不是移植错误）：用真值 Y 和真值关键点代入，互补深度中位相对误差 0.1%，但物体中心行 v≈c_v（与相机等高）时分母趋零，最大相对误差 7×（截到 DEPTH_RANGE），由官方不确定度加权负责降权。KITTI 车辆总在相机下方，所以官方不会遇到这个问题。

### 过滤
- `FILTER_MORE_STRICTLY` 只保留截断 ≤0.5（用投影框算）；其中 KITTI 评测口径的 2D 高 ≥25 px、深度 2–65 m 关掉（`UAV.STRICT_HEIGHT_DEPTH_FILTER=False`）：无人机扁，sim_train 在 960×540 下 2D 高中位 30 px，开着会丢 45% 的框，并丢掉 <2 m 的近距离样本（真实测试 1.6–5.6 m）。`FILTER_ANNOS [0.9,20]` 保留（无影响）。
- 输入 960×544：544 = 17×32，DLA 的 32 倍下采样已对齐，无需再补。

## 轮数建议：19 轮（衰减在第 15、17 轮）
官方：KITTI train 3,712 张、batch 8、100 轮 → 46,400 次迭代，第 80/90 轮 ×0.1。sim_train 19,490 帧、batch 8 → 2,436 次/轮，19 轮 = 46,284 次 ≈ 官方迭代数；衰减点按 80%/90% 换算为 15/17（`--epochs E` 自动按 0.8E/0.9E 设）。

## 显存 / 耗时（共享 GPU 上实测，仅供估计）
- 官方 batch 8 在 GPU_FRAC=0.2（4.77 GB）下 OOM；batch 4 也 OOM；batch 2 稳态峰值 4.1 GB（已扣除 cudnn.benchmark 首几步的调优工作区）。按线性外推 batch 8 约 13–16 GB → 正式训练需 `GPU_FRAC≈0.65`（或按线性规则用 batch 2、lr 7.5e-5，但迭代数变 4 倍）。未用更高显存实测 batch 8。
- 速度：batch 2 约 0.235 s/次（期间 GPU 有他人负载）；推算 batch 8 约 0.8–0.9 s/次 → 每轮约 35 min，19 轮约 11 h。数据加载 2 worker 约 4 ms/次，不是瓶颈。预测约 0.2 s/帧（batch 1，官方推理方式）。

## 预训练权重（本机没有，未下载）
官方用 ImageNet DLA-34：`dla34-ba72cf86.pth`，URL `http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth`（约 60–65 MB，未核实）。放到本地后 `export DLA34_PTH=<路径>` 即可（`MODEL.PRETRAIN=True` 为官方默认）。不设时官方代码会尝试联网下载，train.py 会给警告。冒烟全部用 `--no-pretrain` 随机初始化。

## 冒烟结果（2026-10-29）
- 往返自检（sim_val 50 帧 108 框，真值→数据集编码→官方解码→boxes9d→原图 K 重投影）：中心误差 ≤5e-7 m，lwh ≤6e-8 m，旋转 ≤2e-5°，8 角点重投影 ≤8e-5 px；翻转样本解码后镜像回去：中心 ≤5e-7 m、旋转 ≤3e-5°。108 框全部被编码（过滤没丢框）。
- 训练 200 次（batch 2、lr 7.5e-5、随机初始化、GPU_FRAC 0.2）：总损失 54.9→22.3（前/后 20 次均值）；hm 4.50→3.41，朝向 6.64→2.10，直接深度 3.52→1.57，关键点深度 14.8→0.71，角点 8.45→0.90，加权深度相对误差 6.57→0.55；y3d 损失 0.70→0.76（200 次内还没动）。产物 `D:/cmpfull_runs/monocd_smoke_bs2/`。
- predict（50 帧 sim_val / real_test）：格式、同序、≤10 框、分数降序均正确。官方阈值 0.2 下冒烟模型 0 框（不确定度≥1 m 时官方打分直接为 0）；`--thresh 0` 时每帧 10 框，尺寸/深度数值合理。

## 不确定的保真度点
1. 互补深度的 Y 来源（见上）：官方地面求交 → 官方代码里的 y3d 回归头；Y3D_REG 改 linear；y3d 损失权重 1 是自选。
2. 关键点深度 `f·H/Δv` 假设物体高度轴竖直；无人机 pitch/roll ≤±40°，这是方法本身的近似，未改。
3. 官方打分 `1−clamp(深度不确定度,0.01,1)` 以米为单位，对 1–8 m 的无人机同样适用但可能造成大量 0 分；保持官方。
4. 官方 `decode_depth_from_keypoints_batch`/互补深度里 `calibs[idx]` 按"有目标的图"序号取标定，batch 内有空图时会取错图的 K（官方 bug）。仿真各帧 K 相同无影响；混入真实伪标签（K 不同）时可能有影响，保持官方未改。
5. torchvision 可变形卷积与 DCNv2 CUDA 数值实现不同（同一算子定义）。
6. 正式训练若显存不够只能用小 batch+线性缩 lr，会偏离官方 batch 8。
- 10-30：官方 detector_loss 里遇 NaN/Inf 进 pdb，后台会直接崩（10:39 崩过一次）→ 改为 train.py 跳过该步更新并记 warning。
