# Open 3D UAV Detection Codebase (2D UAV, 3D UAV detection, 6D UAV pose estimation)

6D pose estimation require object shape and class prior, 3D UAV detection does not require such prior.

## Detection Framework




## Datasets

UG2 challenge 

https://drive.google.com/drive/folders/1wk-c5xVX6701WNI_In1ba3_D4LSjRYv5

MAV6D

## Model Zoo and Benchmark

## 2D

## Getting Started
### Dependency
Our released implementation is tested on.


### Prepare dataset

MAV6D dataset
```
data
├── MAV6D
│   ├── phantom4
│   │   │── JPEGImages
│   │   │   │──01(scene)
│   │   │   │   │──0101 (seq)
│   │   │   │   │   │──1.jpg
│   │   │   │   │   │──...
│   │   │   │──02
│   │   │   │──...
│   │   │── labels
│   │   │── split
│   │   │   │── train.txt
│   │   │   │── val.txt
│   │   │   │── test.txt
├── uavdet3d
├── tools
```



### Setup

```
cd Open3DUAVDet
python setup.py develop
```

### Training

### Testing

## License

This code is released under the [Apache 2.0 license](LICENSE).

## High-level API



## Citation

```
@inproceedings{VirConv,
    title={Virtual Sparse Convolution for Multimodal 3D Object Detection},
    author={Wu, Hai and Wen,Chenglu and Shi, Shaoshuai and Wang, Cheng},
    booktitle={CVPR},
    year={2023}
}
```





---

# MAV6D 迁移学习：环境、修复与运行说明

本节记录把这套代码在新机器上跑通、并让 **MAV6D（西湖大学室内 MAV 6-DoF 位姿数据集）**
可用于迁移学习所做的全部改动。

## 1. 本次修复的问题

代码原样是跑不通 MAV6D 的，以下都是实际触发过的阻断性问题：

| # | 位置 | 问题 | 处理 |
|---|------|------|------|
| 1 | `uavdet3d/datasets/dataset.py` | `DatasetTemplate` 只为 `LAA3D_Det_Dataset` / `LAAM6D_Det_Dataset` 创建 `data_pre_processor`，两个 MAV6D dataset 在 `__getitem__` 里却会调用它 → `AttributeError` | 补上 MAV6D 分支，走 `DataPreProcessor`（其 `object_encoder` 用 `'xyz'` 欧拉角，与 `mav6d_*_dataset.py` 的 `as_euler('xyz')` 一致） |
| 2 | `uavdet3d/utils/object_encoder.py` | `center_point_encoder` 用 `gt_box9d_with_cls[:, 0, :]` / `obj[-1, 0]` 按三维取下标，而 gt 实际是 `(N, 10)` 二维 → `IndexError: too many indices` | 新增 `object_encoder_mav6d.py`，注册为 `center_point_encoder_mav6d` / `center_point_decoder_mav6d`，**不改动原有条目行为** |
| 3 | 同上 | 配套的 `backProject_with_opencv_to_world` 里带 CARLA↔OpenCV 坐标轴置换。MAV6D 的 `read_truth_Rt` 返回的已经是标准 OpenCV 相机系，再置换一次就错了 —— 原 encoder/decoder 对 MAV6D **并不互逆** | 新实现是严格互逆的一对（往返实测位置误差 0.0000 m） |
| 4 | 同上 | MAV6D 镜头畸变不小（k1 = −0.23），原实现在投影与反投影两侧都把畸变当 0 | 新实现支持 `use_distortion`，默认开启；关掉即退化为纯针孔，便于与源域行为对齐做消融 |
| 5 | `tools/cfgs/dataset_configs/uavdet_3d/mav6d.yaml` | 缺 `MAX_SIZE`，而 `convert_box9d_to_centermap` 会做 `dim / MAX_SIZE` → `AttributeError` | 补 `MAX_SIZE: 1`（目标本体约 0.24 m） |
| 6 | `uavdet3d/datasets/mav6d/eval.py` | `val_rotation_enler` 写死 `from_euler('zyx')`，但整条 MAV6D 链路用的是 `'xyz'`。**按错误顺序解释欧拉角后算出的角度差不等于真实角度差 —— 这个指标本身是错的** | 改为 `'xyz'`，并把 `euler_seq`、`fold_180` 提成参数 |
| 7 | 同上 | `all_error[all_error>90] = 180 - all_error` 把大于 90° 的误差折叠，等价于假设目标 180° 对称，会让指标显著偏好 | 保留默认行为但改成 `fold_180` 开关，可与不折叠的数一起看 |
| 8 | `uavdet3d/datasets/mav6d/mav6d_det_dataset.py` | `evaluation()` 里用 gt 长度的掩码去索引预测框（`pre_box[name_mask]`），两者行数一般不等 → `IndexError` | 改为按预测框最后一列的类别索引筛选 |
| 9 | `tools/eval_utils/eval_utils.py` | `eval_one_epoch` 末尾没有 `return`（返回 `None`），`repeat_eval_ckpt` 随后调用 `.items()` → 训练结束时必崩 | 补 `return {}`，并在 `tools/test.py` 加空值保护 |
| 10 | `tools/train_utils/optimization/fastai_optim.py` | `from collections import Iterable` 在 Python ≥3.10 已失效 | 改为 `collections.abc` |

## 2. 环境

已在以下环境验证通过：

- Windows 11 + conda env `city`
- Python 3.10 / PyTorch 2.11.0+cu128 / torchvision 0.26 / OpenCV 4.11 / numpy 1.26
- GPU: RTX 5090 Laptop

```bash
pip install easydict tensorboardX transforms3d
pip install -e . --no-deps
```

`--no-deps` 是必要的：`setup.py` 的 `install_requires` 里有 `torch`，不加会让 pip
重装 torch、破坏已有的 CUDA 版本。

运行时需要把仓库根目录放进 `PYTHONPATH`（脚本里是 `from tools.train_utils...` 的绝对导入）：

```bash
export PYTHONPATH=/path/to/repo    # Windows: $env:PYTHONPATH="C:\path\to\repo"
```

## 3. MAV6D 数据准备

数据来源：<https://github.com/WestlakeAerialRobotics/MAV6D>（论文
*Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for Micro Aerial Vehicles*）。
下载得到 RGB 图 + 标签 + mask，**不含 split**，需要自己生成。

标签每行 16 个数：

```
ts_cam  t_x t_y t_z  r_x r_y r_z r_w    ts_uav  t_x t_y t_z  r_x r_y r_z r_w
```

`read_truth_Rt` 只取后 8 个（MAV 在 VICON 系下的位姿），相机位姿用的是一个
**硬编码的 camera→VICON 外参**。

目录整理成：

```
<MAV6D_ROOT>/
├── phantom4/
│   ├── JPEGImages/<scene>/<seq>/*.jpg
│   ├── labels/<scene>/<seq>/*.txt
│   └── split/{train,test}.txt        # 由下面的脚本生成
└── mavic2/
    └── ...
```

> 官方标签目录可能叫 `label`（单数），而代码里写死的是 `labels`；图像目录必须是
> `JPEGImages`，且扩展名必须是 `.jpg`（dataset 里用 `replace('jpg','txt')` 推标签名）。
> `mav6d_prepare.py check` 会把这些都报出来。

### 体检 / 切分 / 可视化

```bash
# 1) 体检：目录结构、帧数、距离分布、GT 投影是否落在画面内、相机是否真的固定
python tools/mav6d_prepare.py check --root <MAV6D_ROOT>

# 2) 按序列生成 split（不是按帧！相邻帧几乎一样，按帧切会让指标虚高）
python tools/mav6d_prepare.py split --root <MAV6D_ROOT> --train-ratio 0.8

# 3) 把 GT 3D 框投影回图上肉眼确认
python tools/mav6d_prepare.py vis --root <MAV6D_ROOT> --num 8 --out ./mav6d_vis
```

`check` 会重点验两件"会静默出错"的事：

- **GT 投影落在画面内的比例**。偏低说明内参/外参和这批数据对不上。
- **标签里逐帧记录的相机位姿是否恒定**。`read_truth_Rt` 用的是写死的外参，
  只有相机全程不动才成立；若 `check` 报出相机位姿有变化，就必须改成逐帧
  使用标签里的相机位姿，否则整段序列会系统性偏移。

另外注意 `util.py` 里可视化时用过 `offset=[0.2, 0.2, 0]`，暗示 VICON 原点与
几何中心存在偏移，而训练路径并未施加该偏移 —— 用 `vis` 确认框是否贴合。

## 4. 运行

### 源域（UAV-MM3D / data_collect）

```bash
python tools/train.py --cfg_file cfgs/models/uavdet_3d/laam6d/centerdet.yaml \
    --batch_size 2 --workers 4
```

数据路径在 `tools/cfgs/dataset_configs/uavdet_3d/laam6d.yaml` 的 `DATA_PATH`。

### MAV6D — 关键点路径（数据集原生方法，PnP 解位姿）

```bash
python tools/train.py \
    --cfg_file cfgs/models/pose_estimation_6dof/mav6d_phantom4/key_point9.yaml \
    --batch_size 4 --workers 4 \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>/phantom4
```

### MAV6D — 中心点路径（与源域共享检测头，做迁移用这条）

```bash
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --batch_size 4 --workers 4 \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>
```

### 迁移：载入源域权重

```bash
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --pretrained_model <源域 ckpt>.pth ...
```

`load_params_from_file` 用的是 `strict=False`，形状对不上的层会自动跳过。

## 5. 迁移学习需要注意的四件事

1. **模态塌缩。** 源域是 RGB+IR+DVS+LiDAR+radar 五模态、backbone 为
   `ResNet8xAttention`（`NUM_MODALITIES: 5`）；MAV6D 只有单目 RGB、backbone 是
   `ResNet8x`。两者第一层输入通道与 `NUM_FILTERS` 都不同，**直接迁几乎没有权重能对上**。
   要做迁移，应先在源域上训一个 RGB-only、backbone 与 MAV6D 一致的模型。
2. **尺度断层。** `center_dis` 头回归米制距离，源域 `MAX_DIS: 150`，MAV6D 是 `8`，
   差一个数量级。这个头建议重新初始化，或改用归一化/对数距离。
3. **旋转约定。** 源域链路是 `'zyx'`，MAV6D 是 `'xyz'`。`rot` 头是 6 通道
   =（cos, sin）× 3 个角，源域第一对是绕 z、MAV6D 第一对是绕 x，
   **直接迁 `rot` 头等于把旋转轴接错**。
4. **内参差异。** MAV6D 是 1920×1080、fx≈1979 的长焦，源域是 1280×720、fx=480。
   模型从图像回归米制深度，焦距变化直接改变"表观大小↔距离"的关系。

---

# 跨域对齐：模态 / 内外参 / 坐标系

迁移到 MAV6D 之前必须先确认三件事对齐了。`tools/align_audit.py` 会用**真实样本和真实模型**把它们全部量化出来：

```bash
python tools/align_audit.py --dst-data <MAV6D_ROOT>        # 完整审计
python tools/align_audit.py --no-samples                   # 只看配置和权重（快）
python tools/align_audit.py --ckpt <源域ckpt>.pth           # 实测能载入多少
```

## 1. 模态对齐

源域原本是 5 模态（rgb + ir + dvs + LiDAR 世界坐标图 + Radar 速度图，`K=5`，骨干
`ResNet8xAttention`），MAV6D 只有单目 RGB（`K=1`，骨干 `ResNet8x`）。两者第一层
通道与每层宽度都不同，**直接迁几乎没有权重能对上**。

为此 `LAAM6D_Det_Dataset` 的模态改成可配置（`MODALITIES`，默认仍是全部 5 个，
行为不变），并新增对齐后的源域配置
`cfgs/models/uavdet_3d/laam6d/centerdet_rgb_resnet8x.yaml`：

| | 原 5 模态配置 | RGB-only 对齐配置 | MAV6D |
|---|---|---|---|
| image 张量 | `(5,3,560,960)` | `(1,3,560,960)` | `(1,3,256,512)` |
| 骨干 | ResNet8xAttention `[32,64,128,128]` | ResNet8x `[64,128,256,512]` | ResNet8x `[64,128,256,512]` |
| 检测头 | CenterHeadLaam6d（吃 4D） | CenterHead（吃 5D） | CenterHead |

> `MODALITIES` 只决定往 image 张量里堆哪几个模态；rgb/ir/dvs 的三模态配准
> （`register_images_by_center`，其输出的 aligned RGB 内参会影响下游几何）照常执行，
> 保证 RGB 在两种配置下的处理完全一致。

**实测权重可迁移性**：

| 源域 ckpt | 能载入目标模型的比例 |
|---|---|
| 原 5 模态 `centerdet.yaml` | **41/178 张量，2.97M，15.3%**（135 个 key 目标模型里根本没有）|
| RGB-only `centerdet_rgb_resnet8x.yaml` | **177/178 张量，19.43M，100.0%**（只差 `hm` 层 7 类 vs 2 类）|

## 2. 内外参对齐

| | 源域 LAAM6D | MAV6D |
|---|---|---|
| 原始分辨率 | 1280×720 | 1920×1080 |
| 训练分辨率 / stride | 960×560 / 8 → 热图 120×70 | 512×256 / 8 → 热图 64×32 |
| 实测内参 | fx=480.0 fy=502.8（已按 IM_RESIZE 缩放过）| fx=1979.4 fy=1979.1（原始分辨率）|
| 折回原分辨率 | fx=640.0 fy=646.4 | fx=1979.4 |
| FOV | 90.0° × 58.2° | 51.7° × 30.5° |
| 畸变 | 全零 | k1=−0.23（不可忽略）|
| 外参 | 非单位阵（世界↔相机）| 单位阵 |
| MAX_DIS / MAX_SIZE | 150 / 4 | 8 / 1 |

**两套内参缩放约定不同，都各自自洽，但绝不能混用**：

- 源域：编码器拿到的是**已缩放**的内参，投影后只乘 `1/stride`
- MAV6D：编码器拿到的是**原始**内参，投影后乘 `new/raw/stride`

目标表观大小（网络就是靠这个回归距离的）：

| 域 | 目标 | 距离 | 原图像素 | 热图像素 |
|---|---|---|---|---|
| 源域 | 1.30 m | 10 / 50 / 150 m | 83.2 / 16.6 / 5.5 px | 7.80 / 1.56 / 0.52 px |
| MAV6D | 0.24 m | 2 / 5 / 8 m | 237.5 / 95.0 / 59.4 px | 7.92 / 3.17 / 1.98 px |

深度归一化差 **18.8 倍**（`center_dis` 学的是 `Z / MAX_DIS`）：同一个归一化输出，
源域代表 150 m 量级、MAV6D 代表 8 m 量级。

## 3. 坐标系对齐

| | 源域 LAAM6D | MAV6D |
|---|---|---|
| GT 存储 | **世界系**（`convert_box_opencv_to_world`）| **相机系**（`read_truth_Rt` 已用 camera→VICON 外参转好）|
| 送进编码器前 | `pre_processor_laam6d` 用 `inv(extrinsic)` 转回相机系 | 不转换 |
| 检测头学到的 | 相机系深度 `Z / MAX_DIS` | 相机系深度 `Z / MAX_DIS` |
| 解码器输出 | 再转回**世界系** | 留在**相机系** |

**结论：两边检测头学的是同一种东西**（相机系深度 + 图像平面中心），世界/相机的
差异只发生在编码前和解码后。所以 head 权重可迁移，不必为坐标系改网络，
但**解码器必须各用各的**：

- 源域 → `center_point_decoder_laam6d`
- MAV6D → `center_point_decoder_mav6d`
- `object_encoder.py` 里同名的那个带 CARLA 轴置换，对 MAV6D 与其编码器并不互逆

### 欧拉角

`rot` 头是 6 通道 `[cos a1, sin a1, cos a2, sin a2, cos a3, sin a3]`：

- 源域 `zyx`：a1=绕 z，a2=绕 y，a3=绕 x
- MAV6D `xyz`：a1=绕 x，a2=绕 y，a3=绕 z

**通道 0/1 与 4/5 的物理含义在两边是对调的**，直接迁 `rot` 头等于把第一和第三个
旋转轴接反。把同一组角按两种顺序解释，实测旋转差异 **中位 134°、90 分位 175°** ——
这也是修 `eval.py` 之前角度指标的误差量级，不是小偏差。

## 4. 迁移操作建议

```bash
# 第一步：用对齐后的配置训源域，产出可迁移的 ckpt
python tools/train.py --cfg_file cfgs/models/uavdet_3d/laam6d/centerdet_rgb_resnet8x.yaml

# 第二步：迁到 MAV6D
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --pretrained_model output/models/uavdet_3d/laam6d/centerdet_rgb_resnet8x/default/ckpt/checkpoint_epoch_30.pth \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>
```

逐层建议：

| 层 | 建议 | 原因 |
|---|---|---|
| backbone | **直接迁**，主要收益来源 | 对齐后 100% 可载入 |
| `hm` | 必然重初始化 | 类别数 7 vs 2 |
| `rot` | 建议重初始化 | 通道语义对调 |
| `center_dis` | 建议重初始化，或前几轮只训这层 | MAX_DIS 差 18.8 倍 |
| `dim` | 迁不迁都行 | MAV6D 目标尺寸恒定，该头没什么信息量 |

> `load_params_from_file` 已改为**先按 名字+形状 过滤再载入**。
> `strict=False` 只放过多余/缺失的 key，**形状不符依然会抛 `RuntimeError`** ——
> 跨域必然有几层形状不同，不过滤的话 `--pretrained_model` 直接崩。
> 现在它会打印实际载入了多少、跳过了哪些、哪些保持随机初始化。

---

# 近距离源域子集（与 MAV6D 深度尺度对齐）

## 动机

上一节量化出的最大域差是**深度尺度差 18.8 倍**：源域是室外几十米（中位 42 m，
`MAX_DIS=150`），MAV6D 是室内 ≤8 m，而 `center_dis` 头学的正是 `Z / MAX_DIS`。

仿真数据里本来就有近距离的帧，把它们单独抽出来，就得到一个**深度尺度与 MAV6D
同量级、但仍带完整 6-DoF 标注**的源域——迁移时不必再跨这个数量级。

## 全库距离分布（实测 1493 条序列 / 394,209 帧 / 131 万目标样本）

单目标欧氏距离：min 0.20 m，中位 41.89 m，max 205.08 m
每帧目标数：1 个 55,305 帧；2 个 99,572；3 个 73,652；…；7 个 14,321

| 阈值 | any（至少一个在范围内） | all（全部在范围内）|
|---|---|---|
| ≤5 m | 9,949 (2.52%) | 514 (0.13%) |
| ≤8 m | 19,440 (4.93%) | 1,137 (0.29%) |
| ≤15 m | 56,782 (14.40%) | 7,685 (1.95%) |
| ≤20 m | 96,758 (24.54%) | 20,529 (5.21%) |
| ≤30 m | 181,812 (46.12%) | 68,037 (17.26%) |

**用 `all` 而不是 `any`**：`any` 只要求有一个目标近，画面里仍会留着几十米外的
目标，深度分布拉不回来。

## 两个坑

**1. 欧氏距离 ≠ 相机系深度。** 每个序列的 `distance_info.txt` 已存好逐帧距离
（读它比逐帧读 `boxes_rgb/*.pkl` 快几百倍：19 个/秒 → 全量要 5 个多小时），
但它给的是**欧氏距离**，而 `center_dis` 学的是**相机系 Z**。实测存在欧氏距离
只有几米、但 **Z 为负（目标在相机后方）** 的帧 —— 编码器遇到 `Z<=0` 直接跳过，
留着就是一张空监督图。

所以工具用**两阶段**：欧氏距离粗筛（放宽 1.5 倍）→ 读 pkl 按真实 Z 精筛。
实测 8 m 档粗筛 3,413 帧，精筛后只剩 1,188 帧（34.8%）。

**2. 帧名必须按数值排序。** 帧名是秒数时间戳（`9.7944.png` / `10.0612.png`），
`set_split` 原本用 `sorted()` 字典序，会把 `10.x` 排到 `9.x` 前面 —— 实测约
**8% 的序列时间顺序错乱**。而 `include_CARLA_data` 是用 `frame_list[i + lidar_offset]`
取 LiDAR/Radar 帧的，顺序一错多模态就对不上时间。已改为数值排序。

## 生成子集

```bash
# 扫描 + 看分布（结果缓存，后续 build 秒出）
python tools/build_near_subset.py scan --root E:/data_collect

# 生成（默认 <=8m、mode=all、metric=z、按序列切 train/test）
python tools/build_near_subset.py build --root E:/data_collect \
    --max-dist 8 --mode all --out-dir cfgs/subsets/near8 --verify 300
```

| 子集 | 阈值 | 序列 | 帧数（train/test）| 实测 Z |
|---|---|---|---|---|
| `near8` | Z ≤ 8 m | 69 | 1,188（985 / 203）| min 0.55，中位 4.89，max 7.99 |
| `near15` | Z ≤ 15 m | 310 | 11,134（8,773 / 2,361）| ≤15 |

`near8` 与 MAV6D 尺度最贴但只有 1,188 帧；`near15` 帧数实用得多，深度仍比原始
源域近一个数量级。**切分一律按序列**，避免相邻帧泄漏。

## 数据集侧的接法

配置项 `FRAME_SUBSET: {'train': ..., 'test': ...}`，指向生成的帧列表。

> 实现上**只在发射样本时按帧过滤，不动每条序列的完整有序帧列表**。
> 因为 `include_CARLA_data` 靠 `frame_list[i + lidar_offset]` 取 LiDAR/Radar 帧，
> 直接把帧列表抽稀会让偏移量指到错误的帧上。

## 训练与迁移

```bash
# 1) 训近距离对齐源域
python tools/train.py --cfg_file cfgs/models/uavdet_3d/laam6d/centerdet_rgb_near8.yaml
#    帧多一些的版本：centerdet_rgb_near15.yaml

# 2) 迁到 MAV6D
python tools/train.py --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --pretrained_model output/models/uavdet_3d/laam6d/centerdet_rgb_near8/default/ckpt/checkpoint_epoch_60.pth \
    --set DATA_CONFIG.DATA_PATH <MAV6D_ROOT>
```

这两个配置与 `mav6d/centerdet.yaml` 在**模态、骨干、检测头、深度尺度**四个维度
全部对齐，`center_dis` 头因此也变得可迁（不再像原配置那样必须重初始化）。
唯一仍不可迁的是 `hm` 层（源域 7 类 vs MAV6D 2 类）。

> 放宽阈值时记得把配置里的 `MAX_DIS` 改成同一个数，否则 `center_dis` 的
> 归一化尺度又和子集对不上了。

---

# 在服务器上跑（给协作者的最短路径）

配置文件里的 `DATA_PATH` 是开发机上的 Windows 路径，**服务器上一律用 `--set` 覆盖**，
不用改文件。

## 1. 环境

在以下组合验证通过：Python 3.10 / PyTorch 2.11.0+cu128 / torchvision 0.26 /
OpenCV 4.11 / numpy 1.26。

```bash
pip install easydict tensorboardX transforms3d
pip install -e . --no-deps          # --no-deps 必须加
export PYTHONPATH=$(pwd)            # 脚本里是 from tools.train_utils... 的绝对导入
cd tools                            # 所有命令都在 tools/ 下跑，配置里的相对路径以此为基准
```

> `--no-deps` 不能省：`setup.py` 的 `install_requires` 里有 `torch`，不加会让 pip
> 重装 torch、破坏服务器上已配好的 CUDA 版本。

## 2. 源域预训练（近距离 + 单目 RGB，与 MAV6D 对齐）

```bash
python train.py \
    --cfg_file cfgs/models/uavdet_3d/laam6d/centerdet_rgb_near15.yaml \
    --batch_size 8 --workers 4 --epochs 30 \
    --logger_iter_interval 20 \
    --set DATA_CONFIG.DATA_PATH /你的路径/data_collect
```

帧子集 `cfgs/subsets/near15/near_{train,test}.txt` 已随仓库提供，**不需要重新扫描**，
只要 `data_collect` 是同一份即可。想自己重新生成：

```bash
python build_near_subset.py build --root /你的路径/data_collect \
    --max-dist 15 --mode all --out-dir cfgs/subsets/near15
```

单卡参考性能（RTX 5090 Laptop，640×384，batch 8，workers 4）：
**0.89 it/s，1097 iter/轮 ≈ 20 分钟/轮**，显存 14 GB。

## 3. 迁移到 MAV6D

```bash
python train.py \
    --cfg_file cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
    --batch_size 8 --workers 4 \
    --pretrained_model <上一步的 ckpt>.pth \
    --pretrained_skip hm rot \
    --pretrained_src_max_dis 15 \
    --set DATA_CONFIG.DATA_PATH /你的路径/MAV6D
```

- `--pretrained_skip hm rot`：`hm` 类别数不同（源域 7、MAV6D 2）；`rot` 通道语义
  在两个欧拉约定下含义不同
- `--pretrained_src_max_dis 15`：把 `center_dis` 输出层按 15/8 缩放，米制深度预测
  原样保持（输出层是线性的，精确换算）

MAV6D 需要先整理成本仓库的布局，见上文「MAV6D 数据准备」一节的
`mav6d_prepare.py organize / check / split`。

## 4. 跑得动多大

- 显存：640×384 + batch 8 约 14 GB。80G 卡可以直接把 batch 开到 32 以上，
  相应地把 `OPTIMIZATION.LR` 按比例调大
- 分辨率：`IM_RESIZE` 在两个 near 配置里都是 `[640, 384]`。改大能提精度但要注意
  显存（960×560 + batch 8 就到 23.5 GB 了），且会拉开与 MAV6D 的表观尺度差
- 多卡：`tools/dist_train.sh` 在仓库里，但本轮没有验证过

## 5. 两个已知的坑

- **`--use_amp` 是空开关**：`train.py` 接受这个参数，但 `train_utils.py` 里没有
  `autocast` / `GradScaler`，实际不生效。别指望它省显存
- **进度只在日志里**：训练循环用 tqdm 写 stderr，重定向到文件会被 Python 缓冲住。
  已额外加了每 `--logger_iter_interval` 步往 logger 写一行
  （`iter / loss / lr / it·s⁻¹ / 本轮剩余分钟`），看进度请 tail 输出目录下的
  `train_*.log`，不要看 stdout

---

# 实验结果

Sim → Real 三臂对比的完整结果、图表和复现命令见 **[docs/results/](docs/results/)**。

一句话结论：**仿真零样本完全不可用**（位置误差中位 7.9 m、角度 120°），
**而在 MAV6D 训练数据充足（1.5 万帧）的条件下，仿真预训练没有带来增益**
（迁移 0.084 m vs 从零训 0.075 m）。96.9% 的权重能成功迁入，说明五个维度的对齐是有效的，
只是收益被充足的目标域数据掩盖了 —— 下一步应在低数据量（1%/5%/10%/25%）下重做对比。

| | A 纯模拟 | B 迁移 | C 纯真实 |
|---|---|---|---|
| 位置误差 中位 | 7.925 m | 0.084 m | **0.075 m** |
| 角度误差 中位 | 119.99° | 10.005° | **8.561°** |
| 位置误差 < 0.2 m | 0.4% | 82.1% | **88.4%** |
