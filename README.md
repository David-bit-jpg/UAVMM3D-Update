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
