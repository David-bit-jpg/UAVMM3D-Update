# 整图对比实验规格（对比方法按 AntiUAV3D 表 2 的方式跑）

## 背景
无人机单目 3D 检测（位置 + 尺寸 + 三个旋转角）。我们的方法从仿真迁移到真实（MAV6D 数据集）。
导师要求的对比口径：**所有方法训练都只用我们的仿真数据（有标签），可以在真实图像上用伪标签自训练（不用真实标签），最后只看真实测试集结果。**
对比方法要像 AntiUAV3D 论文（AAAI 投稿）表 2 那样跑：**官方代码、整幅图像输入、方法自己检测目标**，只做必要的最小改动。

## 数据（统一，所有方法读同一份）
`@@ROOT@@/data/cmpfull/<split>/images/*.jpg` + `@@ROOT@@/data/cmpfull/<split>/labels.pkl`（导出脚本 `cmpfull/export_data.py`，进度看 `cmpfull/export.log`）
- `sim_train`：仿真训练 19,490 帧，1280×720，带标签
- `sim_val`：仿真验证 4,924 帧，1280×720，带标签
- `real_train`：真实 MAV6D 训练集图像 12,866 帧，512×288（已去畸变），**无标签**（只给以后的自训练用）
- `real_test`：真实测试 4,800 帧，512×288，labels.pkl 里 boxes 为空（真值在 gt_sealed.pkl，**方法代码不许读**）

`labels.pkl` = list，每帧 dict：`file`（相对 split 目录）、`seq`、`frame`、`W`、`H`、`K`（3×3，对应存下来的这张图）、`boxes`（list of dict）：
- `center` (3,)：**框几何中心**（不是 KITTI 的底面中心），相机系 x 右、y 下、z 前，单位米
- `lwh` (3,)：物体自身 x/y/z 轴方向的边长（米）
- `euler_xyz` (3,)：`R = scipy Rotation.from_euler('xyz', euler_xyz).as_matrix()`，物体系 → 相机系
- `R` (3,3)：同上的矩阵
- `cls`：机型名。**所有机型合并成一个类别 drone。**
- 8 个角点 = `(PROTO * lwh) @ R.T + center`，`PROTO = {±0.5}^3`

距离：仿真 ≤ 8 m，真实 1.6–5.6 m。

## 输入分辨率（统一）
每张图缩放到 **960×540，再在下方补零到 960×544**，K 按同样比例缩放（只缩放，不平移）。训练和测试一样。
（仿真 1280×720 缩小，真实 512×288 放大；这是对比方法的正常用法，不做我们方法里的焦距归一化 / 裁窗。）

## 改动范围（最小改动）
1. 数据加载：读上面的 labels.pkl 代替 KITTI 标签/标定。只保留一个类别。KITTI 专有的东西（底面中心、occlusion/truncation 过滤、只有 ry）按上面的定义换掉；过滤规则如需保留，用投影 2D 框在图内这类等价条件。
2. 朝向（**已更新，用 KITTI 式角度 `ypr`，不要用 euler_xyz 做监督**）：labels.pkl 每个框新增 `ypr` = [yaw, pitch, roll]（弧度），定义和换算在 `cmpfull/angles.py`：`R = Ry(yaw)·Rx(pitch)·Rz(roll) @ R0`。**yaw 就是 KITTI 的 ry**（绕相机 y 轴，ry=0 时机头/长边沿相机 +x），原方法里所有 ry 的处理（alpha↔ry、用 ry 旋转角点）原样作用在 yaw 上；水平翻转增广 = (yaw, pitch, roll) → (π−yaw, −pitch, roll)（注意本约定 pitch=Rx 是绕机头轴、roll=Rz 是绕侧向轴）；pitch、roll（仿真里都在 ±30° 内）用同一种编码（同 bin 数 + 残差）直接回归绝对角，损失同权相加。KITTI 尺寸顺序 (h, w, l) = (lwh[2], lwh[1], lwh[0])。predict 输出 boxes9d 的角度用 `angles.euler_xyz_from_ypr` 换回 euler_xyz。旧的 euler_xyz 有分支跳变（第 2 个角 ±90° 限制导致另外两个角跳 180°），不要用来监督。
   - 注意：方法里如果用 ry 去算别的东西（例如 alpha ↔ ry 换算、用 ry 旋转角点做关键点/深度几何），改成用完整 R。alpha（观察角）的做法若只对 ry 有意义，就对 R 的偏航分量照原逻辑做，其余两个角直接回归绝对角。在 NOTES 里写清楚。
3. 尺寸：保留方法原有的尺寸编码（均值 + 残差等），类均值用 sim_train 统计（单类）。
4. 深度/位置：保留原方法。投影中心用框几何中心的投影。
5. 超参数：保留官方默认（骨干、优化器、学习率、batch、增广）。轮数：按官方「迭代次数」大致对齐（官方 KITTI 训练集 3,712 张），在 NOTES 里写你选的轮数和理由；如果单个 GPU 跑不了官方 batch，可以按线性规则缩 batch+学习率。
6. 不要加我们方法的任何东西（裁窗、焦距归一化、AdaBN、多模态）。

## 必须提供的入口（放在你的方法目录下）
- `train.sh`（或 .py）：参数 `--labels <pkl 路径>...`（可给多个，后面自训练会追加 `real_train` 的伪标签 pkl，格式同 labels.pkl）、`--out <输出目录>`、`--seed`、`--epochs`；
- `predict.py --ckpt <> --split <real_test|sim_val|real_train> --out <pkl>`：输出 list，与该 split 的 labels.pkl **同序**，每帧 dict(`seq`, `frame`, `boxes9d` (n×9：center, lwh, euler_xyz，**我们的定义**)、`scores` (n,))；NMS 后最多保留 10 个；
- `NOTES.md`：改了哪些文件哪些地方、为什么、轮数选择、每轮耗时和显存、冒烟测试结果。

## 规则（重要）
- 在 `@@ROOT@@/Open3DUAVDet/output/camnorm/geo1004/cmpfull/<方法名>/` 里工作：先把 `E:/ext_repos/<方法名>` 复制过来再改，**不要改 E:/ext_repos 下的原始仓库**，不要改其他目录。
- Python 环境：`python`（Windows，Git Bash 可用），运行前 `export PYTHONUTF8=1`。缺包先看能否不用；必须装的话只用 pip 装纯 Python 包并在 NOTES 里写明；需要编译 CUDA 算子的，优先用官方仓库自带的纯 PyTorch 实现（例如 deformable attention 的 pytorch 版本），不行再编译。
- **GPU 是共享的，会崩**：每次用 GPU 前先 `nvidia-smi` 看占用，显存已用 > 16 GB 或利用率持续 > 80% 就等；代码里加 `torch.cuda.set_per_process_memory_fraction(0.2)`（环境变量 GPU_FRAC 可覆盖）。**只做冒烟测试**：训练 ≤ 200 次迭代、预测 ≤ 50 张，确认损失下降、输出格式对、能跑通。**不要启动正式训练。**
- Windows 上 DataLoader 多进程容易锁死：num_workers 用 0 或 2。
- 只能结束你自己启动的进程，并且只按 PID 结束（先确认命令行），绝对不要按进程名批量杀。
- 不要读 `real_test/gt_sealed.pkl`。
- 最后用 sim_val 的 50 张做一次格式自检：把 predict.py 输出的 boxes9d 角点投影回图像，和 labels.pkl 真值角点投影比一比（冒烟模型不准没关系，检查的是坐标/角度约定一致：可以用真值喂进你的编码→解码往返，误差应接近 0）。
