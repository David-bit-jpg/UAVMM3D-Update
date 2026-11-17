# 我们的检测器按对手方法的协议跑三行（11-16）

目的：主表里给我们自己的检测器（CenterDet，ResNet-34-FPN）加三行，协议与五个对手方法（GUPNet / MonoCD / MonoDGP / MonoDETR / MonoCoP）完全相同，两个目标域：
MAV6D（室内，源域 = 室内 UE5 仿真 `in`）和 OutdoorUAV3D（AntiUAV3D 室外子集 v2 `antifs`，源域 = CARLA 室外仿真 `out`）。

| 行 | 做法 |
|---|---|
| 1. 只用仿真 | 种子 0 的纯仿真检测器单独跑（不集成、不翻转），预测尺寸解码 |
| 2. + 自训练（ST） | 用行 1 的模型给目标域无标签训练图打伪标签：每帧取分数最高的 1 个框，分数排前 80%（分位数阈值）的帧留下，中心/尺寸/朝向都用它自己的框；仿真标签 + 伪标签拼接，从 ImageNet 骨干重新训练，轮数同纯仿真训练，取最后一个存档 |
| 3. + PST 伪标签 | 同上，伪标签换成对手 +PST 行用的同一张表（同帧、同框） |

## 伪标签

| 目标域 | +PST 表（与对手完全相同） | 帧数 | +ST（我们的种子 0 纯仿真模型） |
|---|---|---|---|
| MAV6D | `D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl`（中心 = PST 第 1 步伪中心，尺寸 = s̄，朝向 = 红外/LiDAR 教师伪朝向，不筛） | 5,227 / 12,866 | 见 `labels/in2mav_st.json` |
| OutdoorUAV3D | `D:/UAV3D_pack_update_1112/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl`（已核对 = `index_pseudo_srcx_out2antifs.pkl` ∩ `pseudo_tr1.npz` gate 帧，249 帧、中心逐位相同） | 249 / 3,293 | 见 `labels/out2antifs_st.json` |

转换：`lib/make_labels.py` 把表写成我们数据层的索引（目标域缓存 train 划分的 index.pkl 拷贝，只换 boxes9d = [中心, lwh, euler_xyz]，float64，不带 pseudo 标记 = 所有头都监督，同对手）。
往返核对：8 个角点用对手 labels.pkl 的内参投影，表里的框 vs 写回读出的框，最大差 0（要求 < 1e-6）；数据层转 float32 后最大 1.3e-5 px（和仿真标签同一处理）。
缓存内参 K_in 与对手 labels.pkl 的 K 一致（MAV6D 逐位相同；室外 = K_raw × 512/1920，半像素中心约定，差 0）。
索引写在缓存拷贝 `D:/S2MUAV3D/ourdet_baseproto_cache/<缓存>/train/`（原缓存未动）。

## 训练

- 配方 = 我们纯仿真检测器自己的配方（`srcx/stage1_det.sh`）：配置 `sim_full_r34_geo_tr_mix_sym8_mblur_ramp`（室内）/ `sim_out_r34_geo_ramp_out`（室外），5 轮，批 8，OneCycle，rot 权重 2→16 爬升，从 ImageNet 骨干开始，种子 0，取第 5 轮存档。
- 拼接：`lib/ourdet_concat.py`（torch ConcatDataset，打乱 = 均匀采样）。仿真部分 = 配方原样；真实伪标签部分 = 同一份数据配置，只换缓存路径、索引文件、输入归一化（目标域自己的 NORM，测试时本来就用它）、可见度过滤关（真实缓存没有 vis_score）。真实缓存是 512×288 npy，裁窗缩放增广（VIEW_AUG）对它不起作用；其余增广（翻转、光度、噪声、模糊、目标光环、运动模糊）与仿真同一套。
- 配置：`tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/{in2mav,out2antifs}_{pst,st}.yaml`；入口 `ourdet_train.py`（= 原 tools/train.py main()，只把运行目录改到 D:、注册拼接数据集）。
- 室内仿真缓存读 F:/mmcache/full_v2（与 E: 逐字节相同的拷贝）。存档在 `D:/S2MUAV3D/ourdet_baseproto_models/`。

## 评测

单模型、不集成、不翻转、不用 s̄：目标域配置 `srcx/real_<combo>.yaml`，`SIZE_SOURCE pred` + `SIZE2D_MODE width`（深度由预测 2D 宽和模型自己预测的 3D 尺寸解出），每帧取最高分 1 个框（`lib/ourdet_infer.py`，后处理就是 `srcx/lib/det_infer_x.run`）。
打分 = `srcx/lib/score_x.py`（即对手用的 pack score.py 的度量；MAV6D 用 real_test 规则，室外用 outdoor_test 规则 + v2 封存真值），每个运行只打一次测试集。
对照：`preds/srconly_test.pkl`（srcx 链里）是 3 种子集成 + 翻转 + s̄，不是这里的行 1。

## 追加两行（11-16 用户批准）

### CenterNet + DANN（`ourdet_dann_train.py`，`dann_eval.sh`）
照 GUPNet + DANN 对手（`cmpfull/gupnet/code/tools/uda_train.py --method dann`，运行记录 `D:/cmpfull_runs/gupnet_uda_dann_s0/run_info.json`）：
- 起点 = 行 1 的种子 0 纯仿真检测器；域分类器与 uda_train.py 相同（3 层 3×3 步长 2 卷积 + 全局池化 + MLP，无 BN），接在颈部 FPN 输出 features_2d（512 通道，步长 8）上，经梯度反转层，系数 0.1×(2/(1+e^(−10p))−1)。
- 每步：仿真 batch = 真值检测损失 + 域损失（标签 0）；真实无标签训练图 batch（无增广，只读图像）= 只有域损失（标签 1）。
- 检测损失权重 = 纯仿真训练结束时的值（rot 爬升终点 16）；学习率常数 = 纯仿真峰值的 0.1（1e-4），AdamW（配方同族，wd 0.01），域分类器 ×10，梯度裁剪 35，BN 统计冻结。
- 预算：迭代数 = 我们 5 轮仿真配方的迭代数（室内 11,315、室外 3,355，批 8；GUPNet 是 40×3,712 样本 = 18,560 步）。取最后存档。
- 运行记录 `D:/S2MUAV3D/ourdet_baseproto_models/dann/<combo>_dann/run_info.json`。

### 上界（真实有监督）（`ub_eval.sh`，配置 `ourdet_baseproto/{in2mav,out2antifs}_ub.yaml`）
- 配方同纯仿真（ImageNet 起步、种子 0、只用 RGB），只把训练数据换成目标域真实训练集 + 真值；不设验证集（训练时不碰真实测试集）。
- 轮数按“迭代数与 5 轮仿真配方相当”：MAV6D 7 轮×1,609 = 11,263 步（仿真 11,315）；OutdoorUAV3D 8 轮×412 = 3,296 步（仿真 3,355）。
- MAV6D 真值 = 缓存 index.pkl（12,866 帧）。**OutdoorUAV3D 真值来自封存的训练集标签 `D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/train_gt.pkl`（box9d_ours，3,293 帧），用户明确批准只用于这一行**，写成 `index_ub_gt.pkl`（`lib/make_ub_outdoor.py`），别无他用。

## 运行中的偏差 / 事故

- 数据加载：首次以 workers 0 跑时 CPU 成瓶颈（约 1 s/步），改为线程池批加载（`lib/thread_loader.py`，6 线程，不起子进程；每轮按种子打乱，与 DataLoader shuffle 等价，不 drop_last）。随机流与 spawn workers 不逐位相同。
- 15:27 主机内存提交量耗尽，当时我的 6 个进程全部报错退出（均未到第 1 个存档），从头重跑；包装脚本此后遇主机内存错误自动重试/续训。之后同时最多约 3 个我的训练（`sequencer.sh`）。
- 缓存位置：D: 机械盘随机读严重拥堵（队列 10–25，约 4 MB/s）。MAV6D 上界第 1 轮读 E: 原缓存，第 2–7 轮从第 1 轮存档续训，改读 F: 上的逐字节拷贝 `F:/ourdet_baseproto_cache/mav6d_cn_obj/train`（index.pkl cmp 相同；rgb.npy 大小相同，末 200 MB cmp 相同）；MAV6D +ST 的伪标签索引与训练也在这份 F: 拷贝上。MAV6D +PST、两个 DANN、室外各行读 D: 拷贝。
- 16:42–16:49 我杀自己的 gpu_slot 等待进程时留下了陈旧的 `gproto/.gpu_slot.lock`，16:49 已删除。
- 一个本以为已停掉的旧 sequencer 实例重复启动了 MAV6D 行 1 训练集推理和 MAV6D +ST；17:35 已杀掉重复的那份（都还没有存档）。`logs/train_in2mav_st.log` 前约 5 分钟混有被杀那份的行，运行目录里的 train_*.log 是干净的。
- 测试真值诊断（带符号的深度误差）被权限检查拦下，没有跑；只做了不读真值的尺寸 / 深度对比。

## 结果（测试集，各打分一次；单模型、不翻转、预测尺寸）

| 目标域 | 行 | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|---|
| MAV6D | 只用仿真（种子 0） | 13.09 | 0.666 | 5.920 | 20.42 | 69.79 | 32.59 |
| MAV6D | + ST | 14.88 | 0.564 | 4.170 | 20.16 | 73.96 | 37.52 |
| MAV6D | + PST | 23.84 | 0.485 | 2.943 | 20.89 | 95.88 | 47.10 |
| MAV6D | + DANN | 7.33 | 0.894 | 6.793 | 23.75 | 80.60 | 26.39 |
| MAV6D | 上界（真实有监督） | 45.46 | 0.276 | 0.053 | 17.62 | 99.94 | 65.36 |
| OutdoorUAV3D | 只用仿真（种子 0） | 2.93 | 0.470 | 36.975 | 26.31 | 71.95 | 17.22 |
| OutdoorUAV3D | + ST | 5.47 | 0.253 | 26.160 | 23.06 | 59.68 | 17.67 |
| OutdoorUAV3D | + PST | 14.77 | 0.128 | 15.279 | 23.21 | 72.83 | 30.01 |
| OutdoorUAV3D | + DANN | 0.005 | 0.568 | 20.393 | 23.50 | 52.74 | 13.97 |
| OutdoorUAV3D | 上界（真实有监督） | 0.61 | 0.402 | 1.678 | 24.42 | 69.40 | 26.56 |

（+ST 伪标签：MAV6D 10,293 / 12,866 帧，室外 2,634 / 3,293 帧，阈值见 `labels/*_st.json`。打分 json：`preds/<combo>_<行>_test.pkl.score_<split>.json`。）

## 11-17 追加：上界行加上与仿真相同的裁窗缩放增广（用户批准，正文主表）

为什么：旧上界（真实有监督）测试 UDS 只有 26.56，而它在自己训练集上 56.12。原因是训练集里的无人机近而大（中位 12.9 m），测试集远而小（中位 38.9 m、8.4 px）；仿真训练用裁窗缩放增广（VIEW_AUG，虚拟换焦距，内参同步改）覆盖了这一档，真实 512×288 npy 缓存却用不上它。

做法（只新增文件，没改任何共享库文件）：
- `lib/build_ub_jpeg.py`：把 3,293 帧真实训练图的 1920×1080 原图（已去畸变；512×288 缓存就是它的 INTER_AREA 缩小，已逐位核对）写成与仿真缓存同格式的 JPEG 存储 `F:/ourdet_baseproto_cache/outdoor_real_fs_jpeg/train/`（RGB 顺序、质量 95、内参 K_in = 原图内参），标签仍是 `index_ub_gt.pkl`（封存训练标签，只用于这一行）。
- `lib/ub_zoom_dataset.py`：数据层 = 仓库的 MMCache_Det_Dataset 原样走 VIEW_AUG 代码（同一随机流程、t_range [1.0, 4.6] 均匀、max_upscale 2、窗口放置与丢框规则、内参公式），只加一个开关 `zoom_out_max`。原因：VIEW_AUG 只会裁小窗口（放大），仿真里本来就有远处的机，所以够用；真实训练帧都近（原生 t = log(虚拟深度/对角线) 中位 2.47、p95 3.55），只裁不缩就永远变不远。`zoom_out_max = 8` 允许窗口最大到原图 8 倍：原图按比例缩小放到均值色画布上（输入归一化后正好为 0），内参用同一公式精确更新。8 = e^(4.6−2.47) 取整，只由训练标签决定（让中位训练帧能到配方的 t 上限）。
- 配置 `tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml`；入口 `ub_zoom_train.py`（= ourdet_train.py + 注册该数据层）；包装 `ubzoom_eval.sh`（= ub_eval.sh 换名字和入口）。
- CPU 自检 `lib/check_ub_zoom.py`：20 个增广样本用增广后的内参投影真值 3D 框，框都落在机身上（`check/ubzoom_contact.png`）。1,500 个增广样本的 3D 框投影宽度（512 输入）分位 p1/p5/p10/p50/p90 = 4.5/5.4/6.3/25.2/105.6 px，宽度 < 9 px 占 21%；t 分位 p5/p50/p95 = 1.22/2.84/4.38。增广前（原 512 缓存）宽度 p1/p50 = 10.0/36.3 px、t 范围 2.06–3.73（`check/ubzoom_contact.json`）。

训练：配方与旧上界相同（ImageNet 起步、室外配方、种子 0、批 8、8 轮 × 412 = 3,296 步、线程加载 6 线程、无子进程），一次跑完、未续训、未 OOM。第一次启动因数据层名字不在 DatasetTemplate 的分派表里（没有预处理器）在第 1 步前报错，改为实例内 DATASET 名沿用 'MMCache_Det_Dataset' 后重跑（日志 `logs/train_out2antifs_ubzoom.attempt1_preproc_bug.log`）。存档 `D:/S2MUAV3D/ourdet_baseproto_models/.../out2antifs_ubzoom/DET_s0/ckpt/checkpoint_epoch_8.pth`。

评测与旧行完全相同（单模型、不翻转、预测尺寸，score_x.py，antifs_test 只打一次）：

| RealDrone3D | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|
| 上界（旧，无裁窗缩放） | 0.61 | 0.402 | 1.678 | 24.42 | 69.40 | 26.56 |
| 上界（加裁窗缩放增广） | 5.73 | 0.195 | 1.155 | 24.73 | 84.91 | 34.39 |

按距离（测试）：0–25 m UDS 34.07（AP 4.53）、25–50 m 34.08（AP 6.69）、50 m 以上 36.84（AP 7.05，检出率 69.8）。旧行三档 AP 为 3.49 / 0 / 0。
诊断（同模型在自己训练集上，封存训练标签）：UDS 38.73、AP 13.26、ATE 0.164（旧 56.12 / 31.54 / 0.100），即同样步数下增广让训练集拟合变弱。
仍低于完整 PST 的 41.73。
文件：`preds/out2antifs_ubzoom_test.pkl`、`preds/out2antifs_ubzoom_test.pkl.score_antifs_test.json`、`preds/out2antifs_ubzoom_TRAINDIAG.score_train_sealed.json`（`lib/score_traindiag.py`，用 score_x 的同一度量代码；对旧模型复现 56.12）。
