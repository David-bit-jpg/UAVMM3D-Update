# 无人机单目 3D 检测：仿真 → 真实（最终方法，11-10 无标签版）

**做什么**：只输入**一张**彩色图，输出画面里无人机的 3D 位置、尺寸和朝向；部署时每帧只用本帧图像，不看任何其他帧。
**怎么训练**：标签只来自仿真；真实数据（MAV6D）只用训练集的**无标签**图片。红外和 LiDAR 只在仿真里当"老师"，部署只用彩色图。
**真实标签怎么用**：既不当任何训练目标，也不在任何一次运行里被读入（伪标签、常数、权重 w、筛帧都不读）；只在整条链冻结之后，对验证集 / 测试集各打一次分。

这个分支只放代码（不含任何权重、数据和中间结果）。论文所写的方法 = 本文件描述的链；旧版说明在 `release/history/`（11-06 多帧版）和 `release/single_frame/README.md`（11-09 精简版），都已被本版取代。

> **标签使用协议**：MAV6D 是**开发基准**。开发阶段（11-10 以前）在 MAV6D 上定下的取舍原样沿用、论文如实说明：距离修正系数 **LAM = 0.75**、尺寸联动指数 **α = 0.5**、朝向学生的配置（教师组成、一致阈值 10°、镜像损失权重 12 等）。11-10 起事先登记（`output/camnorm/geo1004/gproto/nolabel/PREREG.md`）后整条链重跑一次：凡是曾经看过真实标签的步骤都改成不看标签的规则（下面 §1 的"无标签规则"），跑完只评测一次，结果不回头改任何设置。

---

## 1. 方法链（按执行顺序）

| # | 步骤 | 做了什么 | 网络 / 种子 |
|---|---|---|---|
| 0 | 上游（仿真） | 仿真检测器（ResNet-34 中心点检测，**距离 = 焦距 × 横向跨度 / 预测 2D 宽度**）；红外关键点老师；被老师塑形的朝向网络（红外蒸馏 / LiDAR / 红外+LiDAR 各 2）；高度网络（学 LiDAR 结构点）；第二段裁窗测距网络第 1 步（仿真里学红外剪影） | 仿真检测器 5（自训练起点用种子 0–2）；测距网络第 1 步种子 5、6 |
| 1 | 自举检测器 ST2 → 第一批位置伪框 | ST2 五集成在真实训练帧上做**单帧**检测（不开热图时序背景）；第一代第二段测距网络 V2Pp_s0..4 给每个检测一个对数距离修正量 | ST2 5、V2Pp 5 |
| 2 | **位置伪标签（无标签规则）** | 修正量 c = **LAM × 全训练集修正量中位数**（LAM = 0.75），沿视线缩放 ST2 的位置；**只按检测置信度 ≥ 0.5 筛帧**；尺寸 = 检测器在无标签验证集图像上的自标定常数 | — |
| 3 | 检测器自训练（弱增强） | 从仿真检测器出发微调 3 轮；增强只留水平翻转 | **3（种子 0、1、2）** |
| 4 | 测距网络第 2 步（适应真实） | 真实训练裁窗 = 纯仿真检测器单帧最高分检测且与本帧位置伪标签相距 < 0.5 m；伪目标 = log(伪距离 / 检测距离)；中间域 = 仿真机身贴到**单帧**真实训练背景（背景窗用新检测器的单帧检测避开原有无人机，不读真值） | 2（种子 5、6） |
| 5 | 朝向学生 + 高度网络：只重估 BN | 朝向学生（读 8 个角点 → PnP 得朝向和第二个距离）**不重训**，沿用开发阶段权重，只在新检测器的真实训练窗口上重估 BN；高度网络同样只重估 BN，再出逐帧高宽比 | 朝向学生 2（S6I 种子 0、1）；高度网络 1 |
| 6 | **合并权重 w（无标签规则）** | 两种测距（红外教的测距网络 / 角点 PnP 距离）在**仿真验证集**上的对数误差稳健散布（1.4826·MAD，先减中位）→ 逆方差权重 w = σ²_红外 / (σ²_红外 + σ²_角点)；只读仿真标签。主结果 w = 0.158 | — |
| 7 | 两教师合并伪距离 → 测距网络第 3 步 | 朝向学生在全部真实训练帧上读角点（原图 + 镜像 = 4 个读数）→ PnP 深度 z_kp；`log z_合 = (1−w)·log z_伪 + w·(log z_kp − b_k)`，b_k = 颜色原型 k 上 median(log z_kp / z_伪)（无标签训练集） | 2（种子 5、6） |
| 8 | 部署（单帧） | 检测器 3 集成 + 翻转 → 测距网络 ×2 修正距离 → 朝向学生 ×2（原图 + 镜像）出朝向和 z_kp → 两种距离按第 7 步同一公式、同一个 w 合并（部署的 b_k 取无标签验证集**图像**）→ 高度 = 该帧颜色原型的高宽比常数 × 平均(长, 宽)；**置信度不重排**（AP 直接用检测器置信度） | — |

**常数从哪来（都不读标签）**：颜色原型中心、各原型的 c_k 与高宽比 r_k 用无标签训练集；部署偏移 b_k 和常数尺寸 s̄ 用无标签验证集**图像**；高度网络只用来给 r_k 常数和训练时 PnP 用的机身高度。

**什么是单帧**：部署时每一步只用本帧图像 + 上面这些预先在无标签图像上求好的全局常数；伪标签、背景库、BN 统计、常数也都按单帧处理（不做任何跨帧统计，不做时序背景）。
**唯一的间接例外**：最早的自举检测器 **ST2**（产生第一批位置伪框和朝向裁窗）在开发阶段训练时，用过按相邻帧筛选的伪标签；朝向学生沿用开发阶段的权重（只重估 BN）。这两处之后的每一步都严格单帧、不读真实标签。

**无标签规则对比开发版（11-09）改了什么**：位置伪标签不再用"对照训练集真值挑出"的不确定度筛选，只按置信度；w 不再在验证集上选，改为仿真验证集逆方差；去掉置信度重排（原温度 τ 是验证集选的）；按机型（来自序列路径 = 真值）分组的统计全部去掉；下游脚本只读"去真值帧表"（`make_train_nogt_index.py`，只留 seq / frame / 内参），不再打开带真值的索引。

---

## 2. 结果（真实 MAV6D；验证集 2336 帧 / 测试集 4800 帧；冻结后一次性评测）

| 配置 | 验证 UDS | 验证 AP | 测试 UDS | 测试 AP | 测试位置误差 m（中位） | 测试尺寸误差 cm | 测试朝向误差° / 航向中位° | 测试检出率 % |
|---|---|---|---|---|---|---|---|---|
| **最终（LAM = 0.75，w = 0.158）** | **62.49** | **48.33** | **70.16** | **56.44** | **0.226（0.138）** | **1.81** | 13.54 / 5.81 | 99.8 |
| 同一冻结链，不做部署合并（w = 0） | 61.51 | 46.73 | 68.26 | 52.97 | 0.239（0.156） | 1.81 | 13.53 / 5.81 | 99.8 |
| 敏感性：LAM = 1（w = 0.114） | 59.80 | 43.63 | 69.70 | 55.46 | 0.222（0.144） | 1.80 | 13.63 / 6.00 | 99.7 |
| 敏感性：LAM = 0.5（w = 0.141） | 61.45 | 46.52 | 64.56 | 46.63 | 0.277（0.196） | 1.85 | 13.59 / 5.85 | 99.8 |

- LAM 在开发阶段于 MAV6D 上定为 0.75；LAM = 1 / 0.5 是同一条链换系数重跑（检测器、第二段、整链全部重做），只报告、不用于选择（放补充材料）。w = 0 行是同一冻结链去掉部署合并，作参照。
- 验证集朝向误差 20.57°（航向中位 12.83°）、位置误差 0.312 m（中位 0.174）。测试近 / 中 / 远（<3 / 3–4 / ≥4 m）：AP 57.65 / 62.74 / 48.06，UDS 70.36 / 74.02 / 65.31。
- 源记录：`results/nolabel_1110/`（`chain_nl{075,100,050}.txt` = 整链运行记录，`results_nl*.json` = 全部指标与常数，`detector_nl*.txt` = 检测器训练 / 推理记录，`selfcheck_nl.txt` = 无真值自查）。
- **平均精度（AP）**：按位置误差阈值 0.1 / 0.2 / 0.4 / 0.8 m 的 3D AP 取平均（MAV6D 里无人机只有 1.6–5.6 m 远，按比例缩小了常用的 1/2/4/8 m 阈值）。
- **位置 / 尺寸 / 朝向误差**：只在检中的帧上算（置信度最高的框与真值 2D IoU ≥ 0.1）；朝向按四旋翼对称算（转 90°/180°/270° 不算错）。**检出率**：真值被检中的比例。
- **UDS**：MAV6D 原文没有给公式，这里按 nuScenes NDS 的写法自己定义：0.5 × AP + 0.5 × 平均(1 − min(1, 位置误差/1 m), 1 − min(1, 尺寸误差/10 cm), 1 − min(1, 朝向误差/1 rad), 检出率)。代码见 `output/camnorm/geo1004/cmpeval/paper_table.py` 的 `metrics()`。

---

## 3. 怎么跑

### 3.0 环境与路径（一次）

```bash
pip install -r release/requirements.txt        # Python 3.10；torch / torchvision 按显卡自行安装（研究时 CUDA 版）
python release/fix_paths.py --mmcache <缓存根目录> --mav6d <MAV6D 原图根目录> \
       --torch-home <ImageNet 预训练权重目录> --data-collect <仿真录制数据根目录>
```

代码里的绝对路径都换成了占位符（`@@ROOT@@` 仓库根、`@@MMCACHE@@` 缓存根、`@@MAV6D@@`、`@@TORCH_HOME@@`、`@@DATA_COLLECT@@`），`fix_paths.py` 把它们换成本机路径；不带参数时用原机器的路径（`E:/mmcache` 等），仓库根自动取。提交代码前用 `python release/fix_paths.py --reset` 换回占位符。
研究时加的"防改动哈希核对"默认关闭（路径一换哈希必然变），要核对就设 `UAV_STRICT_HASH=1`。

### 3.1 数据（不在仓库里，要自己准备）

| 数据 | 生成命令 |
|---|---|
| MAV6D 相机归一化缓存（官方物体框；train / 按序列留出的 val / 官方 test） | `python tools/build_mav6d_cache.py` → `<缓存根>/mav6d_cn_obj` |
| 仿真缓存（≤ 8 m 整帧，按序列 8:2 划分；彩色 JPEG + 多焦距裁窗） | `cd tools && python run_full_v2.py`（第 1 步会用 `build_near_subset.py` 重新生成帧清单 `cfgs/subsets/full_v2/`，该清单 6 MB 没放进仓库） |
| 仿真红外 / LiDAR 边车（老师用） | `python tools/build_mm_sidecar.py --cache <缓存根>/full_v2 --root <仿真录制数据根> --split train test --workers 8` |
| ImageNet 预训练权重 | 放到 `TORCH_HOME`，脚本不会联网下载 |

### 3.2 上游与开发阶段网络（§1 第 0、1 步 + 朝向学生）

```bash
bash release/run_upstream.sh                     # 仿真检测器、ST2、第一代测距网络、朝向 / 高度网络等；已完成的步骤自动跳过
bash release/single_frame/run_train_final.sh     # 开发阶段（11-09）的老师塑形网络、朝向学生 S6I_s{0,1}、测距网络第 1 步 V2P_s{5,6}_st
```

最终链只从第二个脚本里取**朝向学生**（不重训）和**测距网络第 1 步**（纯仿真）；它后面那些依赖旧检测器的步骤会被最终链用新检测器重做。统一评测帧表 `cmpeval/{val,test}_meta.npz` 用 `release/run_final_3seed.sh` 第 0 步那条命令生成（真值只用于打分）。

### 3.3 最终链（§1 第 2–8 步，无标签）

```bash
bash release/run_final_nolabel.sh                        # 主结果：LAM = 0.75（臂 nl075）
LAM=1.0 ARM=nl100 bash release/run_final_nolabel.sh      # 敏感性；LAM=0.5 ARM=nl050 同理
```

内部顺序：去真值帧表 → ST2 单帧检测 + 第一代修正量 → 位置伪标签（`make_nolabel_pseudo.py`）→ 检测器 3 种子弱增强自训练 + 部署推理（`proto/run_proto.py` 的 `train()` / `infer()`，配置 `st2nl075wa_mav6d_r34_geo_full_tr.yaml`）→ 无标签整链 `output/camnorm/geo1004/gproto/nolabel/run_chain_nl.sh <arm>` → 无真值自查 `selfcheck_nl.sh`。结果写到 `gproto/nolabel/results_<arm>.json`，运行记录在 `gproto/nolabel/<arm>/logs/`。

`run_chain_nl.sh` 每一步输出已存在就跳过（可续跑），内部是：

| 阶段 | 脚本（都在 `output/camnorm/geo1004/gproto/nolabel/`） | 作用 |
|---|---|---|
| 三路推理 | `eval_dump_nl.py`、`dump_train_kz_nl.py` | 新检测器在真实训练集全部帧上的单帧检测（只存预测，不存真值）；训练集朝向窗口（间隔 3 / 间隔 1） |
| 测距网络第 2 步 | `xpaste_bank_sf2_nl.py`、`extract_nbm_nl.py`、`secondstage_v2_nl.py V2Pp` | 单帧背景库；真实裁窗 + 伪目标；种子 5、6 |
| 朝向 / 高度 | `adabn_all_nl.py`、`kps_nl.py`、`gproto/wf23/fusetrain/recut_mirror_t.py` | 朝向学生 BN 重估；高度网络 BN 重估 + 高宽比读数；原图 + 镜像重裁读角点 |
| w | `w_sim.py` | 仿真验证集逆方差（只读仿真标签） |
| 合并伪目标 + 第 3 步 | `final_unc_nl.py`、`ztrain_nl.py`、`build_targets_nl.py`、`secondstage_v2_nl.py V2Rp` | 训练帧角点距离 z_kp → 合并伪距离 → 测距网络第 3 步（种子 5、6） |
| 冻结 + 一次性评测 | `final_unc_nl.py`、`chain_eval_m_nl.py`、`fuse_eval_nl.py` | 部署输出、整链组装（置信度不重排，α = 0.5）、两种距离合并、对 val / test 打分（真值只在这里从 `cmpeval/<split>_meta.npz` 读入） |

GPU 训练不逐位可复现，重训结果会和上表有小幅差别。研究时在 Windows + Git Bash 上跑；Linux 也可以（`env.sh` 自动处理路径分隔符）。

---

## 4. 代码在哪

| 内容 | 文件 |
|---|---|
| 检测网络、数据集、损失（库） | `uavdet3d/`（`model/detectors/center_det.py`、`model/dense_head_2d/center_head.py`、`datasets/mmcache/`） |
| 训练 / 评测入口 | `tools/train.py`、`tools/eval_camnorm.py`、`output/camnorm/geo1004/eval_dump.py` |
| 检测器配置 | `tools/cfgs/models/uavdet_3d/camnorm/`：仿真检测器 `sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml`；最终自训练 `st2nl075wa_mav6d_r34_geo_full_tr.yaml`（敏感性 `st2nl100wa_…`、`st2nl050wa_…`；都基于 `st2pluswa_…` → `st2plus_…` → `st_…`，只换伪标签索引） |
| 无标签最终链 | `output/camnorm/geo1004/gproto/nolabel/`（`PREREG.md` 事先登记、`run_chain_nl.sh` 驱动、各 `*_nl.py`、`make_nolabel_pseudo.py`、`make_train_nogt_index.py`、`w_sim.py`、`selfcheck_nl.sh`） |
| ST2 修正量 | `output/camnorm/geo1004/gproto/strict_deltas.py`；ST2 单帧检测 `tools/annot_audit/dump_real_preds.py --bg none` |
| 检测器训练 / 推理驱动 | `output/camnorm/geo1004/proto/run_proto.py`（`ARMS['nl075']` 等）、`proto/dump_orient_p.py`（朝向裁窗窗口） |
| 测距网络 | `output/camnorm/geo1004/secondstage_v2.py`（无标签链用副本 `secondstage_v2_nl.py`，只改背景库读取） |
| 朝向学生 / 伪朝向（开发阶段） | `gproto/wf21/orient_uda/`（`pseudo_sets.py`、`ou_train.py`）、`gproto/wf23/fusetrain/recut_mirror_t.py` |
| 组装 + 指标 | `gproto/wf23/mirror/chain_eval_m.py`（副本 `chain_eval_m_nl.py`）、`output/camnorm/geo1004/cmpeval/paper_table.py` |

`output/` 在 `.gitignore` 里（研究时产物都写在那里），这里只把其中的代码和少量控制文件强制加进了版本库，相对路径与原工作目录一致，所以脚本之间的导入不用改。无标签链里的 `*_nl.py` 是开发版脚本的副本，每个文件开头写明了相对原脚本改了什么。

---

## 5. 不包含 / 注意

- **不含**：任何权重（`*.pth` / `*.pt`）、数据（MAV6D、仿真录制、缓存）、中间结果（`*.pkl` / `*.npz` / `*.npy`）、图片、日志。只有代码、配置和小的结果文本（`results/`）。
- **脚本没有端到端连续跑过**：`run_final_nolabel.sh` 把研究时 11-10 实际执行的命令（研究时由排队脚本串起来）按顺序写成一个脚本；`run_chain_nl.sh` 是研究时实际跑过的驱动（只把绝对路径换成相对仓库根）。中途断了重跑同一条命令即可接着跑。
- 研究时的无真值自查结果见 `results/nolabel_1110/selfcheck_nl.txt`：链上脚本里命中的真值字段只有伪框（`index_pseudo_<arm>` 的 `boxes9d`）、`w_sim.py` 的仿真真值、最终打分处读的封存 val/test 真值，以及去真值帧表只取 seq / frame / 内参。
