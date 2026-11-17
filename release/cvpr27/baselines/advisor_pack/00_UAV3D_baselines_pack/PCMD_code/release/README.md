# 无人机单目 3D 检测：仿真 → 真实（定稿方法）

**做什么**：只输入一张彩色图，输出画面里无人机的 3D 位置、尺寸和朝向。
**怎么训练**：标签只来自仿真；真实数据（MAV6D）只用训练集的**无标签**图片打伪标签；真实标签**只在最后打分**时使用。
红外和 LiDAR 只在仿真训练时当"老师"，测试时只用彩色图。

这个分支只放定稿方法的代码（约 290 个文本文件），不含任何权重、数据和中间结果。

---

## 1. 方法链（按执行顺序）

| # | 步骤 | 做了什么 | 种子数 |
|---|---|---|---|
| 1 | 仿真检测器 | ResNet-34 中心点检测网络（CenterNet 式）。距离不直接回归，而是 **距离 = 焦距 × 无人机横向跨度 / 预测的 2D 宽度** | 5 |
| 2 | 第二段裁窗测距网络 | 在检测框周围裁一个小窗口，单独再估一次距离；训练时用仿真**红外剪影**当老师（蒸馏），再用真实无标签裁窗做域适配 | 生成伪标签用 5 个，推理用另外 5 个 |
| 3 | 真实训练图伪标签 | 检测器在真实训练图上打框，伪标签的距离用第 2 步的测距网络修正，再按置信度筛选 | — |
| 4 | **实例原型修正伪标签距离（本方法新增）** | 同一飞行片段里相机不动：取时间中位数当背景，帧减背景得到无人机剪影；把第 t 帧的剪影当"原型"，和 t+1/2/3/5/10 帧做连续尺度匹配，得到帧间尺度比（= 距离比）。每个片段做一次最小二乘：伪标签距离（σ 5%）+ 帧间尺度比（σ 1.2%，匹配质量 ≥ 0.8），沿视线缩放伪标签位置 | — |
| 5 | 检测器自训练（**弱增强**） | 用修正后的伪标签，从仿真检测器出发微调 3 轮；数据增强只留水平翻转（关掉光度和噪声增强） | **3（种子 0、1、2）** |
| 6 | 推理 | 3 个种子集成 + 翻转测试增强 + 时序热力图背景 + 尺寸自标定；朝向 = 裁窗关键点网络 5 成员 + PnP + 投票 + 因果隐马尔可夫平滑；高度 = LiDAR 关键点给出的高宽比 × 长宽均值；距离再用第 2 步部署版测距网络修正；最后**推理时实例原型**：第 t 帧用 [t−2000, t+5] 帧窗口做第 4 步同样的最小二乘（**等后 5 帧**，其余全是过去帧），沿视线缩放最终位置 | — |

实例原型用在两处：**训练**（第 4 步，修正伪标签距离）和**推理**（第 6 步最后，延迟 5 帧修正输出距离）。两处都不看真值，裁窗位置都来自检测器自己的框。

---

## 2. 结果（真实测试集 4800 帧，自训练检测器 3 个种子集成）

| 模型 | 平均精度 | 平均位置误差 m | 平均尺寸误差 cm | 平均朝向误差 ° | 检出率 % | UDS* |
|---|---|---|---|---|---|---|
| 实例原型之前的定稿模型（正常增强，伪标签不修，推理不修） | 57.01 | 0.223 | 1.79 | 14.03 | 99.8 | 70.40 |
| **定稿：弱增强 + 实例原型（训练 + 推理等后 5 帧）** | **60.53** | **0.199** | **1.77** | **13.84** | **99.9** | **72.53** |

- 同一套 3 个种子、同一套上游网络、同一套评测代码；源记录在 `results/online_final3.txt` 和 `results/final_3seed_table.md`（还含验证集和中间组合：只训练修 / 只推理修 / 正常增强 + 原型）。
- **平均精度**：按位置误差阈值 0.1 / 0.2 / 0.4 / 0.8 m 的 3D AP 取平均（MAV6D 里无人机只有 1.6–5.6 m 远，按比例缩小了常用的 1/2/4/8 m 阈值）。
- **平均位置/尺寸/朝向误差**：只在检中的帧上算（置信度最高的框与真值 2D IoU ≥ 0.1）；朝向按四旋翼对称算（转 90°/180°/270° 不算错）。
- **检出率**：真值被检中的比例。
- \***UDS**：MAV6D 原文没有给公式，这里按 nuScenes NDS 的写法自己定义：0.5 × 平均精度 + 0.5 × 平均(1 − min(1, 位置误差/1 m), 1 − min(1, 尺寸误差/10 cm), 1 − min(1, 朝向误差/1 rad), 检出率)。代码见 `output/camnorm/geo1004/cmpeval/paper_table.py` 的 `metrics()`。

---

## 3. 怎么跑

### 3.0 环境与路径（一次）

```bash
pip install -r release/requirements.txt        # Python 3.10；torch / torchvision 按显卡自行安装（研究时 CUDA 版）
python release/fix_paths.py --mmcache <缓存根目录> --mav6d <MAV6D 原图根目录> \
       --torch-home <ImageNet 预训练权重目录> --data-collect <仿真录制数据根目录>
```

代码里的绝对路径都换成了占位符（仓库根、缓存根、MAV6D、预训练权重、仿真录制数据），`fix_paths.py` 把它们换成本机路径；不带参数时用原机器的路径（`E:/mmcache` 等），仓库根自动取。提交代码前用 `python release/fix_paths.py --reset` 换回占位符。
研究时加的"防改动哈希核对"默认关闭（路径一换哈希必然变），要核对就设 `UAV_STRICT_HASH=1`。

### 3.1 数据（不在仓库里，要自己准备）

| 数据 | 生成命令 |
|---|---|
| MAV6D 相机归一化缓存（官方物体框；train / 按序列留出的 val / 官方 test） | `python tools/build_mav6d_cache.py` → `<缓存根>/mav6d_cn_obj` |
| 仿真缓存（≤ 8 m 整帧，按序列 8:2 划分；彩色 JPEG + 多焦距裁窗） | `cd tools && python run_full_v2.py`（第 1 步会用 `build_near_subset.py` 重新生成帧清单 `cfgs/subsets/full_v2/`，该清单 6 MB 没放进仓库） |
| 仿真红外 / LiDAR 边车（老师用） | `python tools/build_mm_sidecar.py --cache <缓存根>/full_v2 --root <仿真录制数据根> --split train test --workers 8` |
| ImageNet 预训练权重 | 放到 `TORCH_HOME`，脚本不会联网下载 |

### 3.2 上游网络（第 1–3 步 + 部署用的测距 / 朝向 / 高度网络）

```bash
bash release/run_upstream.sh          # 已完成的步骤自动跳过；bash release/run_upstream.sh 5 = 从第 5 步开始
```

内部 9 小步，种子数保持研究时的设置：仿真检测器（5）→ 第一轮自训练（5）→ 测距网络训练数据 → 测距网络种子 0–4 → 第二轮自训练（5，同时生成实例原型要修的伪标签和裁窗用的检测框）→ 部署用测距网络种子 5–9 → 朝向网络（不蒸馏版，5）→ 朝向网络定稿版（红外 + LiDAR 辅助，5 成员）→ LiDAR 关键点定高度（2）。

### 3.3 定稿最后几步（第 4–6 步，自训练检测器 3 个种子）

```bash
bash release/run_final_3seed.sh                     # 实例原型修伪标签 → 弱增强自训练种子 0/1/2 → 推理 → 推理时实例原型 → 出表
WITH_BASELINE=1 bash release/run_final_3seed.sh     # 另外出"实例原型之前"的对照行
```

结果写到 `runs/final/table_final3.md`（格式同上表）。实例原型的剪影匹配和最小二乘只用 CPU（训练集帧对 4 个进程）。

对应的单条命令（脚本里就是这些）：

```bash
# 第 4 步：训练伪标签的实例原型修正（在 output/camnorm/geo1004/proto/inst/ 下）
for w in 0 1 2 3; do python inst2_train.py $w 4 & done; wait      # 剪影帧对 → pairs2_train_w*.pkl
python make_inst_index.py                                          # → <缓存根>/mav6d_cn_obj/train/index_pseudo_st2inst.pkl
# 第 5 步：弱增强自训练（在 tools/ 下），k = 0 1 2
python train.py --cfg_file cfgs/models/uavdet_3d/camnorm/st2instwa_mav6d_r34_geo_full_tr.yaml --batch_size 8 --workers 2 \
  --fix_random_seed --cudnn_benchmark --use_amp --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --skip_test_eval \
  --seed $k --extra_tag WAINST_s$k --pretrained_model ../output/release_20260920/ramp_s$k.pth
# 第 6 步：推理（集成/翻转/时序背景/自标定 → 朝向 5 成员 → LiDAR 高度 → 测距修正），见 run_final_3seed.sh 的 infer()
# 推理时实例原型（在 output/camnorm/geo1004/proto/inst/ 下）
python inst2.py val test
ONLINE_W=2000 ONLINE_LAG=5 CMP_SPEC='wainst:弱增强+实例原型(3种子)' CMP_OUT=final3 python online_fuse.py
```

---

## 4. 代码在哪

| 内容 | 文件 |
|---|---|
| 检测网络、数据集、损失（库） | `uavdet3d/`（`model/detectors/center_det.py`、`model/dense_head_2d/center_head.py`、`datasets/mmcache/`） |
| 训练 / 评测入口 | `tools/train.py`、`tools/eval_camnorm.py`、`output/camnorm/geo1004/eval_dump.py`（集成 + 翻转 + 时序背景 + 自标定） |
| 配置 | `tools/cfgs/models/uavdet_3d/camnorm/`：仿真检测器 `sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml`；定稿自训练 `st2instwa_mav6d_r34_geo_full_tr.yaml`（基于 `st2pluswa_…` → `st2plus_…` → `st_…`） |
| 伪标签 | `tools/annot_audit/dump_real_preds.py`、`tools/make_pseudo_index.py`、`output/camnorm/geo1004/make_st2plus_dump.py` |
| 第二段裁窗测距（红外剪影蒸馏 + 域适配） | `output/camnorm/geo1004/secondstage_v2.py` |
| **实例原型** | `output/camnorm/geo1004/proto/inst/`：`inst2_train.py`（训练集剪影帧对）、`make_inst_index.py`（修伪标签）、`inst_solve.py`（最小二乘）、`inst2.py`（验证/测试帧对）、`online_fuse.py`（推理时修正 + 打分） |
| 研究时的驱动（定稿组 = `wainst`） | `output/camnorm/geo1004/proto/run_proto.py`（`train()` / `infer()`）、`run_inst.py`、`compare_gen.py` |
| 朝向 | `output/camnorm/geo1004/proto/dump_orient_p.py`、`output/camnorm/night1003/scratch/repl/dump_members_repl.py`、`output/camnorm/geo1004/kzst7/`、`finalads/temporal_filter.py` |
| 高度（LiDAR 关键点） | `output/camnorm/geo1004/lidar/kpsize/`（`lidar_templates.py`、`train_mmk.py`、`kps_shape_chain.py`） |
| 组装 + 指标 | `output/camnorm/geo1004/ablation/ablation.py`（`build()` / `unc`）、`output/camnorm/geo1004/cmpeval/paper_table.py` |

`output/` 在 `.gitignore` 里（研究时产物都写在那里），这里只把其中的代码和少量控制文件（预注册 json、验证集选出的超参数 json）强制加进了版本库，相对路径与原工作目录一致，所以脚本之间的导入不用改。

---

## 5. 不包含 / 注意

- **不含**：任何权重（`*.pth` / `*.pt`）、数据（MAV6D、仿真录制、缓存）、中间结果（`*.pkl` / `*.npz` / `*.npy`）、图片、日志。只有代码、配置和两个小的结果表文本（`results/`）。
- **两个脚本都没有端到端连续跑过**：`run_upstream.sh` 的每条命令与研究时的训练日志一致（改编自交接包的整链脚本），`run_final_3seed.sh` 是把研究驱动 `run_proto.py` / `run_inst.py` / `online_fuse.py` 的逻辑展开成 shell。中途断了重跑同一条命令即可接着跑。
- GPU 训练不逐位可复现，重训结果会和上表有小幅差别（研究时 5 种子复核：实例原型 + 弱增强的增益方向一致）。
- 研究时在 Windows + Git Bash 上跑；Linux 也可以（`env.sh` 自动处理路径分隔符）。
