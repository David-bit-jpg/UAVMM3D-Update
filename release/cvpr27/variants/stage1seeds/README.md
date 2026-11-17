# 第一阶段也换种子的整链重跑（补充材料用；11-16 搭建，未启动）

**目的。** 已有的种子散布（`review1113/seeds`、`sup3`、`sensorfree`）只换了下游的种子：测距修正网络、自训练检测器、朝向学生。第一阶段的纯仿真网络都是直接拷贝的。这里连第一阶段一起换种子。三个组 × 两个目标域，每个组合各新跑 2 轮，加上论文里已报告的那一轮，每组凑够 3 轮：

| 组 | 含义 | 现有已报告的那一轮（第 0 轮） |
|---|---|---|
| PST（`full`） | 完整方法：红外、LiDAR 教师 + 剪影合成 | `srcx/in2mav`、`srcx/out2antifs` |
| E1 无传感器对照 | 只用 RGB 朝向网络当教师；所有合成蒙版换成 3D 真值框投影的凸包；高度取常数比例 | `sensorfree/E1/{mav,antifs}_s0` |
| E4 纯 RGB 自训练 | 只用 RGB 朝向网络当教师；不做任何合成（学生 `--paste 0`）；高度取常数比例 | `sensorfree/E4/{mav,antifs}_s0` |

**配对方式：按轮配对。** 同一轮 k 里，PST、E1、E4 共用同一组重训的仿真检测器，因此 det0 和第一阶段的共享产物完全相同。E1 和 E4 还共用同一对重训的 RGB 朝向网络。三组用同一个下游种子偏移。

**分工。** 只交付脚本和空跑（dry run）结果。启动由主会话审核后执行。本目录下的任何脚本都还没有占用过 GPU。

## 1. 第一阶段包括哪些网络（两个源相同，都按原日志的参数逐项重建）

种子组 `sd`：第 1 轮新种子 `sd=30`，用到种子 30、31、32；第 2 轮 `sd=40`，用到种子 40、41、42。原链里编号为 k 的成员，在新链里的种子是 sd + k。所有路径和参数集中在 `s1paths.py` 一个文件里。

| 网络 | 每组个数 | 配置（室内 / 室外） | 训练入口 | 轮数 | 原版依据（训练日志里的参数块） |
|---|---|---|---|---|---|
| 仿真检测器（三个集成） | 3 | `sim_full_r34_geo_tr_mix_sym8_mblur_ramp` / `sim_out_r34_geo_ramp_out` | `tools/train.py` | 5 | 室内 `RAMP`、`RAMPs1`、`RAMPs2`；室外 `DET_s0..2`（`srcx/stage1_det.sh`） |
| 纯 RGB 朝向网络（E1、E4 的教师） | 2 | `sim_full_r34_kp_w12_zoom` / `sim_out_r34_kp_w12_zoom_out` | `train.py` | 30 | `KZs0/1`、`OKZrgb_s0/1` |
| 红外教师（只用来给红外蒸馏网络当老师） | 1 | `..._irT` / `..._irT_out` | `train.py` | 30 | `KZTIRs0`、`OKZirT_s0` |
| 红外蒸馏 RGB 网络（教师组「IR」） | 2 | `..._kdir` / `..._kdir_out`，`--set MODEL.DISTILL.TEACHER_CKPT <本组红外教师>` | `train.py` | 30 | `KZKDir_s0/1`、`OKZkdir_s0/1` |
| 红外 + LiDAR 网络（教师组 + 学生起点） | 2 | `..._irlid` / `..._irlid_out` | `lidar/kzlid/train_mm.py` | 30 | `KZMs0/1`、`OKZirlid_s0/1` |
| LiDAR 网络（教师组「LiD」） | 2 | `..._lid` / `..._lid_out` | `lidar/kzlid/train_kz_lid.py` | 30 | `KZLs0/1`、`OKZlid_s0/1` |
| 高度网络 | 1 | `..._irlid_kps` / `..._irlid_kps_out`，`--pretrained_model <本组红外+LiDAR 种子 sd 第 30 轮原始权重>` | `lidar/kpsize/train_mmk.py` | 8 | `KPSs0`（起点 `KZMs0` 第 30 轮）、`OKZkps_s0`（起点 `OKZirlid_s0`） |

### 公共参数（照抄日志）

- **室内全部网络**：`--batch_size 8 --workers 2 --fix_random_seed --cudnn_benchmark --use_amp --logger_iter_interval 500 --val_split test --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval`。保留的检查点个数沿用各网络原值：检测器 10，RGB / 红外+LiDAR / LiDAR 3，其余 2。
- **室外朝向网络**：参数相同（`gproto/outdoor/train_kp_out.sh` 里的 COMMON），保留检查点 2 个。
- **室外检测器**：照 `srcx/stage1_det.sh`。不用 amp，也不开 cudnn_benchmark；`val_match` 用默认的 top1，日志间隔用默认的 50。
- 训练期间只在**仿真**测试集上做监测。

### 发布格式（链里实际加载的文件）

- **室内检测器**：只保留 `model_state`，即 `release_20260920/ramp_s*.pth` 的格式。已核对 s0–s2 的发布文件和训练第 5 轮权重逐张量相同。
- **室内红外+LiDAR、LiDAR 网络**：用 `lidar/irlid/strip_aux.py` 去掉辅助头，即 `lidar/irlid/release/kzm_s*`、`kzl_s*` 的格式。已核对保留的张量逐个相同。
- **室外**：直接用第 30 轮原始检查点，和 `srcx/lib/sx.py` 一致。

新发布文件放在 `release/in_sd<sd>/`。

### 已核对的事项

- **配置未变**。`check_cfg_drift.py` 用当前 yaml 解析出的完整配置，逐键比对 14 个原版网络训练日志里的配置转储，结果全部相同（`logs_cfg_drift.txt`）。室外检测器 `DET_s0` 第一次启动的日志是 `MIN_RGB_VIS 5`，但那次中途放弃了；正式那次和 s1、s2 都是 0，与当前配置一致。
- **代码只能部分核对**。比如 `mmcache_det_dataset.py` 11-07 新加的贴图分支，按配置开关、默认关闭。但室内网络训练之后，`uavdet3d` 下仍有多个文件改过，无法逐位证明与当年的代码等价。室外网络训练于 11-11、11-12，代码与现在相同。
- **测距修正网络的第一阶段**本来就在下游链里随种子偏移重训，第 1 段即仿真段，种子为 5+sd、6+sd。

## 2. 命名（全部是新名字，空跑已确认都不存在）

- **第一阶段新训练目录**：`D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm/<配置名>/S1X_<网络>_s<种子>/`。
  - 网络标签：室内 DET、KZ、KZTIR、KZKDir、KZM、KZL、KPS；室外 DET、OKZrgb、OKZirT、OKZkdir、OKZirlid、OKZlid、OKZkps。
  - 放在 D: 的原因：E: 只剩 59 GB。按原版检查点保留数，第一阶段约 60 GB，自训练检测器约 12 GB，放不下。
  - 实现：`s1_train.py` 原样调用原训练入口，只给 `EXP_GROUP_PATH` 加上 D: 前缀。这和 `lidraw/lidraw_train.py` 用的是同一种做法；断点续训逻辑不变。
- **共享第一阶段产物**：`shared/<目标>_sd<sd>/`。
- **下游各轮**：`runs/<目标>_<组>_s1sd<sd>/`；裁窗放 `D:/S2MUAV3D/srcx_work/s1seeds/<同名>/`，每轮 MAV6D 约 11 GB，AntiUAV3D 约 3.4 GB。
- **配置文件**：`tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_<C>_s1<组>sd<sd>.yaml`。
- **伪标签索引**：`<目标缓存>/train/index_pseudo_srcx_<C>_s1<组>sd<sd>.pkl`。
- **自训练检测器**：`D:/S2MUAV3D/stage1seeds_models/models/uavdet_3d/camnorm/srcx/st_<C>_s1<组>sd<sd>/DET1_s{0,1,2}/`。

其中 C = `in2mav`、`in2mav_none`、`out2antifs`、`out2antifs_none`；共享产物的配置后缀是 `_s1prepsd<sd>`。

## 3. 依赖关系

```
仿真检测器 ×3 ─┬─> 共享产物 prep_shared.sh（det0 集成推理、srconly 预测、测距修正的仿真样本 + 红外剪影 + 同一遍生成的真值框凸包、真实背景库、真实 d0 裁窗）
               │        └─> 三个组的下游链（自训练检测器 k 从仿真检测器 k 出发）
纯 RGB ×2 ─────┴─> E1、E4：教师 + 学生起点
红外教师 ──> 红外蒸馏 ×2 ─┐
红外+LiDAR ×2 ──> 高度网络（从同种子 sd 的红外+LiDAR 出发）
红外+LiDAR ×2、LiDAR ×2 ─┴─> PST：6 个教师 → 红外 vs LiDAR 一致门 → 学生（从同号红外+LiDAR 出发）；高度网络读数
```

**下游链**与 `seeds/run_seed_chain.sh` 完全相同，唯一区别是不拷贝第 0 轮的第一阶段产物，改用 `shared/`。顺序：测距修正第 1–2 段（种子 5+sd、6+sd）→ 伪标签 → 自训练检测器 ×3（种子 k+sd）→ 裁窗 → 教师 → 伪朝向 → 学生（种子 0+sd、1+sd）→ 融合权重 w（仿真验证集）→ 测距修正第 3 段 → 部署 → **测试集打分一次**。

库文件补丁与已有脚本逐字节一致，只差路径（已 diff 核对）：

- PST 的测距修正和学生 = `seeds/*/lib`；
- E1 = `sensorfree/E1/*/lib`；
- E4 的测距修正 = none 臂，学生加 `--paste 0`。

E1 的真值框凸包有两处：测距修正用的是共享产物里和仿真样本**同一遍**生成的凸包，与样本天然对齐；`prep_shared.sh` 会写 `hull_check.json`，并断言红外剪影至少 80% 落在凸包内。学生用的凸包是 `sensorfree/hull_<源>.npy`，它对应的源裁窗缓存和第一阶段无关。

**E4 与已有 E4 第 0 轮做法不同（等价）**：已有 E4 把学生之前的部分从同种子的 RGB-only 轮拷贝过来。新种子没有 RGB-only 轮，所以这里整条 none 链自己跑一遍。代码相同，学生之前的部分本来就不依赖学生，结果等价。

## 4. GPU 时估计

表中 GPU 时按两个训练同时跑时测得的每轮分钟数折算，单独跑约快 5–15%；每轮迭代数取自原日志。

| 网络 | 室内 每轮迭代 / 分钟每轮 | 室内 GPU 时（每组） | 室外 每轮迭代 / 分钟每轮 | 室外 GPU 时（每组） |
|---|---|---|---|---|
| 检测器 ×3 | 2263 / 5.6–6.1 | 3 × 0.5 = 1.5 | 671 / 1.5（workers 2） | 3 × 0.13 = 0.4 |
| 纯 RGB ×2 | 2263 / 5.0–5.3 | 2 × 2.6 = 5.2 | 671 / 1.6 | 2 × 0.8 = 1.6 |
| 红外教师 ×1 | 2263 / 5.9 | 3.0 | 671 / 1.5 | 0.75 |
| 红外蒸馏 ×2 | 2263 / 6.0 | 2 × 3.0 = 6.0 | 671 / 1.7–1.8 | 2 × 0.9 = 1.8 |
| 红外+LiDAR ×2 | 2263 / 7.1（单独）– 9.8（并跑） | 2 × 4.0 = 8.0 | 671 / 2.0–2.1 | 2 × 1.05 = 2.1 |
| LiDAR ×2 | 2263 / 7.6（最快）– 16（拥挤） | 2 × 3.8 = 7.6（拥挤时到 17） | 671 / 2.1–2.3 | 2 × 1.1 = 2.2 |
| 高度网络 ×1 | 2263 / 7.0 | 0.9 | 671 / 2.1 | 0.3 |
| **合计（每个种子组）** | | **≈ 32 GPU 时** | | **≈ 9.2 GPU 时** |

- **第一阶段合计**：2 × 32 + 2 × 9.2 ≈ **82 GPU 时**。
- **下游每轮墙钟**：按已有种子轮的 `run.log` 起止时间估计，第一阶段拷贝的部分不在内。
  - MAV6D：PST 2.8–3.1 h，E1 4.3–6.3 h（当时与其他任务并跑），none 链即 E4 约 2.5 h。
  - AntiUAV3D：PST 1.8 h，E1 2.1–3.8 h，none 链 2.6–2.7 h。
  - 共享产物：MAV6D 约 0.6 h，AntiUAV3D 约 0.3 h。
- **下游合计**：约 2 × (3.2 + 4.5 + 2.6 + 2.2 + 2.8 + 2.7) + 1.8 ≈ **38 h**。

## 5. 顺序与墙钟（三条通道，最多 2 个带数据加载进程的训练）

- **通道 A**（第一阶段，经 `gpu_slot.sh heavy`）：每个种子组依次训 检测器 ×3 → 纯 RGB ×2 → 红外教师 → 红外蒸馏 ×2。种子组顺序：室内 sd30 → 室内 sd40 → 室外 sd30 → 室外 sd40。之后跑 E4 MAV6D sd30 → E4 AntiUAV3D sd30。
- **通道 B**（第一阶段）：每个种子组依次训 红外+LiDAR ×2 → 高度网络 → LiDAR ×2，种子组顺序同上。之后跑 E4 MAV6D sd40 → E4 AntiUAV3D sd40。
- **通道 C**（下游，经 `gpu_slot.sh light`，DataLoader workers 0）：
  1. 共享 MAV6D sd30 → E1 MAV6D sd30 → PST MAV6D sd30
  2. 共享 MAV6D sd40 → E1 sd40 → PST sd40
  3. 共享 AntiUAV3D sd30 → PST → E1
  4. 共享 AntiUAV3D sd40 → PST → E1

每一步只等自己用到的网络。比如 E1 MAV6D sd30 只要通道 A 训完室内检测器和 RGB 网络就能开始，名义上在约 7 h 处。

**名义时间线（不计拥挤）：**

| 事件 | 名义时间 |
|---|---|
| 室内 sd30 第一阶段完成（A / B） | 15.8 h / 17.9 h |
| 室内 sd40 第一阶段完成（A / B） | 31.6 h / 35.9 h |
| 室外两组第一阶段完成 | 约 45.7 h |
| 通道 C：E1 MAV6D sd30 | 6.8 → 11.3 h |
| 通道 C：PST MAV6D sd30 | 17.9 → 21.1 h |
| 通道 C：E1 MAV6D sd40 | 22.6 → 27.1 h |
| 通道 C：PST MAV6D sd40 | 35.9 → 39.1 h（**优先级 1 完成**） |
| AntiUAV3D 四轮 | 40.8 → 51.1 h（**优先级 2 完成**） |
| E4 四轮 | 约 46 → 51 h（**优先级 3 完成**） |

- **全部完成**：名义约 **51 h**。3 个 GPU 任务并跑，加上 E: 是机械盘，按 1.3–1.6 倍算是 **66–82 h**；若别人也在用显卡、再翻一倍，约 100 h。
- **能否赶上截止**：11-16 中午启动的话，预计 11-19 早上到晚上全部完成，最坏约 11-20 晚上。截止时间是机器时间 11-24 19:59，**余量 4 天左右**。优先级 1（MAV6D PST + E1）名义 39 h，按 1.5 倍约 59 h，大约在 11-18 深夜。
- **启动保护**：第一阶段训练在 11-23 12:00 之后不再开新的；下游轮在 11-24 10:00 之后不再开新的。可用环境变量 `LATEST_START=MMDDHHMM` 覆盖。

### 磁盘

| 盘 | 用量 | 现状 |
|---|---|---|
| D:（第一阶段 + 自训练检测器 + 裁窗） | ≈ 60 + 12 + 6 × 11 + 6 × 3.4 ≈ 160 GB | 剩 1.5 TB |
| E:（各轮目录 + 共享产物 + 发布文件） | ≈ 12 × 1–1.7 + 4 × 1 + 1.1 ≈ 22 GB | 剩 59 GB |

E: 由多个会话共用，请留意剩余空间。

## 6. 启动

把 `queue_stage1seeds.txt` 的三行按原顺序追加到 `gproto/local_queue_1113.txt`：先 C，再 A、B。C 刚开始只等文件、不开训练，这样调度器（训练进程少于 2 个时才开新任务）还能接着开 A 和 B。

也可以直接手动起：

```
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/lane.sh C
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/lane.sh A
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/lane.sh B
```

单项也能单独跑：

- `bash train_stage1.sh in 30 kdir:0`
- `bash prep_shared.sh mav 30`
- `bash run_group.sh E1 mav 30`

所有脚本都支持 `DRY=1`。

**汇总。** `python summarize.py` 只读已有的打分文件，输出每轮指标、每轮相对 PST 的差值（以及 E4 − E1）和跨轮 均值 ± 标准差，写到 `summary/SUMMARY.md` 和 `summary.json`。`python summarize.py --boot` 再调用现有的 `heads/heads_boot.py`，在 3 对上做合并配对自助法：MAV6D 按序列重采样，AntiUAV3D 按块重采样，块长 L = 176。不满 3 对时不算，加 `--partial` 后 2 对也算。

## 7. 健壮性（沿用 `heads/run_arm.sh`、`lidraw`、`seeds` 的做法）

- **GPU 调度**：每个 GPU 步骤都经 `gpu_slot.sh`。第一阶段走 heavy 槽，DataLoader workers 2，这就是原配方；设 `S1_WORKERS=0` 可改走 light 槽，会记为偏差。下游全部 workers 0，推理用 `infer_wrap` 把显存占比限在 0.2。
- **显存不足**：批量减半后续训，训练类最多 3–4 次。批量减半记为「RECIPE DEVIATION」，写在 `logs/stage1_deviations.log` 或各轮的 `logs/retry.log`。
- **非显存不足的失败**：第一阶段续训重试一次，再失败就写 `.failed`，依赖它的轮会带明确原因退出。
- **断点续跑**：`train.py` 自动从最新的 epoch 检查点续训；测距修正每轮或每 250 步存一次状态，学生每 250 步存一次。产物已存在的步骤直接跳过。
- **标记文件**：`.done`、`.failed`、`.skipped`。
- **日志**：总日志 `logs/stage1seeds.log`；另有各通道的 `logs/lane_*.log` 和各轮的 `runs/*/logs/run.log`。
- **从不杀进程。**

**真实标签。** 只有测试集标签会被读，而且只通过现有打分器：每轮跑完由 `score_x.py` 打一次分；PST 轮另打一次 srconly，即重训的纯仿真检测器；汇总时由 `heads_boot.py` 读。不做验证集打分。任何选择都不依赖真实标签。

## 8. 无法逐位还原、或需要主会话知道的地方

1. **红外教师的种子**：原版两个红外蒸馏网络共用**一个**红外教师（种子 0）。这里每个种子组各重训一个红外教师（种子 sd），两个红外蒸馏网络共用它。这是对「整组第一阶段换种子」的理解，结构与原版相同。
2. **室外检测器的 workers**：原版 s1、s2 是用 workers 0 训的（`launch_det_out12_w0.sh`，当时为了躲 heavy 槽排队），s0 用的是 2。新的一律用 2，即 `stage1_det.sh` 的配方。这只影响数据增广的随机流。
3. **代码等价性**：见第 1 节，配置逐键相同，但室内网络训练之后 `uavdet3d` 代码有改动，无法逐位证明。看到的改动都是按配置开关、默认关闭的分支。
4. **训练输出位置**：放到 D:，原因是 E: 空间不够。只改了路径，权重和续训逻辑不受影响。
5. **验证集**：不打分。早先的 `seeds` 轮打过验证集作健全检查；这里遵守 11-14 只报测试集的规定，预测文件本身不受影响。
6. **复现性**：即使种子相同，GPU 上的非确定性也会让结果不能逐位复现，这一点和已有各轮相同。

## 9. 文件

| 文件 | 作用 |
|---|---|
| `s1paths.py` | 第一阶段配方与路径的唯一来源 |
| `train_stage1.sh` | 训练第一阶段并转成发布格式 |
| `s1_train.py` | 训练入口包装，只改输出目录 |
| `make_s1_lib.py` | 生成各轮的隔离库副本 |
| `prep_shared.sh` | 生成共享第一阶段产物 |
| `run_group.sh` | 一轮下游链 |
| `common.sh` | 公共函数 |
| `lane.sh` | 三条通道 |
| `queue_stage1seeds.txt` | 队列行 |
| `summarize.py` | 按轮汇总 + 合并自助法 |
| `check_cfg_drift.py` | 配置漂移核对 |
| `parse_train_logs.py` | 从训练日志读参数和速度 |
| `dry_run.txt` | 全部空跑输出 |
| `logs_cfg_drift.txt` | 配置核对结果 |
