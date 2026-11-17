# S2MUAV3D 对比实验包（11-11 版，给老师，最后一包）

**本包替代 11-10 发的那个包（那个可以删掉）。** 新增：源域对照（in / out / mix 三种仿真源域，各测两个真实目标域）、AntiUAV3D 上的 PCMD 教师伪标签。

**解压后三步，跑完把一个几 MB 的结果 zip 发回即可。**

```bash
bash setup/install.sh          # 1. 装环境（一次）：conda 环境 uav3d + 依赖 + 把代码路径换成本包位置
conda activate uav3d
bash scripts/run_all.sh        # 2. 一键全跑：8 张卡（缺省 GPUS="0 1 2 3 4 5 6 7"）按队列自动分活，后台运行，关终端也不停
bash scripts/collect_results.sh  # 3. 随时运行：先打分（runs/SCORES.md），再把预测和分数打成 results_to_send_<时间>.zip，发回给我们
```

- **看进度**：`cat runs/STATUS.txt`；单个任务的日志在 `runs/logs/`。
- **断了**：重新运行 `bash scripts/run_all.sh` 即可接着跑，已完成的自动跳过。某个任务失败：修好后删掉 `runs/queue/<任务>.lock` 和 `.fail` 再运行。
- **显存**：4090 放不下官方批大小时自动把批大小减半重试，学习率按比例缩小。
- **打分**：`python scripts/score.py`，表在 `runs/SCORES.md`：每个目标域一张表，带"源域"列，多个种子取均值 ± 标准差。测试集真值只在 `data/gt/`，**只给打分用，任何训练都不读**。
- 如果老师已经用 11-10 的包跑出了 `runs/<方法>_in_sim_s<种子>` 或 `runs/<方法>_out_sim_s<种子>`：数据和设置相同，把这些目录拷进本包的 `runs/` 即可复用（缺的那个测试集会自动补预测）。

## 实验设计：3 个源域 × 2 个目标域

| 源域（仿真，有标签） | 帧数 |
|---|---|
| `in`：UE5 室内仿真 `sim_train` | 19,490 |
| `out`：CARLA 室外仿真 `sim_outdoor_train`（晴天白天） | 5,367 |
| `mix`：in + out + CARLA Town10 按画质筛过的帧 `sim_outdoor_town10_kept`（只按仿真标签筛：真值 3D 框投影区域内灰度拉普拉斯方差 ≥ 600 且投影框宽 ≥ 16 像素，1,818 帧留 1,564） | 26,421 |

| 目标域（真实） | 无标签训练图（只给自训练） | 测试集（真值只在 `data/gt/`） |
|---|---|---|
| MAV6D（`mav`） | `real_train` 12,866 帧 | `real_test` 4,800 帧 |
| AntiUAV3D 80 m 内（`anti`） | `outdoor_train` 3,361 帧 | `outdoor_test` 2,825 帧 |

- **只用仿真（sim）**：每个源域训一个模型，**同一个模型同时测两个目标域**。
- **+ 无标签自训练（st）/ + PCMD 教师伪标签（kd）**：按目标域做，从该目标域主表用的源域出发：MAV6D ← `in`，AntiUAV3D ← `out`。
- `mix` 不复制图像：三个标签文件直接一起传给各方法的数据读取（每个文件的图像相对它自己的目录找），五个方法都支持。
- 各方法轮数换源域时不变（按 `in` 的帧数和官方样本数对齐；GUPNet 每轮固定 3,712 个样本），所以 `out` 训练快、`mix` 慢约 35%。

PCMD 教师伪标签（不含任何真实标签）：
- `pcmd_teacher_pseudo_real_train.pkl`：MAV6D，4,088 帧。
- `pcmd_teacher_pseudo_outdoor_train.pkl`：AntiUAV3D，298 帧。来自我们室外 PCMD 的**每组 1 个网络**那次运行：位置 = 检测 × 红外剪影蒸馏过的距离修正；朝向 = 红外 / LiDAR 塑形教师在两位老师一致（≤10°）时给的朝向（四个等价朝向里取离检测器最近的那个代表，评测按四旋翼对称算，不影响）。

`pretrained/`：我们已训好的 GUPNet、MonoDGP 的 `in` 源域种子 0 模型（不在队列里），MAV6D 上的 st 直接用它们打伪标签。MonoDGP 这个是 24 轮版本（队列里重训的都是 48 轮）。

## 目录

```
setup/      装环境、换路径          scripts/   run_all.sh（一键）、run.sh（单个任务）、tasks.txt（任务队列）、score.py、collect_results.sh
data/       数据（见上）            torch_home/ 各方法官方用的预训练权重（不联网下载）     pretrained/ 见上
Open3DUAVDet/output/camnorm/geo1004/cmpfull/<方法>/   五个对手方法代码（官方代码只改数据读取和角度编码，改动处标 [UAV port]，说明见各自 NOTES.md）
PCMD_code/  我们方法的最终代码（参考，本包任务不需要运行它）
runs/       运行后生成：模型、预测、日志、STATUS.txt
```

单独跑一个任务：`CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh monodetr 0 mix sim both`（方法 / 种子 / 源域 in|out|mix / 步骤 sim|st|kd / 目标域 mav|anti|both）。运行目录名 `runs/<方法>_<源域>_<步骤>[_<目标域>]_s<种子>`。

**任务清单（按急需程度排、带预计完成时间）见 `任务清单.md`**；更多说明见 `EXPERIMENTS.md`。
