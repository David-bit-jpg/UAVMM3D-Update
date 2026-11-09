#!/bin/bash
# 完全单帧最终版（11-09）的训练命令，按研究时用过的顺序整理；没有从头到尾连续跑过一次。
# 前提：已运行 release/fix_paths.py；上游产物（检测器 wsfnb、伪标签 st2sfnb、真实裁窗 real_s{j}_nbm*、
#       朝向教师读数 / 伪朝向 pseudo_train.npz、11-08 版朝向网络 adabn_all/PSP_s*.pth）已按 11-08 单帧整链生成。
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
G=$R/output/camnorm/geo1004; W=$G/gproto
export PYTHONUTF8=1

# ---------- 测距学生 ① ② （种子 5..9；真实数据用 real_s{种子%5}） ----------
cd $G
for s in 5 6 7 8 9; do
  V2_TAG=st V2_BANK=sf2 python secondstage_v2.py V2P $s
  V2_TAG=st V2_BANK=sf2 V2_REAL=nbm V2_DPS=nbmdp_sfnb python secondstage_v2.py V2Pp $s
done

# ---------- 两教师合并伪目标 ----------
F=$W/wf21/final
ck=""; for k in 0 1 2 3 4; do ck="$ck,$F/adabn_all/PSP_s$k.pth"; done; ck=${ck#,}
cd $R/tools && python $W/wf23/fusetrain/dump_train1.py                       # 检测器在全部真实训练帧上的检测 + 朝向裁窗窗口
RM_REPL=$W/wf23/fusetrain/tags RM_TAGS=$W/wf23/fusetrain/tags \
  python $W/wf23/fusetrain/recut_mirror_t.py --src-tag wsfnb_train1_kz --sets "wsfnb_train1_ouPSPallM=$ck"   # 朝向 5 个 + 镜像 = 10 成员
cd $W/wf23/fusetrain
python ztrain.py wsfnb_train1_ouPSPallM $W/wf23/fusetrain/tags wsfnb_train1_kz   # 每帧 LiDAR 角点 PnP 深度
python build_targets.py ztrain_wsfnb_train1_ouPSPallM.pkl 0.15                    # → selftrain/real_s{j}_nbmdp_fz15.npy

# ---------- 测距学生 ③ 伪标签精修（两教师合并伪目标） ----------
cd $G
for s in 5 6 7 8 9; do
  V2_TAG=st V2_BANK=sf2 V2_REAL=nbm V2_DPS=nbmdp_fz15 V2_OUTARM=FZ python secondstage_v2.py V2Rp $s
  cp softkd/FZ_s${s}_st.pt softkd/XFFZ_s$s.pt
done

# ---------- 朝向学生 ③ 伪标签精修 + 镜像一致损失（种子 0..4），再重估 BN ----------
for k in 0 1 2 3 4; do
  python $W/wf21/orient_uda/ou_train.py --seed $k --tag PSPM --tgt gate --proto 0.1 --mir 12 --pseudo pseudo_train
done
cd $F && python adabn_all.py PSPM                                              # → adabn_all/PSPM_s{0..4}.pth
