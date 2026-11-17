#!/bin/bash
# 完全单帧最终版（11-09 精简版）的训练命令，按研究时实际用过的顺序整理；没有从头到尾连续跑过一次。
# 网络：红外关键点老师 1；被老师塑形的朝向网络 6（红外蒸馏 2、LiDAR 2、红外+LiDAR 2）；高度网络 1；检测器 3；测距网络 2；朝向学生 2。
# 前提：已运行 release/fix_paths.py；上游产物按 11-08 单帧整链生成（见 release/README.md）：
#   检测器 wsfnb（3 种子）、位置伪框 st2sfnb、真实裁窗 real_s{j}_nbm*、背景库 xpaste/bank_bg_sf2.npy 与 ou/bank_kz_sf_v3.npy、
#   老师读数 teach/teach_train.npz（研究时 14 个网络的读数都在里面；最终规则只用其中 6 个）、高度网络 KPS 种子 0。
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
G=$R/output/camnorm/geo1004; W=$G/gproto; OU=$W/wf21/orient_uda; F=$W/wf21/final; H=$W/wf23/fusetrain
export PYTHONUTF8=1
export FE_KPSSFX=_k0                                  # 高度网络只用 1 个（KPS 种子 0）

# ---------- 高度网络（1 个）在真实裁窗上的读数 → 按颜色类的高宽比常数 ----------
cd $W/wf21/strict && KPS_SEEDS=0 KPS_SFX=_k0 python kps_shape_strict.py train val test

# ---------- 测距网络 ① 仿真打底 ② 适应真实（种子 5、6；真实数据用 real_s{种子%5}） ----------
cd $G
for s in 5 6; do
  V2_TAG=st V2_BANK=sf2 python secondstage_v2.py V2P $s
  V2_TAG=st V2_BANK=sf2 V2_REAL=nbm V2_DPS=nbmdp_sfnb python secondstage_v2.py V2Pp $s
  cp softkd/V2Pp_s${s}_st.pt softkd/XFPp_s$s.pt
done

# ---------- 朝向：伪朝向 = 6 个塑形网络对称集成，红外蒸馏组与 LiDAR 组相差 ≤10° 才用 ----------
cd $OU && python pseudo_sets.py                        # → pseudo/pseudo_train_t6il.npz（同时写出研究时比较过的另外几种规则）
# 朝向学生 ×2（初始化 = 红外+LiDAR 组成员 0、1）：仿真精确标签 + 通过帧上的四旋翼对称角点损失 + 镜像一致损失；不加视角原型
for k in 0 1; do python ou_train.py --seed $k --tag S6I --tgt gate --mir 12 --pseudo pseudo_train_t6il; done
cd $F && AB_SEEDS=0,1 python adabn_all.py S6I          # BN 统计重估 → adabn_all/S6I_s{0,1}.pth

# ---------- 位置：最终朝向学生读角点距离 → 两教师合并伪距离 → 测距网络 ③ ----------
ck="$F/adabn_all/S6I_s0.pth,$F/adabn_all/S6I_s1.pth"
cd $W/wf21/strict && python final_unc.py pp2 XFPp 5,6 train      # 训练目标里的颜色类：第二段测距网络的剪影
cd $R/tools && python $H/dump_train1.py                          # 检测器在全部真实训练帧上的检测 + 朝向裁窗窗口
cd $H
RM_REPL=$H/tags RM_TAGS=$H/tags python recut_mirror_t.py --src-tag wsfnb_train1_kz --sets "wsfnb_train1_ouS6I2M=$ck"   # 2 个学生 + 镜像 = 4 个读数
FE_UNC=pp2 ZT_NMEM=4 python ztrain.py wsfnb_train1_ouS6I2M $H/tags wsfnb_train1_kz           # 每帧角点 PnP 深度 z_L
FE_UNC=pp2 BT_SUF=_S6Ik python build_targets.py ztrain_wsfnb_train1_ouS6I2M.pkl 0.15          # → selftrain/real_s{j}_nbmdp_fz15_S6Ik.npy
cd $G
for s in 5 6; do
  V2_TAG=st V2_BANK=sf2 V2_REAL=nbm V2_DPS=nbmdp_fz15_S6Ik V2_OUTARM=ZS6Ik python secondstage_v2.py V2Rp $s
  cp softkd/ZS6Ik_s${s}_st.pt softkd/XFZS6Ik_s$s.pt
done
