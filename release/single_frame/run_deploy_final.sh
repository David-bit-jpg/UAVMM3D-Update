#!/bin/bash
# 完全单帧最终版（11-09 精简版）的部署 + 打分（只推理；真值只用于打分）。没有从头到尾连续跑过一次。
# 前提：run_train_final.sh 的产物（softkd/XFZS6Ik_s{5,6}.pt、wf21/final/adabn_all/S6I_s{0,1}.pth、kps/shape_*_wsfnb_k0.pkl）；
#       检测器部署记录 night1003/repl/wsfnb_{val,test}_kzt7q 与 wsfnb_ens/ens_{val,test}.pkl（11-08 单帧整链）。
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
G=$R/output/camnorm/geo1004; W=$G/gproto; F=$W/wf21/final
export PYTHONUTF8=1 FE_KPSSFX=_k0

cd $W/wf21/strict && python final_unc.py z2S6Ik XFZS6Ik 5,6 val,test,train       # 测距网络 ×2 逐检测输出 + 颜色类亮度
cd $W/wf23/mirror
ck="$F/adabn_all/S6I_s0.pth,$F/adabn_all/S6I_s1.pth"
for sp in val test; do
  python recut_mirror.py --src-tag wsfnb_${sp}_kzt7q --sets "wsfnb_${sp}_ouS6I2M=$ck"   # 朝向学生 ×2，读原图 + 镜像 = 4 个读数
done
# 整链 + 部署时两种距离合并（w 网格；主数字取 w=0.15，常数 = 无标签训练集，部署 b_k = 无标签验证集图像）
FE_NMEM=4 FE_UNC=z2S6Ik FE_ORI=ouS6I2M FE_CONST=train FE_COLOR=new FE_KPS=strict FE_HT=color FE_OUT=simpk python fuse_eval.py
cd $W/wf24 && python summ2.py w15 simpk                                          # 论文口径：UDS / AP / ATE / ASE / AOE / 航向中位 / 近中远
# 区间：cd $W/wf23/mirror && python boot_ci.py <对照标签> simpk_w15 300
