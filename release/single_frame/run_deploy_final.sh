#!/bin/bash
# 完全单帧最终版（11-09）的部署 + 打分命令（只推理；真值只用于打分）。没有从头到尾连续跑过一次。
# 前提：run_train_final.sh 的产物（softkd/XFFZ_s{5..9}.pt、wf21/final/adabn_all/PSPM_s{0..4}.pth）；
#       检测器部署记录 night1003/repl/wsfnb_{val,test}_kzt7q 与 wsfnb_ens/ens_{val,test}.pkl、recs/wsfnb_train.pkl（11-08 单帧整链）。
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
G=$R/output/camnorm/geo1004; W=$G/gproto; F=$W/wf21/final
export PYTHONUTF8=1

cd $W/wf21/strict && python final_unc.py fz5 XFFZ 5,6,7,8,9 val,test,train      # 测距学生逐检测输出 + 颜色原型亮度
cd $W/wf23/mirror
for sp in val test; do
  ck=""; for k in 0 1 2 3 4; do ck="$ck,$F/adabn_all/PSPM_s$k.pth"; done; ck=${ck#,}
  python recut_mirror.py --src-tag wsfnb_${sp}_kzt7q --sets "wsfnb_${sp}_ouPSPMallM=$ck"   # 朝向 5 个 + 镜像读数 = 10 成员
done
# 整链 + 部署两种测距合并（w 网格；论文主数字取 w=0.15，常数 = 无标签训练集）
FE_NMEM=10 FE_UNC=fz5 FE_ORI=ouPSPMallM FE_CONST=train FE_COLOR=new FE_KPS=strict FE_HT=color FE_OUT=final_train python fuse_eval.py
# 区间（需先有对照臂的 fe_<标签>_<split>_P.pkl）：python boot_ci.py <对照标签> final_train_w15 300
