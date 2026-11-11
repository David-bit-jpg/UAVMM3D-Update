#!/usr/bin/env bash
# 最终方法（11-10 无标签版，论文所写的版本）的最后一段：位置伪标签（只按置信度筛帧，LAM × 全局中位修正）→ 检测器弱增强自训练 3 种子 → 部署推理
# → 无标签整链 output/camnorm/geo1004/gproto/nolabel/run_chain_nl.sh（第二段第 2/3 步、朝向学生 BN 重估、高度读数、w、合并伪目标、冻结后一次性评测）。
# 规则见 output/camnorm/geo1004/gproto/nolabel/PREREG.md；整条链里没有任何一步读真实标签（评测除外，见 selfcheck_nl.sh）。
# 前提：
#   1) release/run_upstream.sh 已跑完（仿真检测器、ST2 五种子、第二段 V2Pp_s0..4、朝向/高度网络等上游产物）；
#   2) release/single_frame/run_train_final.sh 的朝向学生（gproto/wf21/orient_uda/runs/S6I_s{0,1}）与第二段第 1 步 softkd/V2P_s{5,6}_st.pt 已有
#      （朝向学生不重训，本脚本只在新检测器窗口上重估它们的 BN）；
#   3) 统一评测帧表 cmpeval/{val,test}_meta.npz（release/run_final_3seed.sh 第 0 步，真值只用于打分）。
# 【未端到端连续跑过】：每条命令与研究时 11-10 的实际运行一致（研究时由排队脚本串起来），本脚本本身没有从头连续跑过。已完成的步骤自动跳过。
# 用法：bash release/run_final_nolabel.sh                 # 主结果：LAM = 0.75（臂 nl075）
#       LAM=1.0 ARM=nl100 bash release/run_final_nolabel.sh   # 敏感性（补充材料）；LAM=0.5 ARM=nl050 同理
set -e
source "$(dirname "$0")/env.sh"
LAM=${LAM:-0.75}; ARM=${ARM:-nl075}
NL=$G/gproto/nolabel
LOG=$RUNS/nolabel/logs; mkdir -p "$LOG"
say() { echo "[$(date '+%F %T')] $*"; }
need() { for f in "$@"; do [ -s "$f" ] || { echo "缺少 $f（上游没跑完？）"; exit 1; }; done; }
MST2=$M/st2_mav6d_r34_geo_full_tr
for k in 0 1 2 3 4; do need $MST2/ST2_s$k/ckpt/checkpoint_epoch_3.pth; done
for j in 5 6; do need $G/softkd/V2P_s${j}_st.pt; done
for k in 0 1; do need $G/gproto/wf21/orient_uda/runs/S6I_s$k/ckpt_last.pth; done
for sp in val test; do need $G/cmpeval/${sp}_meta.npz; done

# ---------- 1. 去真值帧表（只留 seq / frame / 内参，下游不再打开带真值的 index.pkl）----------
[ -s $NL/test_frames_nogt.pkl ] || { say "去真值帧表"; $PY $NL/make_train_nogt_index.py > $LOG/nogt_index.log 2>&1; }

# ---------- 2. 位置伪标签的输入：ST2 五集成在真实训练帧上的单帧检测（无热图时序背景）+ 第二段 V2Pp_s0..4 的修正量 ----------
# ST2 = 最早的自举检测器；它训练时用过跨帧筛过的伪标签（唯一的间接例外，见 release/README.md）
if [ ! -s $R/output/camnorm/st/real_train_preds_st2nb.pkl ]; then
  say "ST2 训练帧检测（--bg none）"
  (cd $R/tools && $PY annot_audit/dump_real_preds.py --split train --bg none --ckpt $MST2/ST2_s0/ckpt/checkpoint_epoch_3.pth \
     --ensemble $MST2/ST2_s1/ckpt/checkpoint_epoch_3.pth $MST2/ST2_s2/ckpt/checkpoint_epoch_3.pth $MST2/ST2_s3/ckpt/checkpoint_epoch_3.pth $MST2/ST2_s4/ckpt/checkpoint_epoch_3.pth \
     --out ../output/camnorm/st/real_train_preds_st2nb.pkl) > $LOG/dump_st2nb.log 2>&1
fi
[ -s $G/st2m/delta_IR_nb.pkl ] || { say "第二段修正量"; (cd $G/gproto && $PY -u strict_deltas.py) > $LOG/strict_deltas.log 2>&1; }

# ---------- 3. 位置伪标签：c = LAM × 全训练集修正量中位；只按检测置信度 ≥ 0.5 筛帧；不读真值 ----------
[ -s $CN/train/index_pseudo_$ARM.pkl ] || { say "位置伪标签 $ARM（LAM=$LAM）"; (cd $NL && CUDA_VISIBLE_DEVICES= LAM=$LAM TAG=$ARM $PY make_nolabel_pseudo.py) | tee $LOG/pseudo_$ARM.log; }

# ---------- 4. 检测器弱增强自训练 3 种子（起点 = 仿真检测器 ramp_s{k}，3 轮）+ 部署推理（3 集成 + 翻转，单帧无时序背景；朝向裁窗 kzt7q）----------
# 配置 tools/cfgs/models/uavdet_3d/camnorm/st2${ARM}wa_mav6d_r34_geo_full_tr.yaml；驱动 = output/camnorm/geo1004/proto/run_proto.py 的 train() / infer()
say "检测器 $ARM 3 种子 + 推理"
(cd $G/proto && GP_WORKERS=0 $PY -c "
import run_proto as RP
for k in range(3):
    RP.train('$ARM', k)
RP.infer('$ARM')") > $LOG/detector_$ARM.log 2>&1

# ---------- 5. 无标签整链 + 冻结后一次性评测 ----------
say "无标签整链 $ARM"
PY=$PY bash $NL/run_chain_nl.sh $ARM 2>&1 | tee $LOG/chain_$ARM.out
say "结果：$NL/results_$ARM.json（同 release/results/nolabel_1110/）"

# ---------- 6. 无真值自查（grep 链上每个脚本的真值字段）----------
bash $NL/selfcheck_nl.sh > /dev/null && say "自查清单：$NL/selfcheck_nl.txt"
