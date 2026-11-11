#!/bin/bash
# 11-10 无标签链（PREREG.md）：在新位置检测器 <arm>（nl100 / nl075 / nl050）上重建全部依赖检测器或伪标签的产物，重训第二段第 2、3 步（种子 5、6），
# 重做朝向推理 / BN / 高度读数，重算常数，w 按仿真验证集逆方差（w_sim.py），α=0.5，不重排、不在验证集上选 w，最后一次性评测 val/test。
# 每一步输出已存在就跳过（可续跑）。所有产物名带 <arm>，不覆盖旧产物。
# 用法：bash run_chain_nl.sh <arm> [阶段]   阶段缺省 = all；dumps 只跑三路推理
ARM=$1; PH=${2:-all}
NL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$NL/../../../../.." && pwd); RW=$(cd "$R" && (pwd -W 2>/dev/null || pwd))   # 仓库根（RW = Windows 写法，给 python 参数用）
G=$R/output/camnorm/geo1004; GW=$RW/output/camnorm/geo1004
A=$NL/$ARM; AW=$GW/gproto/nolabel/$ARM
PY=${PY:-python}; GATE=${GATE:-:}   # GATE：可选的"等显卡空闲"命令（研究时 = bash gproto/gpu_gate.sh）；缺省不等
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=@@TORCH_HOME@@ NL_ARM=$ARM
mkdir -p $A/recs $A/tags $A/logs $A/unc $A/kps $A/adabn $A/adabn_all
L=$A/logs/driver.log
say() { echo "$(date '+%m-%d %H:%M:%S') [$ARM] $1" | tee -a $L; }
die() { say "!! $1"; exit 2; }
M=$RW/output/models/uavdet_3d/camnorm
CFGD=$($PY -c "import sys; sys.path.insert(0,'$GW/proto'); import run_proto as R; print(R.ARMS['$ARM'][0], R.ARMS['$ARM'][2])")
set -- $CFGD; DC=$1; DT=$2
CK0=$M/$DC/${DT}_s0/ckpt/checkpoint_epoch_3.pth; CK1=$M/$DC/${DT}_s1/ckpt/checkpoint_epoch_3.pth; CK2=$M/$DC/${DT}_s2/ckpt/checkpoint_epoch_3.pth
REAL=cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml
SETS="--set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width"

# ---------- 阶段 dumps：三路推理（只推理；真值不存 / 封存）
jobA() {  # 训练集全部帧单帧检测 → 背景库选窗、训练集第二段输出（常数、颜色、build_targets 的颜色类）
  [ -f $A/recs/${ARM}_train.pkl ] && return
  $GATE; say "A 训练集检测 开始"
  (cd $R/tools && env PYTHONPATH=$RW DUMP_PKL=$AW/recs/${ARM}_train.pkl $PY $NL/eval_dump_nl.py --cfg $REAL --ckpt $CK0 --ensemble $CK1 $CK2 \
    --flip-tta --tag nl_${ARM}_train --split train --interval 1 --workers 0 --match greedy --self-calib val $SETS > $A/logs/dumpA.log 2>&1)
  [ -f $A/recs/${ARM}_train.pkl ] && say "A 结束" || die "A 失败"
}
jobC() {  # 训练集间隔 3 的朝向钩子窗口 → 朝向 BN、高度网络 BN、高度比训练集常数
  [ -f $A/tags/${ARM}_train_kz/kz_members_${ARM}_train_kz.npz ] && return
  $GATE; say "C 训练集朝向窗口（间隔 3）开始"
  (cd $R/tools && $PY $NL/dump_train_kz_nl.py $ARM 3 ${ARM}_train_kz > $A/logs/dumpC.log 2>&1)
  [ -f $A/tags/${ARM}_train_kz/kz_members_${ARM}_train_kz.npz ] && say "C 结束" || die "C 失败"
}
jobD() {  # 训练集间隔 1 的朝向钩子窗口 → 训练帧上的角点距离 z_kp（合并伪目标）
  [ -f $A/tags/${ARM}_train1_kz/kz_members_${ARM}_train1_kz.npz ] && return
  $GATE; say "D 训练集朝向窗口（间隔 1）开始"
  (cd $R/tools && $PY $NL/dump_train_kz_nl.py $ARM 1 ${ARM}_train1_kz > $A/logs/dumpD.log 2>&1)
  [ -f $A/tags/${ARM}_train1_kz/kz_members_${ARM}_train1_kz.npz ] && say "D 结束" || die "D 失败"
}
dumps() {
  [ -f $G/${ARM}_ens/ens_val.pkl ] && [ -f $G/${ARM}_ens/ens_test.pkl ] || die "缺部署检测 ${ARM}_ens（RP.infer 未完成）"
  for sp in val test; do [ -f $R/output/camnorm/night1003/repl/${ARM}_${sp}_kzt7q/kz_members_${ARM}_${sp}_kzt7q.npz ] || die "缺 ${ARM}_${sp}_kzt7q"; done
  jobA & sleep 60; jobC & wait
  jobD
}
case $PH in dumps) dumps; exit 0;; esac

# ---------- 第二段（红外教的测距网络）第 2 步：真实适应（种子 5、6；第 1 步 V2P_s{5,6}_st 纯仿真，复用）
v2run() {  # $1 = 阶段名（V2Pp | V2Rp），$2 = 种子，其余 env
  local st=$1 j=$2; shift 2
  $GATE; say "第二段 $st s$j 开始"
  (cd $G && env PYTHONUTF8=1 V2_TAG=st_$ARM V2_BANK=sf2_$ARM V2_REAL=nbm_${ARM}_ "$@" $PY -u $NL/secondstage_v2_nl.py $st $j > $A/logs/v2_${st}_s$j.log 2>&1)
  say "第二段 $st s$j 结束 $?"
}
stage2() {
  for j in 5 6; do cp -n $G/softkd/V2P_s${j}_st.pt $G/softkd/V2P_s${j}_st_$ARM.pt; done
  local todo=""
  for j in 5 6; do [ -f $G/softkd/V2Pp_s${j}_st_$ARM.pt ] || todo="$todo $j"; done
  if [ -n "$todo" ]; then
    set -- $todo
    v2run V2Pp $1 & [ -n "$2" ] && { sleep 120; v2run V2Pp $2 & }; wait
  fi
  for j in 5 6; do [ -f $G/softkd/V2Pp_s${j}_st_$ARM.pt ] || die "第二段第 2 步 s$j 没生成"; cp -n $G/softkd/V2Pp_s${j}_st_$ARM.pt $G/softkd/XFPp${ARM}_s$j.pt; done
}
stage3() {
  local todo=""
  for j in 5 6; do [ -f $G/softkd/Z${ARM}_s${j}_st_$ARM.pt ] || todo="$todo $j"; done
  if [ -n "$todo" ]; then
    set -- $todo
    v2run V2Rp $1 V2_DPS=nbm_${ARM}_dpfz V2_OUTARM=Z$ARM & [ -n "$2" ] && { sleep 120; v2run V2Rp $2 V2_DPS=nbm_${ARM}_dpfz V2_OUTARM=Z$ARM & }; wait
  fi
  for j in 5 6; do [ -f $G/softkd/Z${ARM}_s${j}_st_$ARM.pt ] || die "第二段第 3 步 s$j 没生成"; cp -n $G/softkd/Z${ARM}_s${j}_st_$ARM.pt $G/softkd/XFZ${ARM}_s$j.pt; done
}
# ---------- 朝向 / 高度：BN 重估、高度读数、镜像重裁（GPU 推理，不训练）
orient() {
  CKS="$AW/adabn_all/S6I_s0.pth,$AW/adabn_all/S6I_s1.pth"
  [ -f $A/adabn_all/S6I_s1.pth ] || { $GATE; say "朝向学生 BN 重估"; $PY $NL/adabn_all_nl.py $ARM S6I > $A/logs/adabn_all.log 2>&1 || die "adabn_all 失败"; }
  [ -f $A/adabn/kps_s0_adabn.pth ] || { $GATE; say "高度网络 BN 重估"; $PY $NL/kps_nl.py $ARM adabn > $A/logs/kps_adabn.log 2>&1 || die "kps adabn 失败"; }
  [ -f $A/kps/shape_test_${ARM}_k0.pkl ] || { $GATE; say "高度读数"; $PY $NL/kps_nl.py $ARM shape train val test > $A/logs/kps_shape.log 2>&1 || die "kps shape 失败"; }
  for sp in val test; do
    [ -f $A/tags/${ARM}_${sp}_ouS6I2M/kz_members_${ARM}_${sp}_ouS6I2M.json ] && continue
    $GATE; say "镜像重裁 $sp"
    (cd $R/tools && RM_TAGS=$AW/tags $PY $G/gproto/wf23/fusetrain/recut_mirror_t.py --src-tag ${ARM}_${sp}_kzt7q --sets "${ARM}_${sp}_ouS6I2M=$CKS" > $A/logs/recut_$sp.log 2>&1) || die "recut $sp 失败"
  done
  if [ ! -f $A/tags/${ARM}_train1_ouS6I2M/kz_members_${ARM}_train1_ouS6I2M.json ]; then
    $GATE; say "镜像重裁 train1"
    (cd $R/tools && RM_REPL=$AW/tags RM_TAGS=$AW/tags $PY $G/gproto/wf23/fusetrain/recut_mirror_t.py --src-tag ${ARM}_train1_kz --sets "${ARM}_train1_ouS6I2M=$CKS" > $A/logs/recut_train1.log 2>&1) || die "recut train1 失败"
  fi
  say "朝向 / 高度 完成"
}
pre2() {
  jobA
  [ -f $G/xpaste/bank_bg_sf2_$ARM.npy ] || { say "背景库"; $PY $NL/xpaste_bank_sf2_nl.py $ARM > $A/logs/bank.log 2>&1 || die "背景库失败"; grep -v Warn $A/logs/bank.log | tee -a $L; }
  [ -f $G/selftrain/real_s1_nbm_${ARM}_dp.npy ] || { say "第二段真实裁窗"; $PY $NL/extract_nbm_nl.py $ARM 0 1 > $A/logs/extract.log 2>&1 || die "裁窗失败"; grep -v Warn $A/logs/extract.log | tee -a $L; }
}
allsteps() {
  dumps
  pre2
  orient > $A/logs/orient.out 2>&1 &
  OPID=$!
  stage2
  wait $OPID; [ -f $A/tags/${ARM}_train1_ouS6I2M/kz_members_${ARM}_train1_ouS6I2M.json ] || die "朝向 / 高度 没完成（见 orient.out）"
  [ -f $A/w_sim.json ] || { $GATE; say "w（仿真验证集）"; $PY $NL/w_sim.py $ARM > $A/logs/w_sim.log 2>&1 || die "w_sim 失败"; }
  say "w = $($PY -c "import json;print(json.load(open('$AW/w_sim.json'))['w'])")"
  [ -f $A/unc/unc_train_pp2.pkl ] || { $GATE; say "第二段第 2 步训练集输出"; $PY $NL/final_unc_nl.py $ARM pp2 XFPp$ARM 5,6 train > $A/logs/unc_pp2.log 2>&1 || die "unc pp2 失败"; }
  [ -f $A/ztrain_${ARM}_train1_ouS6I2M.pkl ] || { say "训练帧角点距离"; FE_UNC=pp2 ZT_NMEM=4 $PY $NL/ztrain_nl.py ${ARM}_train1_ouS6I2M ${ARM}_train1_kz > $A/logs/ztrain.log 2>&1 || die "ztrain 失败"; }
  [ -f $G/selftrain/real_s1_nbm_${ARM}_dpfz.npy ] || { say "合并伪目标"; FE_UNC=pp2 $PY $NL/build_targets_nl.py > $A/logs/build_targets.log 2>&1 || die "build_targets 失败"; grep -v Warn $A/logs/build_targets.log | tee -a $L; }
  stage3
  [ -f $A/unc/unc_train_z2.pkl ] || { $GATE; say "第二段第 3 步输出（val/test/train）"; $PY $NL/final_unc_nl.py $ARM z2 XFZ$ARM 5,6 val,test,train > $A/logs/unc_z2.log 2>&1 || die "unc z2 失败"; }
  [ -f $NL/results_$ARM.json ] || { say "冻结整链 + 一次性评测"; FE_UNC=z2 FE_ORI=ouS6I2M FE_NMEM=4 $PY $NL/fuse_eval_nl.py > $A/logs/fuse_eval.log 2>&1 || die "fuse_eval 失败"; }
  grep -v Warn $A/logs/fuse_eval.log | tee -a $L
  say "全部完成"
}
case $PH in all) allsteps;; orient) orient;; stage2) pre2; stage2;; *) die "未知阶段 $PH";; esac
