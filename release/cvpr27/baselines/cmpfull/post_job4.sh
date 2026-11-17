#!/bin/bash
# 等某个训练任务结束 → 真实测试集预测 → 接进对比表。用法：bash post_job.sh <job>（如 gupnet_s0） （v4：加 MonoDETR/MonoCoP）
H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull; R=/d/cmpfull_runs; PY=D:/Miniconda3/envs/city/python.exe
job=$1; m=${job%_s*}; s=${job##*_s}
export PYTHONUTF8=1 GPU_FRAC=0.15 TORCH_HOME=E:/torch_home
until grep -q "结束 $job 退出码 0" $H/driver_*.log 2>/dev/null; do sleep 300; done
case $m in
  gupnet)  ck=$R/$job/checkpoints/checkpoint_final.pth ;;
  monocd)  ck=$R/$job/model_final.pth ;;
  monodgp) ck=$(ls -t $R/$job/checkpoint*.pth 2>/dev/null | head -1) ;;
  monodetr) ck=$R/$job/model_epoch_37.pth ;;
  monocop) ck=$R/$job/model_epoch_${MONOCOP_EPOCHS:-48}.pth ;;
esac
[ -f "$ck" ] || { echo "$(date '+%m-%d %T') $job 没有最终存档 $ck，不预测" >> $H/driver_post.log; exit 1; }
out=$R/$job/pred_real_test.pkl
case $m in
  gupnet)  (cd $H/gupnet; $PY predict.py --ckpt $ck --split real_test --out $out --thresh 0) ;;
  monocd)  (cd $H/monocd; $PY predict.py --ckpt $ck --split real_test --out $out --thresh 0) ;;
  monodgp) (cd $H/monodgp; $PY predict.py --ckpt $ck --split real_test --out $out) ;;
  monodetr|monocop) (cd $H/$m; $PY predict.py --ckpt $ck --split real_test --out $out) ;;
esac > $H/pred_$job.log 2>&1
NAME=$(echo ${m}F | tr a-z A-Z)
$PY $H/to_pf.py $out $NAME $s >> $H/pred_$job.log 2>&1
(cd /e/Open3DUAVDet/output/camnorm/geo1004/cmpeval; $PY paper_table.py adabn > /dev/null 2>&1)
echo "$(date '+%m-%d %T') $job 预测+入表完成（$ck）" >> $H/driver_post.log
