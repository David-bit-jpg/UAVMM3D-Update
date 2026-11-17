#!/bin/bash
# [cleanfix B2] eta^2 domain gap on CLEAN real crops (replaces paper 3_method.tex:65 / X_suppl.tex:66 "development-pipeline crops")
#   + t-SNE CSV D:/CVPR27_UAV/figures/data/tsne_domain_gap.csv (domain, region, x, y).
# Method = geo1004/partgap/partgap.py unchanged (KPSs0..2 sim weights, 885 sim samples, layers 2/3/4/head, 8 pools, domain_gap.metrics),
# only the real side = 2,000 crops (linspace) of srcx/in2mav det1 TRAIN crops (oc_tr1.npy). See partgap_clean.py header. Inference only, no labels.
# Robustness: waits for GUPNet-UDA-MT + initabl done markers (poll 10 min); GPU via gpu_slot.sh light; OOM -> PG_BS 16 -> 8 -> 4 -> 2 -> 1
#   (retry.log); per-seed resume (partgap_clean.json + feats npz); no DataLoader (workers 0 n/a); done marker .done. Never kills processes.
# GPU estimate: original partgap run ~3-5 min per seed (2,885 crops x 1 net, bf16) -> ~15 min incl. queueing; t-SNE CPU ~12 min.
GP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; D=$GP/review1113/cleanfix/B2_etasq; DW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B2_etasq
export PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash $GP/gpu_slot.sh"
CSV=D:/CVPR27_UAV/figures/data/tsne_domain_gap.csv
LOG=$D/run.log; say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
[ -f $D/.done ] && { say "already done"; exit 0; }
n=0; until [ -f /d/cmpfull_runs/gupnet_uda_mt_s0/.done ] && [ -f $GP/review1113/initabl/.done ]; do
  [ $n = 0 ] && say "waiting for gupnet_uda_mt_s0/.done and review1113/initabl/.done (poll 10 min)"; n=1; sleep 600; done
say "start B2"
bs=16
until [ -f $D/feats_KPSs2.npz ] && grep -q KPSs2 $D/partgap_clean.json 2>/dev/null; do
  PG_BS=$bs $SLOT light $PY $DW/partgap_clean.py > $D/partgap.try 2>&1; rc=$?; cat $D/partgap.try >> $LOG
  [ -f $D/feats_KPSs2.npz ] && break
  if grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" $D/partgap.try && [ $bs -gt 1 ]; then
    bs=$((bs / 2)); say "OOM -> PG_BS $bs (resumes at the next seed)" | tee -a $D/retry.log
  else say "!! partgap_clean failed rc=$rc (not OOM or batch 1), see $LOG"; rm -f $D/partgap.try; exit 2; fi
done; rm -f $D/partgap.try
say "eta2 done: $(grep -E '^\| layer2 \| (win|ctr|motor|body8) ' $D/PARTGAP_CLEAN_SUMMARY.md | tr '\n' ' ')"
[ -f $D/tsne_domain_gap.csv ] || $PY $DW/tsne_csv.py $DW/feats_KPSs0.npz $DW/tsne_domain_gap.csv layer2 >> $LOG 2>&1 || { say "!! tsne"; exit 2; }
mkdir -p /d/CVPR27_UAV/figures/data && cp $D/tsne_domain_gap.csv $CSV || { say "!! copy csv"; exit 2; }
say "ALL DONE (summary $DW/PARTGAP_CLEAN_SUMMARY.md, csv $CSV)"; touch $D/.done
