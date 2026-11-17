#!/bin/bash
# [review1113 stage1seeds] SHARED stage-1 outputs of one target x seed set (used by the PST, E1 and E4 runs of that seed set). NOT RUN by its author.
#   usage: bash prep_shared.sh <tgt mav|antifs> <sd 30|40>          DRY=1: build the lib copy, print the config, check inputs; no GPU
# = the "fixed stage-1 outputs" that seeds/run_seed_chain.sh COPIES from the reported run, recomputed here with the re-seeded sim-only detectors
#   (same code: srcx/run_x.sh lines det0 / refiner sim / bank / real):
#   recs/det0.pkl (3 re-seeded sim detectors, ensemble + flip; s_bar from unlabeled val images), preds/srconly_{val,test}.pkl (never scored here),
#   ref/sim_meta.pkl, sim_rgb.npy, sim_sil.npy (IR silhouette; PST/E4), sim_hull.npy (box hull of the same samples; E1), bank.npy,
#   real_d0_meta.pkl, real_d0_rgb.npy. These depend only on the detectors + target data, not on SX_TEACH (checked: refiner_x sim/bank/real and
#   det_infer_x never read sx.NOIR / sx.TEACH). wsim_cache.pkl (sim-detector-s0 inference on sim val) is shared by the first finished run.
# Outputs: review1113/stage1seeds/shared/<tgt>_sd<sd>/ (+ .done), config srcx/real_<C>_s1prepsd<sd>.yaml. No real labels are read.
TGT=$1; SD=$2; DRY=${DRY:-0}
case "$TGT/$SD" in mav/30|mav/40|antifs/30|antifs/40) ;; *) echo "usage: prep_shared.sh <mav|antifs> <30|40>"; exit 2 ;; esac
source /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/common.sh
SRC=in; [ $TGT = antifs ] && SRC=out
export SX_TEACH=full SX_SRC=$SRC SX_TGT=$TGT SX_SEEDSHIFT=$SD
X=$S1/shared/${TGT}_sd$SD; XW=$(wp $X); L=$XW/lib; IW="$PY $L/infer_wrap.py"
mkdir -p $S1/logs; TAGL="prep $TGT sd$SD"; MARK=$X/
if [ $DRY = 1 ]; then
  LOG=/dev/null; GLOG=/dev/null
  $PY $S1W/make_s1_lib.py prep $TGT $SD || exit 2
  SX_NOMKDIR=1 CUDA_VISIBLE_DEVICES="" $PY -c "
import sys, os; sys.path.insert(0, '$L'); import sx
print('  prep %s sd%s | O %s | W %s | REAL_CFG %s' % ('$TGT', '$SD', sx.O, sx.W, sx.REAL_CFG))
for i, p in enumerate(sx.S['det']): print('  sim detector %d: %s (exists now: %s)' % (i, p, os.path.exists(p)))
print('  source sim cache %s exists: %s | target cache %s exists: %s' % (sx.SIM, os.path.exists(sx.SIM), sx.REAL, os.path.exists(sx.REAL)))
assert sx.O == '$XW' and not os.path.exists(sx.ROOT + '/tools/' + sx.REAL_CFG), 'config name collides'
print('  config name free:', sx.REAL_CFG)" || exit 2
  grep -q "sim_hull.npy" $L/refiner_x.py && echo "  refiner sim writes sim_hull.npy: yes" || exit 2
  exit 0
fi
mkdir -p $X/recs $X/ref $X/preds $X/logs; LOG=$X/logs/prep.log; RLOG=$X/logs/retry.log
[ -f $X/.done ] && { say "already done"; exit 0; }
rm -f $X/.failed
say "start shared stage-1 outputs (re-seeded sim detectors of seed set $SD)"
$PY $S1W/make_s1_lib.py prep $TGT $SD >> $LOG 2>&1 || die make_s1_lib
waitnet $SRC $SD det:0 det:1 det:2
eval "$($PY -c "
import sys; sys.path.insert(0, '$L'); import sx; S = sx.S
assert sx.O == '$XW', sx.O
print('DET0=%s DET1=%s DET2=%s' % tuple(S['det']))")" || die "sx config"
$PY $L/mkcfg.py >> $LOG 2>&1 || die mkcfg
cd $X
[ -f recs/det0.pkl ] || oomrun logs/det0_infer.log "DET_BS=8" $SLOT light $IW $L/det_infer_x.py --ckpt $DET0 $DET1 $DET2 --splits train val test --out $XW/recs/det0.pkl || die "det0 infer"
grep -E "s_bar|frames" logs/det0_infer.log | tail -4 >> $LOG
for sp in val test; do [ -f preds/srconly_$sp.pkl ] || $PY $L/to_pred.py $XW/recs/det0.pkl $sp $XW/preds/srconly_$sp.pkl >> $LOG 2>&1 || die "to_pred $sp"; done
[ -f ref/sim_hull.npy ] || oomrun logs/ref_sim.log "REF_SIM_BS=8" $SLOT light $IW $L/refiner_x.py sim $DET0 || die "ref sim"
tail -1 logs/ref_sim.log >> $LOG
$PY -c "
import numpy as np, pickle, json
R = '$XW/ref'; m = pickle.load(open(R + '/sim_meta.pkl', 'rb')); s = np.load(R + '/sim_sil.npy', mmap_mode='r'); h = np.load(R + '/sim_hull.npy', mmap_mode='r')
assert s.shape == h.shape and len(m) == len(s), (s.shape, h.shape, len(m))
a = np.asarray(s) > 128; b = np.asarray(h) > 128
inside = float((a & b).sum() / max(a.sum(), 1))
info = dict(n=len(m), ir_sil_mass_inside_hull=inside, hull_area_frac_mean=float(b.mean()), ir_sil_area_frac_mean=float(a.mean()),
            hull_empty_frac=float(np.mean(~b.any((1, 2)))), ir_sil_empty_frac=float(np.mean(~a.any((1, 2)))))
json.dump(info, open(R + '/hull_check.json', 'w'), indent=1); print('hull check', json.dumps(info))
assert inside > 0.8, 'IR silhouette mostly outside the box hull: crop geometry wrong'" >> $LOG 2>&1 || die "hull check"
say "$(grep 'hull check' $LOG | tail -1 | cut -c1-200)"
[ -f ref/bank.npy ] || $PY $L/refiner_x.py bank $XW/recs/det0.pkl >> $LOG 2>&1 || die "ref bank"
[ -f ref/real_d0_meta.pkl ] || $PY $L/refiner_x.py real $XW/recs/det0.pkl d0 >> $LOG 2>&1 || die "ref real"
for f in recs/det0.pkl preds/srconly_val.pkl preds/srconly_test.pkl ref/sim_meta.pkl ref/sim_rgb.npy ref/sim_sil.npy ref/sim_hull.npy ref/bank.npy ref/real_d0_meta.pkl ref/real_d0_rgb.npy; do
  [ -f $f ] || die "missing $f"; done
say "shared stage-1 outputs done"
touch $X/.done
