#!/usr/bin/env bash
# 11-12 patch b: fixes the three causes of the 11-11 failures.
#   1) The path placeholder @@ROOT@@ in the code was never replaced with the pack location (setup/install.sh stopped before the
#      fix_paths.py step; it is the last step). Effect: GUPNet's file paths are wrong; MonoDETR / MonoCoP / MonoDGP train fine but
#      their prediction cannot find labels.pkl; the config of the bundled MonoDGP model cannot be parsed.
#   2) The packed zip left out monocd/runs/ (MonoCD's two config files), so MonoCD could not start.
#   3) run_all.sh now runs fix_paths automatically before starting workers and checks these two points again;
#      malformed task lines are skipped.
# Usage: bash fix_1112b.sh <pack root>
# Safe to run while other tasks are still training (they get the new paths when they reach the prediction step; nothing is interrupted).
set -e
U="$(cd "$(dirname "$0")" && pwd)"; P="$(cd "${1:?usage: bash fix_1112b.sh <pack root>}" && pwd)"
[ -f "$P/scripts/run.sh" ] || { echo "$P is not the pack root"; exit 1; }
cd "$P"
echo "== 1. Add MonoCD configs"
mkdir -p Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/runs
cp "$U"/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/runs/*.yaml Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/runs/
echo "== 2. Replace path placeholders"
${PY:-python} setup/fix_paths.py
n=$(grep -rl "@@ROOT@@" Open3DUAVDet/output/camnorm/geo1004/cmpfull/*/predict.py Open3DUAVDet/output/camnorm/geo1004/cmpfull/*/train.sh pretrained 2>/dev/null | wc -l)
[ "$n" = 0 ] && echo "   all replaced" || { echo "   $n files still contain @@ROOT@@; send me this output"; exit 1; }
echo "== 3. Update run_all.sh"
cp scripts/run_all.sh scripts/run_all.sh.bak_1112b; cp "$U/scripts/run_all.sh" scripts/run_all.sh
echo "== 4. Requeue failed tasks (only those with a .fail marker; running tasks are untouched)"
shopt -s nullglob; k=0
for f in runs/queue/*.fail; do id=$(basename "$f" .fail); rm -rf "runs/queue/$id.fail" "runs/queue/$id.lock"; k=$((k+1)); done
echo "   requeued $k tasks"
echo
echo "Done. Start workers only on idle GPUs (memory used near 0 MiB in nvidia-smi), e.g.:"
echo "   GPUS=\"0 3\" bash scripts/run_all.sh"
echo "Workers still running on the other GPUs pick up the requeued tasks automatically when their current task ends."
echo "Trained models are kept (e.g. MonoDGP / MonoDETR / MonoCoP out-source sim): the rerun skips training and only redoes prediction."
