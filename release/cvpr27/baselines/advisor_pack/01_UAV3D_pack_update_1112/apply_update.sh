#!/usr/bin/env bash
# Apply the 11-12 update to the 11-11 pack. AntiUAV3D data are replaced with v2 (intrinsics verified and re-split);
# also adds the focal-unification option. Old AntiUAV3D results are moved, not deleted.
#   bash apply_update.sh <path to the 11-11 pack root, i.e. the folder that holds scripts/ and data/>
# Run it once. It refuses to run while tasks of the pack are still running.
set -e
U="$(cd "$(dirname "$0")" && pwd)"
P="${1:?usage: bash apply_update.sh <pack root>}"; P="$(cd "$P" && pwd)"
[ -f "$P/scripts/run.sh" ] && [ -d "$P/data/cmpfull" ] || { echo "$P does not look like the pack root (no scripts/run.sh or data/cmpfull)"; exit 1; }
if pgrep -f "scripts/run.sh" >/dev/null 2>&1; then echo "Pack tasks are still running (scripts/run.sh). Wait for them or stop them first, then run this again."; exit 1; fi

echo "== 1. Replace AntiUAV3D data (old outdoor_train / outdoor_test are AntiUAV3D v1; they are not used any more)"
rm -rf "$P/data/cmpfull/outdoor_train" "$P/data/cmpfull/outdoor_test" "$P/data/cmpfull/outdoor_train_fu" "$P/data/cmpfull/outdoor_test_fu"
cp -r "$U/data/cmpfull/outdoor_train" "$U/data/cmpfull/outdoor_test" "$P/data/cmpfull/"
cp "$U/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl" "$P/data/cmpfull/"
cp "$U/data/gt/outdoor_test_meta.npz" "$P/data/gt/"

echo "== 2. Replace scripts and docs"
for f in scripts/run.sh scripts/score.py scripts/tasks.txt scripts/focal_unify.py README.md 任务清单.md \
         Open3DUAVDet/output/camnorm/geo1004/cmpfull/build_pcmd_labels.py; do
  [ -f "$P/$f" ] && cp "$P/$f" "$P/$f.bak_1111"
  cp "$U/$f" "$P/$f"
done
# the paths in the code are rewritten to the pack location at install time, same as setup/install.sh (env.sh resolves PKG itself, so these scripts need no rewrite)

echo "== 3. Move old AntiUAV3D results to runs/_antiuav3d_v1_old/ and clear their queue markers so they rerun on the new data"
B="$P/runs/_antiuav3d_v1_old"; mkdir -p "$B"
shopt -s nullglob
for d in "$P"/runs/*_st_anti_s* "$P"/runs/*_kd_anti_s*; do echo "  move $(basename "$d")"; mv "$d" "$B/"; done
for f in "$P"/runs/*_sim_s*/pred_outdoor_test.pkl; do n=$(basename "$(dirname "$f")"); echo "  move $n/pred_outdoor_test.pkl (the model is kept; only the AntiUAV3D test prediction is redone)"; mv "$f" "$B/${n}__pred_outdoor_test.pkl"; done
for f in "$P"/runs/queue/*_st_anti_s*.* "$P"/runs/queue/*_kd_anti_s*.* "$P"/runs/queue/*_sim_both_s*.done; do rm -rf "$f"; done
# sim tasks: once the .done marker is removed, run_all skips training (the final checkpoint exists) and predicts only the missing AntiUAV3D test

echo "== 4. Build the focal-unified data (optional set D; takes a few minutes)"
(cd "$P" && ${PY:-python} scripts/focal_unify.py)

echo
echo "Done. Then run as before: bash scripts/run_all.sh (it continues from where it stopped; only AntiUAV3D-related tasks are redone)."
echo "Old AntiUAV3D results are in runs/_antiuav3d_v1_old/; replaced files have a .bak_1111 copy next to them."
