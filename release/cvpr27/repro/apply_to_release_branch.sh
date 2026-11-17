#!/bin/bash
# Put the prepared code release (repro_1117/release/cvpr27) on a NEW branch `cvpr27-paper` of D:/Open3DUAVDet_clean_release and commit it.
# Never pushes, never touches a remote, never changes another branch. Stops if the working tree is not clean.
set -e
SRC=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/repro_1117/release/cvpr27
REPO=/d/Open3DUAVDet_clean_release
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "!! working tree of $REPO is not clean, stop"; exit 1; }
if git rev-parse --verify -q cvpr27-paper >/dev/null; then git checkout cvpr27-paper; else git checkout -b cvpr27-paper; fi
[ "$(git branch --show-current)" = cvpr27-paper ] || { echo "!! not on cvpr27-paper"; exit 1; }
mkdir -p release/cvpr27
cp -r "$SRC"/. release/cvpr27/
git add release/cvpr27
git commit -q -F - <<'MSG'
release/cvpr27: experiment code, configs, advisor patches and rescore of the CVPR'27 PST paper

pipeline/  = gproto/srcx clean chain (run_x.sh + lib) and the cfg yamls it uses
variants/  = every review1113 experiment (run scripts, per-run lib copies, notes, logs)
baselines/ = the five ported baselines (cmpfull), the advisor pack and every patch in order, ADVISOR_RUNS.md
repro/     = MANIFEST.md, manifest.json, rescore_all.py and its report, sha256 lists, ARTIFACTS.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git log --oneline -1
git branch --show-current
