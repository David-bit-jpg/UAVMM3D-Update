# baselines

- `cmpfull/<method>/`: official code of GUPNet, MonoCD, MonoDGP, MonoDETR, MonoCoP with only the data loading replaced and the orientation output extended to a full rotation (`NOTES.md` in each).
  `cmpfull/make_pseudo*.py` builds the pseudo-label tables (`make_pseudo_pcmd_clean.py` = MAV6D +PST, `make_pseudo_pcmd_out.py` = RealDrone3D +PST), `angles.py` the rotation convention, `SPEC.md` the protocol.
- `advisor_pack/00_UAV3D_baselines_pack/` is the pack the advisor ran (scripts/score.py is the scorer of the paper); `advisor_pack/01..11` are the patches in the order applied.
  Large binaries (image caches, GT metas, zips > 20 MB) are not copied; they are listed with size and sha256 in `../repro/ARTIFACTS.md`.
- `ADVISOR_RUNS.md`: run -> Tab. 1 cell, and the patch order.
