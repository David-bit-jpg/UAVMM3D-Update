# Review scripts B3 / B4 / B8 (written 11-14, NOT run)

All three are for the scheduler. Each writes only under its own folder (`review1113/B3`, `B4`, `B8`) plus the new, tagged names listed below; existing run dirs are only read. Shared robustness:
- Every training step resumes from its checkpoint, and every step is skipped once its output exists, so a re-launch continues where it stopped.
- On CUDA OOM the batch is halved and the step retried, up to 4 times. Retries go to `logs/retry.log`; a halved training batch is flagged as a RECIPE DEVIATION.
- DataLoader workers are 0 everywhere.
- Inference goes through `gpu_slot.sh light` + `infer_wrap.py` (memory fraction 0.2).
- At most 2 trainings run concurrently, and no script kills processes.
- The last log line is `ALL DONE: test … | val …`.

Checks done: `bash -n` on all three; lib generation + `py_compile` for all libs; config/isolation printout (no target name existed before). B3 GPU smoke ran in the scratchpad only, with nothing under `B3/feat|both`: calibration on 3, 5 and 20 batches, then `feat` 12 steps, `both` 3 steps; peak 3.8 GB. B4: CPU dry-run of the naive pseudo-label rule.

---

## B3 — transfer route at the same insertion point (`B3/route_commands.sh`)

**What changes.** Only how the knowledge of the frozen teachers reaches the orientation students. The frozen teachers, the real crops (gated pool, 80 % bag per seed, same augmentation), the init (same-seed IR+LiDAR member), L_src with background swap 0.5, mirror loss 12, 2,250 steps, bs 8+4 and seeds 0/1 are all unchanged.

| Arm | Real-crop loss | Source |
|---|---|---|
| hard | L_pl^ori (hm + 2·centre_res + 12·sym. kp2d on the pseudo box) + 12·L_mir | **reused**: `srcx/in2mav` (test 60.81 / val 59.71) |
| feat | w_feat·L_feat + w_soft·(L_hm + 12·L_kp) + 12·L_mir; **no hard pseudo label in any loss** | `B3/feat` |
| both | L_pl^ori + 12·L_mir + w_feat·L_feat | `B3/both` |

**Teachers.** `srcx/in2mav/ou/T_{irlid,kdir,lid}{0,1}.pth`. These are the same AdaBN'd teacher states that produced `pseudo_tr1`. They run frozen in eval mode on the student's own augmented crop.

**Read cell (r\*, c\*).** The argmax of the teachers' mean sigmoid heatmap within the 5×5 cells around the crop's detection cell. This is the deployment rule of `ori_x read`. The detection centre comes from `oc_tr1.pkl` and is mirrored when the crop is h-flipped. A global argmax gave single-batch outliers (L_kp 15 against 0.1) and was replaced.

**Feature tap.** `batch['features_2d']`, the 512-channel map (36×64) that is the shared input of every CenterHead branch, i.e. exactly the input of the kp2d head. L_feat is averaged over the 5×5 cells around (r\*, c\*), which is the kp2d head's receptive field: two 3×3 convs.

**L_feat (default `FEAT_TARGET=adapt`).** mean_t [1 − cos(A_t f_s, f_t)], where A_t is one 1×1 conv per teacher, identity-initialised, Adam 1e-3, discarded after training.
- **Deviation from the brief.** The brief asked for the cosine to the teachers' *mean* feature. The smoke measured the six independently trained teachers' features as near-orthogonal: pairwise cosine 0.09–0.15 on the read window, kdir0/kdir1 0.60. Their mean is therefore no coherent target and would make the feat arm a straw man.
- The literal version is kept as `FEAT_TARGET=mean` and writes to `B3/mean/`.

**Soft terms.**
- L_hm = Σ Bernoulli-KL(mean_t σ(h_t) ‖ σ(h_s)) / Σ mean_t σ(h_t).
- L_kp = mean_t min_q ‖k_s − Π_q k_t‖₁/16 at (r\*, c\*). Π_q is the corner renumbering of R·Rz(90q); it was verified exact for l = w (s̄: l = w = 0.347).
- In feat the mirror cell is (r\*, c\*); in both it stays the pseudo-label cell.

**Weights.** Fixed once at init by step 1 (`calib/calib.json`): 50 real batches of the seed-0 loader, student = IR+LiDAR member 0, train-mode BN, no gradient.
- w_feat = mean L_pl^ori / mean L_feat.
- w_soft = mean L_pl^ori / mean L_soft.
- So each KD term has the magnitude of L_pl^ori at init. The same weights are used for every arm and seed. Nothing is tuned on val/test.
- Smoke (20 batches): L_pl 2.19 (the hard run logged 2.13 at step 150), L_feat 0.757, L_soft 2.19, giving **w_feat ≈ 2.89, w_soft ≈ 1.00**.

**Deploy/score.** As `nomir_commands.sh`: AdaBN, 2-student readings with the mirror view, deploy at the frozen ω 0.190, score val then test once. With the default `FULL=0` the depth refiners are frozen; `FULL=1` retrains stage 3.

**Estimate.** About 2 h alone on the GPU, 2.5–3 h with queueing. The feat step is about 1.0 s against 0.63 s for hard; 2 pairs of students take about 1 h 40 min.

```
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/route_commands.sh
# literal mean-feature variant (optional, separate outputs):  FEAT_TARGET=mean bash .../B3/route_commands.sh
```

## B4 — naive self-training baseline (`B4/naive_st_commands.sh`)

**What changes.** This is Step 1 of the chain with pseudo-labels = the sim-only ensemble's own boxes.
- The frames are those with conf ≥ 0.5: the same 5,227 frames as PST's labels.
- The box is (t_det, s̄, R_det), i.e. its own depth, with no IR-refiner correction. The CPU dry-run confirmed identical frames/size/rotation and depth only differing.
- The recipe is otherwise unchanged: 3 detectors, seeds 0/1/2.

**Row (b).** `recs/det1.pkl` as is, scored against PST's row (b) of 53.38 test / 51.29 val. It is logged as `ROW_B DONE`.

**`REST=1` (default).** The rest of the PST chain runs on top of the naive detector: crops, teachers, students, height, stage 3, deploy.
- Stage-3 targets keep PST's IR-corrected Module-3 labels (`index_pseudo_srcx_in2mav`, read only), so only step 1 differs.
- Refiner st1/st2 are copied from in2mav, because there is no seed shift and retraining them would repeat the identical recipe.

**New names.**
- `tools/cfgs/.../srcx/{real,st}_in2mav_naive.yaml`
- `E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_naive.pkl`
- `output/models/.../srcx/st_in2mav_naive/`
- `D:/S2MUAV3D/srcx_work/in2mav_naive/` (about 9 GB)

**Estimate.** Row (b) arrives after about 45 min. The full run takes about 3 h alone and 3.5–4 h with queueing.

```
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/naive_st_commands.sh      # REST=0 ... for row (b) only
```

## B8 — IR-only fusion control on seed runs 2 and 3 (`B8/w0_seeds_commands.sh`)

**Recipe.** This is the `clean2/cost/w0_commands.sh` recipe, applied per seed run:
- Stage-3 retrain with ω = 0, i.e. targets log(z_pl/z_det0), from the run's own stage-2 weights.
- Seeds 5/6 + SX_SEEDSHIFT, i.e. 15/16 and 25/26.
- Then inference, deploy at the run's own frozen ω, and scoring.

**Libs.** Each lib is copied from the seed run's own lib, so the PSEUDO_TAG, seed shift and resume logic match; only sx.O/W are redirected to `B8/<run>`.

**Runs.**
- `s10` = `sup3/run`, PST 59.28 / 59.54.
- `s20` = `seeds/mav_full_s20`, PST 62.70 / 59.87.

**Summary line.** The final line reports fused − IR-only per run and its mean over 3 runs, including the reported pair 60.81 vs 59.37 = +1.44.

**Estimate.** About 36 min per run: about 1 h 15 min alone, 1.5–2 h with queueing.

```
bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/w0_seeds_commands.sh
```
