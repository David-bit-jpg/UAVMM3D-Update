# Per-head supervision ablations (review1113/heads, written 11-14, NOT run by the author)

**Goal.** The supplement should state what each teacher-supervision head contributes, with evidence. Today the paper can only claim that
teacher supervision as a whole helps: RGB-only teachers lose 2.85 UDS on MAV6D test and 6.87 on AntiUAV3D test, 3 runs each.

**Design.** Each arm changes ONE factor relative to the same-seed reported PST run. Everything else is identical: code (`srcx/lib` +
`review1113/sup_patch.py`), seeds, recipes and constants.

**Scoring.**
- MAV6D TEST only, scored once per run.
- No val or calibration-image labels are read: the val split is never scored.
- Paired sequence bootstrap over the 10 test sequences, B = 2000, rng 0. `heads_boot.py` reproduces `cpu5.json` pair 1 exactly
  (+3.43 [+1.92, +5.51]).
- Pooled over the seed pairs once more runs exist.

## Arms

| arm | what changes (single factor) | rerun | reused (read only) |
|---|---|---|---|
| **H4** LiDAR aux heads | Teacher group: the 2 LiDAR nets `kzl_s0/1` (KZ-LiD) are replaced by the same-seed plain KZ nets (`sim_full_r34_kp_w12_zoom/KZs0/1`, ep 30). These have the same KZ recipe with no auxiliary head. They also take the LiDAR slot of the IR-vs-LiDAR gate. Student init stays `kzm`. | AdaBN + reading of the 2 new nets; pseudo-orientations; students S6I ×2; AdaBN; mirror readings val/test; deploy; test score | det1, crops / bank (hard links), the other 4 teachers' readings (**bit-identical**, merged by `merge_reads.py`), frozen stage-3 depth students, height readings, the run's w |
| **H5** IR intensity channel | The 2 IR+LiDAR nets `kzm_s0/1` (KZ-IRLiD) are replaced by the same-seed KZ-LiD nets. These are `kzl_s0/1`: the KZ-IRLiD recipe without the IR target, `IR_AUX` off. The replacement applies both as teacher and as student init. No new net is needed: both seeds exist. | Same as H4, but no GPU is needed for the teachers: the replacement readings are exact aliases of the reference `T_lid0/1` readings (same checkpoint, same AdaBN, same crops) | Same as H4 |
| **H6** height target | Deployment height ratio comes from the height network's own **box-corner** keypoints (kp2d head, box-supervised): decode → corner PnP with s_bar → least-squares refit of (R, t, h) with l, w = s_bar → r = h / mean(l, w). The reported ratio uses its **LiDAR structure points**. The network (KZ-KPS s0 = KZ-IRLiD + kps head), the BN re-estimation and the read cell are all the same (`kpsbox_x.py`). r_k does not enter the stage-3 targets (`chain_x targets` never reads kps), so only deploy changes. | kpsbox readings tr1/val1/te1; deploy; score | everything else |
| **H1c** (diagnostic for H1) | Deploy-only: the reference run deployed with one colour class (see H1). | deploy (CPU); score | everything |
| **H2** sil_feats → distance head | Refiner `Net.fwd`: the 3 predicted-silhouette features are replaced by zeros. Zero inputs give zero weight gradient, so the features are effectively dropped. Silhouette loss, composites and colour classes are unchanged. | **Whole downstream chain**: refiner st1–2 (seeds 5, 6) → pseudo-labels → 3 self-trained detectors → det1 → crops, 6 teachers, gate, students → height readings → w (sim val) → stage-3 targets → stage 3 → deploy | fixed stage-1 outputs of the pretrained source nets from `srcx/<C>`: det0, refiner sim samples, real bank, real d0 crops (large `.npy` as hard links), wsim cache |
| **H1** silhouette loss | `SILW = 0` in all refiner stages: the sim BCE+Dice term and the stage-2/3 silhouette-consistency terms. Composites stay ON (the IR silhouette is still the alpha matte). See the H1 notes below. | whole downstream chain | as H2 |
| **H3** silhouette composites | Refiner `sim_batch`: no IR-matte composites. The rng draw that decides a composite is kept. Silhouette loss, sil_feats and colour classes are unchanged. Student composites are not touched: that is `cleanfix_B4`. | whole downstream chain | as H2 |
| **H7** silhouette vs box hull | The existing `review1113/boxhull` control, run at seed shifts 10 and 20, paired with `sup3/run` and `seeds/mav_full_s20`. This gives 3 runs together with `boxhull/in2mav`. It uses the same hull array (`boxhull/in2mav/ref/sim_sil.npy`; the sim sample list is md5-identical in all three runs). The refiner stages 1–3 are seed-shifted. | refiner st1–3 + inference + deploy at the reference run's frozen w | det0/det1, pseudo-labels (reference tag), students, height readings, w (= boxhull protocol) |

**H1 notes.**
- **Colour classes.** The colour class is the brightness inside the IR-supervised predicted silhouette. Without the loss that silhouette
  no longer exists, so H1 uses one colour class. This is the rule the RGB-only / LiDAR-only controls already apply (`sx.NOIR`).
- **What H1c separates.** In the reported MAV6D run, r_k is 0.38 for dark drones and 0.70 for white ones, while one class gives 0.45. The
  colour class therefore matters, and H1c isolates it as a deploy-only diagnostic.
- **sil_feats in H1.** The distance head still gets sil_feats of the now-unsupervised logits, which become a free learned feature path.
  H2 is the arm that removes that path.

**Why each scope.**
- **H1–H3 (full downstream).** The refiner feeds the step-1 pseudo boxes, and through them the self-trained detectors, crops, students, w,
  the fused targets and deployment. Only a full rerun is end-to-end faithful. This is unlike boxhull, which held det1 and the students fixed.
- **H4/H5 (initabl protocol).** They follow the user's scope "teacher readings + students + deploy" and the `review1113/initabl`
  protocol: frozen w and frozen stage-3 depth students. The student→stage-3 target path is therefore frozen, and only the student readings
  used at deployment change (rotation + corner-PnP depth in the fusion).

**Caveat for H5.** The same-seed no-IR twin of `kzm_sX` is exactly `kzl_sX`. The latter is already in the group as a LiDAR member, so the
H5 ensemble contains each `kzl` twice. That is what replacing the IR+LiDAR nets means here. Training new KZ-LiD seeds would break "same
seed".

**H6 plumbing check (200 crops).** I ran the box-corner reader on 200 crops (no labels involved):

| split | box-corner ratio (median) | LiDAR structure-point ratio (median) | Spearman |
|---|---|---|---|
| tr1 | 0.381 | 0.391 | 0.45 |
| te1 | 0.354 | 0.369 | 0.53 |

## Files
- **`make_heads_lib.py <arm> <tgt> <shift>`** builds the isolated lib copies, with each patch asserted on its anchor:
  - outputs → `heads/<arm>/<tgt>_s<shift>`;
  - crops → `D:/S2MUAV3D/srcx_work/heads_<arm>_<C>_s<shift>`;
  - new config / pseudo / detector names `*_hd<arm>s<shift>`.
- **`run_arm.sh <arm> <tgt> <shift>`** is the driver. Its FULL part equals `seeds/run_seed_chain.sh` with teach = full. It handles:
  - the gates, `DEPS`, a deadline guard, OOM halving, resume and the markers `.done` / `.failed` / `.skipped`;
  - its final line: `ALL DONE: <arm> test UDS/AP/ATE/ASE/AOE/mDR | ref | paired diff [95% CI] || pooled over finished seed pairs`.
- **`heads_boot.py`** computes the metrics and the paired bootstrap. MAV6D uses sequences; AntiUAV3D uses blocks within sequences with
  L = 176, as `cpu5` does.
- **`kpsbox_x.py`** is the H6 reader. **`merge_reads.py`** is the H4/H5 teacher-reading merge.
- **`heads_seeds.sh`** is the follow-up line. **`heads_anti.sh`** is the AntiUAV3D line.
- Logs: `heads/logs/heads.log` (one line per event), per run `<run>/logs/run.log`, results `<run>/boot_test.json`, `heads/<arm>/pooled_<tgt>.json`.

## Queue (appended at the END of `local_queue_1113.txt`)

Every script first waits for `review1113/initabl/.done` AND `D:/cmpfull_runs/gupnet_uda_mtstrict_s0/.done`, polling every 10 min. The
lines then run one after another via `DEPS`:

1. `heads_H4` — H4 mav s0
2. `heads_H6` — H6 mav s0, then H1c mav s0 (in parallel with H4: inference only)
3. `heads_H5` — after H4
4. `heads_H2` — after H5, H6, H1c
5. `heads_H1` — after H2
6. `heads_H3` — after H1
7. `heads_H7` — after H3: shift 10, then shift 20
8. `heads_seeds` — after ALL single runs: shifts 10 and 20 of H4, H6, H1c, H5, H2, H1, H3 (cheap arms first)
9. `heads_anti` — after `heads_seeds`: AntiUAV3D s0 of H2, H1, H3

**Deadline guard.** A run that has not started does not start after machine time 11-22 18:00 (full-chain arms) or 11-23 08:00 (the
others). It is marked `.skipped` and the queue moves on.

## GPU-time estimate

Estimates are from the logs of the clean runs, with the GPU alone:
- MAV6D full chain: 2 h 47 min (`seeds/mav_full_s20`) to 3 h 29 min (`srcx/in2mav`).
- AntiUAV3D full chain: 1 h 48 min (`seeds/antifs_full_s10`).
- Orientation branch: about 45–75 min.

| block | runs | per run | total, GPU alone |
|---|---|---|---|
| H4 / H5 / H6 / H1c (s0) | 4 | 45 / 35 / 15 / 5 min (CPU) | ~1.6 h (H6 overlaps H4) |
| H2, H1, H3 (s0, full chain) | 3 | ~3 h | ~9 h |
| H7 (s10, s20) | 2 | ~1.5 h | ~3 h |
| **single runs** | | | **~13.5 h** |
| follow-up: H4–H6 + H1c + H1–H3 at s10, s20 | 14 | as above | ~21.5 h |
| AntiUAV3D H2, H1, H3 (s0) | 3 | ~2 h | ~6 h |
| **total** | | | **~41 h** |

With gpu_slot contention, expect about ×1.5 (the boxhull and seed runs took 1.3–2.5× their GPU-alone time), i.e. **~60 h**.

**Disk.**
- E: about 2.7 GB per full-chain run: self-trained detectors about 1 GB plus refiner / ou outputs. That is about 32 GB for 12 runs,
  against 131 GB free on E: as of 11-14.
- D: about 11 GB of crops per MAV6D full-chain run and 3.4 GB per AntiUAV3D run, about 110 GB in total (1.7 TB free).
- ORI / HGT crops are hard links and cost 0 bytes.

## Expected finish order

This assumes the gate opens around 11-15 08:00. The first time is GPU alone; the second (→) is with contention.
- H6 + H1c ~08:20 → 08:30
- H4 ~08:45 → 09:10
- H5 ~09:25 → 10:00
- H2 ~12:30 → 14:30
- H1 ~15:30 → 19:00
- H3 ~18:30 → 23:30
- H7 s10 ~20:00 → 11-16 02:00
- H7 s20 ~21:30 → 11-16 04:30. **All single runs end about 11-15 22:00, or 11-16 05:00 with contention.**
- Seed follow-ups end about 11-16 19:00, or 11-17 14:00 with contention. The H4–H6 seed pairs finish within the first ~3 h of that line.
- AntiUAV3D H1–H3 end about 11-17 01:00, or 11-17 23:00 with contention.

Everything ends before the 11-23 cut-off, even at ×2.

## Not done / open
- **No null rerun.** There is no unchanged-chain rerun at shift 0 to measure same-seed GPU non-determinism; one would cost ~3 h and can
  be added as the same driver with an empty patch. The pooled 3-seed pairs carry the training noise instead.
- **H4/H5 seed scope.** They do not re-derive w or stage 3 (initabl protocol). If a faithful student→stage-3 path is wanted, the FULL
  block of `run_arm.sh` from "height readings, w, stage 3" can be enabled for them.
- **AntiUAV3D scope.** Only H1–H3 run there. The ORI / HGT arms are MAV6D only, as the brief asks.
