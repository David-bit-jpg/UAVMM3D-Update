# Sensor-free controls (review1113/sensorfree, built 11-15; queued, not run in full by the author)

Reviewer request: how much of PST's gain needs the privileged simulated sensors (IR, LiDAR)? Scoring is on TEST only, once per run. Each run is
paired with the same-seed PST run (and, for E1 and E4, also with the same-seed RGB-only run). Pooled over the seed pairs with
`heads/heads_boot.py`: for MAV6D, a sequence bootstrap over 10 test sequences; for AntiUAV3D, a moving-block bootstrap with L = 176. No real label
influences any choice.

| arm | what changes vs PST (same seed) | runs |
|---|---|---|
| **E1** fully sensor-free, composites kept | `SX_TEACH=none`, same as the existing RGB-only arm: the RGB-only orientation nets are the teachers and the student init (KZs0/1, OKZrgb_s0/1), the gate is seed agreement, the refiner has no IR-silhouette loss (`SILW=0`), there is one colour class, and height is the constant aspect of s_bar. **In addition, every composite matte is the filled convex hull of the projected 3-D GT box.** In the refiner, composites are switched back on (`SF_COMPOSITE=True`), with alpha = `boxhull/<C>/ref/sim_sil.npy`, which is aligned to the same sim samples (md5 checked). In the students, the cache's `a_v` (IR∩LiDAR) is replaced by `D:/S2MUAV3D/srcx_work/sensorfree/hull_<src>.npy`; the `has_matte` gate is the same, so the composite rate is the same. The whole chain is recomputed. | MAV6D and AntiUAV3D at shifts 0, 10, 20 |
| **E2** box hull instead of the IR silhouette (as target and as matte) | Same as `review1113/boxhull` / heads H7: refiner stages 1–3 are seed-shifted, deployment uses the reference run's frozen w, and det0/det1, the pseudo-labels, the students and the height readings are held fixed | AntiUAV3D shifts 10, 20; pooled with boxhull s0. MAV6D 10/20 is heads H7 (not duplicated), pooled with boxhull s0 |
| **E2a** | Same as E2, but the hull is used **only as the silhouette target**; the composite matte stays the IR silhouette (`ref/sim_matte.npy`) | MAV6D s0 |
| **E2b** | Same as E2, but the hull is used **only as the composite matte**; the silhouette target stays IR | MAV6D s0 |
| **E4** pure self-training | `SX_TEACH=none`, no composites anywhere: the refiner none-arm already has none, and the students run with `--paste 0`; height is constant. Everything upstream of the students is copied or hard-linked from the same-seed RGB-only run, because none of it depends on the students. The students, w, stage 3 and deployment are recomputed. | MAV6D s0, AntiUAV3D s0 |

## Files
- `make_sf_lib.py` builds the patched lib copies, with each anchor asserted.
- `run_sf.sh <arm> <tgt> <shift>` runs one arm (`DRY=1` for a dry run); `run_sf.sh POOL <arm> <tgt>` writes the pooled lines.
- `sf_lane.sh A|B` runs the two lanes. `prep_hull.py` builds the student hull. `plumb_test.py` is the CPU swap check. `sf_wait_then.sh` is the E.done wrapper.
- Logs: `logs/sensorfree.log` (ALL DONE / POOLED / BLOCK lines), per run `<arm>/<tgt>_s<k>/logs/run.log`, and `logs/E.done` when the block is finished.

## Plumbing (11-15, CPU plus one 20-step GPU smoke)
- **Student hull vs a_v.** 96.5 % (in) and 92.6 % (out) of the a_v mass lies inside the hull. The hull area is 2.3× / 1.9× that of a_v.
- **Refiner swap.** With a synthetic readback, composites occur in 43 % of samples:
  - E1: alpha = hull (IoU 1.00).
  - E2a: alpha = IR (0.998), target = hull.
  - E2b: alpha = hull, target = IR.
  - E4: no composites.
- **Student swap.** The alpha actually used equals the hull (IoU 1.000, min 1.000), versus 0.66 / 0.37 against a_v.
- **GPU smoke.** E1 student, 20 steps: 70 composites out of 70 tried.

## 11-15 change (before any E1 run started)
E1 now ALSO supervises the refiner's region head with the box hull (SILW = 1.0 in the E1 lib; ref/sim_sil.npy = hull array), in addition to
the hull composites in refiner and students. Reason: reviewers asked for a control that keeps every mechanism of PST (region head, composites,
self-training) but uses no sensor information, so that PST − E1 isolates what the IR/LiDAR content adds. Brightness class stays single (as the
none arm). Old libs kept as E1/*/lib_old_silw0 (never run).
