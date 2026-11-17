# Supplement experiment scripts (written 11-13, NOT run)

Reported run (PST, clean MAV6D chain): `bash srcx/run_x.sh in mav`. Test UDS **60.81**, val **59.71** (srconly: test 49.51).
All three scripts follow `clean2/cost/w0_commands.sh` and `clean3/nomir_commands.sh`. Each one copies what it needs and redirects its outputs, so nothing under `srcx/in2mav`, `srcx/in2mav_none` or `D:/S2MUAV3D/srcx_work/in2mav` is written.

**Rules shared by all three scripts**
- **Python:** `D:/Miniconda3/envs/city/python.exe` with `PYTHONUTF8=1`.
- **DataLoader:** workers = 0 everywhere.
- **Inference:** goes through `gpu_slot.sh light` with a torch memory fraction of 0.2. Script 1 sets the fraction itself; scripts 2 and 3 use `infer_wrap.py`.
- **CUDA OOM:** the batch is halved and the step retried, up to 4 times. Each retry goes to `logs/retry.log` and the main log. A halved *training* batch is flagged as a recipe deviation.
- **Resume:**
  - Every training step continues from its last checkpoint:
    - refiner: per epoch / every 250 steps;
    - orientation student: every 250 steps;
    - self-trained detectors: `train.py`'s own epoch auto-resume.
  - A step is skipped once its output exists, so re-launching a script is safe.
- **Scoring:** val is scored, then test once. The log ends with `ALL DONE: <metrics>`.
- **Processes:** no script kills any process.
- **Shared code:** `review1113/sup_patch.py` holds the asserted text patches used to make the lib copies.
- **Dry checks done:** `bash -n` on all three scripts, generated both lib copies, `py_compile` on every copy, imported the redirected `sx`, and diffed the copies against `srcx/lib`.

---

## 1. `review1113/sup1/adabn_commands.sh` — AdaBN baseline (inference only)

**What it does.**
1. Takes the clean chain's sim-only detector ensemble: `sx.S['det']` = `output/release_20260920/ramp_s{0,1,2}.pth`, the 3 seeds behind `srconly_{val,test}`.
2. Re-estimates every BatchNorm layer's statistics on all **12,866 unlabeled MAV6D train frames**. This reuses the existing detector AdaBN routine `tools/eval_camnorm.adapt_bn`:
   - running stats are reset, momentum = None (cumulative average);
   - only the BN layers are in train mode;
   - frames are shuffled with seed 0, with no augmentation and no gradient.
3. Predicts val and test exactly as `det_infer_x.py` does: ensemble hook, then flip hook, then `s_bar` self-calibrated on unlabeled val by the same (now adapted) ensemble, then constant-size decoding.
4. Runs `to_pred.py` and `score_x.py` (mav rules).

There is no detector AdaBN in `srcx/lib`. `ori_x.py adabn` exists, but it is for orientation nets. `eval_camnorm.adapt_bn` is the detector version, and the script reuses it.

**Changes vs the reported run.** It is a baseline arm: srconly + AdaBN. It uses no pseudo-labels, teachers or refiners.

**Writes to.** `review1113/sup1/adabn/`:
- `bn/bn_s{k}.pth` (cached adapted BN states, so a restart skips finished members);
- `recs/det0_adabn.pkl`;
- `preds/adabn_{val,test}.pkl` + score JSONs;
- `logs/`.

**Time.** About 20–30 min. The AdaBN pass (3 × 12,866 frames) takes about 8 min. `s_bar` + val + test with ensemble + flip take about 10–12 min. For reference, `det0` inference of train+val+test took 25 min in the reported run.

**Launch.**
```
bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1/adabn_commands.sh
```

---

## 2. `review1113/sup2/nogate_commands.sh` — 10° check off

**What it does.**
1. Retrains the 2 orientation students (seeds 0, 1) on **all 12,866 train crops that have a pseudo-orientation** (`--tgt all`). The reported run uses only the 4,301 with `gate == True` (`--tgt gate`).
2. Runs AdaBN (all train crops).
3. Reads the 2-student ensemble with the test-time mirror view.
4. Deploys with the frozen ω = 0.190.
5. Scores val, then test once.

**Changes vs the reported run.** Only the gate changes. Everything else is unchanged:
- `--mir 12` (mirror loss kept);
- the same `pseudo_tr1.npz`;
- init = same-seed IR+LiDAR member;
- background swap 0.5, 2,250 steps, batch 8+4, bag 0.8.

As in `nomir_commands.sh` (default `FULL=0`), the depth refiners and the constants c_k, r_k, s_bar stay frozen. b_k is re-estimated from unlabeled val by the deploy rule. `FULL=1` additionally retrains the stage-3 refiners on targets built from the new students.

The lib copies are made by `sup2/make_nogate_lib.py`. They use the same transforms as `make_nomir_lib.py`, plus resume states and `infer_wrap.py`.

**Writes to.** `review1113/sup2/nogate/` (`lib/`, `ou/runs/NOGATE_s{0,1}`, `ou/`, `preds/nogate_*`, `logs/`, `work/`). The large crop arrays are read in place from `D:/S2MUAV3D/srcx_work/in2mav` and never written.

**Time.** About 35–40 min with `FULL=0`: the students take about 25 min wall and run concurrently, and AdaBN + readings take about 8 min. With `FULL=1` it takes about 80 min.

**Launch.**
```
bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate_commands.sh
```
(For the full variant: `FULL=1 bash ...`.)

---

## 3. `review1113/sup3/run_seed2.sh` — second clean-chain run with every trained seed shifted

**What it does.** Runs a copy of `run_x.sh` (TEACH=full) on copies of the libs (`sup3/lib`, made by `make_seed_lib.py`). `SX_SEEDSHIFT` (default **+10**) is added to the seed of every component that `run_x.sh` trains:

| Component | Seeds in the reported run | Seeds in this run | How the shift is applied |
|---|---|---|---|
| Depth refiner, stages 1, 2, 3 | 5, 6 | 15, 16 | `torch.manual_seed` and numpy `rng(1000+seed)` |
| Self-trained detectors ×3 | 0, 1, 2 | 10, 11, 12 | `train.py --seed`; init is still the same sim-detector member |
| Orientation students S6I | 0, 1 | 10, 11 | `set_random_seed`, sim-crop dataset, both loader generators, the 80% target bag |

File names keep the nominal seeds, so the glue code is unchanged. Everything downstream is recomputed: pseudo-labels, `det1`, crops, teacher AdaBN, pseudo-orientations, height readings, w, stage-3 targets, constants, deploy, score.

**Kept fixed (decision).** The indoor stage-1 networks are not trained by `run_x.sh`. They are pretrained checkpoints in `sx.SOURCES['in']`: sim-only detectors, IR+LiDAR / IR-distilled / LiDAR teachers, and the height network. Retraining them would take days, so they are kept.

Their outputs are therefore identical inputs. With `COPY_FIXED=1` (the default) they are copied from `srcx/in2mav` instead of recomputed, which saves about 45 min:
- `det0.pkl` and the srconly predictions;
- the refiner's sim samples, bank and real d0 crops;
- `wsim_cache.pkl`.

`COPY_FIXED=0` recomputes them with the same code. Data-construction seeds that are not training seeds also stay at rng 0: the sim-sample selection, the banks and the BN subsets.

The spread therefore covers the seeds of what the chain adapts, not the source-domain pretraining.

**Writes to.**
- `review1113/sup3/run/` (about 1.6 GB; `logs/run_seed2.log`);
- `D:/S2MUAV3D/srcx_work/in2mav_seed2/` (crops, about 9 GB).

It also creates these new files. None of them overwrites anything:
- `tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_in2mav_seed2.yaml`;
- `E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_seed2.pkl` (the self-training dataloader reads its index from the cache dir);
- `output/models/uavdet_3d/camnorm/srcx/st_in2mav_seed2/DET1_s{0,1,2}/`.

**Time.** From the `in2mav` log: the restart at 04:55 to ALL DONE at 08:24 took **3 h 29 min**, with `det0` and the sim samples already done. That is exactly what `COPY_FIXED` skips. The breakdown, in minutes:

| Step | Minutes |
|---|---|
| Refiners st1+st2 | 36 |
| st2 inference + pseudo-labels | 14 |
| Self-trained detectors | 15 |
| `det1` inference | 17 |
| Crops + bank | 13 |
| Teachers | 18 |
| Students | 25 |
| Student AdaBN, readings, height | 18 |
| st2-on-det1 inference + w | 18 |
| st3 | 16 |
| Final inference | 18 |

This gives about **3.5 h** with the GPU to itself, and **4–5 h** with `gpu_slot` queueing. It adds about 45 min if `COPY_FIXED=0`.

**Launch.**
```
bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run_seed2.sh
```
(A different shift: `SX_SEEDSHIFT=20 bash ...` — it uses the same output folder, so use one shift per folder.)
