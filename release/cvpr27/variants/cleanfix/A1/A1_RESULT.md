# cleanfix A1 — window contamination of the clean banks + halo mask check (11-14, CPU)

Real MAV6D **train** labels (E:/mmcache/mav6d_cn_obj/train/index.pkl boxes9d) used ONLY for this post-hoc measurement; nothing is selected with them.

## 1. Contamination (paper `3_method.tex:64`, `X_suppl.tex:88`)

Script `a1_contam.py` → `a1_contam.json`. Replica of the dev scripts `gproto/wf24/diag_randwin.py` (random windows) and `gproto/wf21/strict/diag_bank.py`
(bank replay): the clean selection rules of `srcx/lib/refiner_x.py cmd_bank` (det0, conf ≥ .5, width 6–170 input px, √3·w square moved 1.8 w in
one of 8 directions, seed 0) and `srcx/lib/ori_x.py cmd_bank` (det1, conf ≥ .6, window (1920/z)×(1080/z), z = clip(95/w2d,1,4), uniform position
avoiding the 20 %-enlarged detected box, seed 0) are replayed with the same RNG sequence, no image read; replayed counts match the saved banks
exactly (asserted: 4,976 = `srcx/in2mav/ref/bank.npy`, 4,420 = `D:/S2MUAV3D/srcx_work/in2mav/ou_bank.npy`). Contamination of a window = max over GT
UAVs of the share of the GT 2D box (8 projected corners, raw K) inside the window. Random baseline = same frames, same window size, uniform position
in the frame (no avoidance), seed 0 (as diag_randwin.py).

| bank (clean in2mav, MAV6D train frames) | n windows | ≥10 % of a UAV | ≥50 % | Mavic 2 / Phantom 4 (≥10 %) |
|---|---|---|---|---|
| depth-refiner bank (used) | 4,976 | **0.04 %** (2) | 0.00 % | 0.00 % (n 394) / 0.04 % (n 4,582) |
| depth-refiner, random windows | 5,225 | **21.5 %** | 11.1 % | 37.3 % / 20.1 % |
| orientation bank (used) | 4,420 | **0.09 %** (4) | 0.00 % | 0.25 % (n 803) / 0.06 % (n 3,617) |
| orientation, random windows | 6,873 | **69.2 %** | 56.3 % | 79.2 % / 65.3 % |

Dev values being replaced: random 20.6 % / 71.9 %; banks 0.10 % (refiner) and 0.7–1.8 % (orientation banks, wf24 README l.13).

Paper mapping:
- `3_method.tex:64` "at most 1.8 % … against 20.6–71.9 % for random windows … on development banks" → "**at most 0.09 %** of windows contain ≥10 % of it, against **21.5–69.2 %** for random windows (checked afterwards with training labels on the deployed banks)".
- `X_suppl.tex:88` "0.10 % … measured on the development bank" → "**0.04 %** contain ≥10 % of a real UAV (2 of 4,976; measured afterwards with training labels)".
- AntiUAV3D (out2antifs banks) not measured: the original measurement was MAV6D-only and AntiUAV3D train labels are sealed.

## 2. Halo / mask code (`X_suppl.tex:276`, "a median of 1.3 % of silhouette pixels")

- Dev 1.3 % = median `halo_bg` in `wf20/design/matte_stats_test.json` (`wf20/design/prep_sim.py`, 552 simulated TEST crops; p90 4.9 %, >10 %: 2 %).
- Clean chain: orientation composites (`srcx/lib/ou_train_x.py` → `wf20/design/kzpaste_train.CacheDS`, MATTE='v') read the precomputed alphas
  `a_v.npy` (IR⇄LiDAR) of `sx.S[src]['sim_crops']` = `…/26739ba4…/scratchpad/kz/crops_6000` (in) and `D:/S2MUAV3D/pcmd_out_work/crops_5367` (out),
  written by `wf20/design/extract_kz_crops.py` / `outdoor/extract_kz_crops_out.py` with `kzpaste_train.matte()`. That function implements the
  **same rules** as prep_sim.py (ring MAD threshold, adaptive LiDAR support r = ⌈√(area/n)⌉∈[1,8] + closing, colour test >20, halo pixels nearer the ring colour removed, σ = 1 feathering)
  but it is a **separate code copy on a different crop set** with slightly different minimum counts (≥15 LiDAR px / silhouette ≥20 px vs 30 / 30;
  IR from the uint8 sidecar). → **Not the same code/crops; recomputed** on the clean crops (`a1_halo.py` → `a1_halo.json`; removed = |M_ir \ M_v| / |M_ir|, M = alpha ≥ 128):

| sim crops used by | n (with matte) | median removed halo | mean | p90 | crops > 10 % removed |
|---|---|---|---|---|---|
| MAV6D chain (in, crops_6000) | 5,911 / 6,000 | **0.71 %** | 1.52 % | 3.80 % | 1.5 % |
| AntiUAV3D chain (out, crops_5367) | 5,196 / 5,367 | **3.11 %** | 3.41 % | 7.30 % | 2.8 % |

Paper mapping `X_suppl.tex:276`: "The LiDAR veto removes IR halo, a median of 1.3 % of silhouette pixels" → "a median of **0.7 %** of silhouette pixels
on the simulated MAV6D-chain crops (3.1 % on the outdoor ones; >10 % for 1.5 % / 2.8 % of crops)".
- Note: the LiDAR veto concerns only the **orientation** composites. The depth-refiner composites (`refiner_x.py sim_batch`) use the IR silhouette
  `softkd_extract.sil_full` (GT-hull-gated IR threshold, no LiDAR veto).
- Threshold-at-128 recovery of the binary mask from the feathered uint8 alpha may miss/add single boundary pixels; effect on a median of 0.7 % is small but nonzero.
