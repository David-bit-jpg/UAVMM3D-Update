# A11: size prior. What a perfect size would give, how sensitive distance is to size bias, and the cancelling AntiUAV3D biases (2026-11-14)

CPU only. Only existing predictions were used; nothing was trained and nothing was killed. GT is used for scoring and for the **oracle** corrections below. These corrections are diagnostics: none of them can be deployed, and none was used to choose anything.

**Files.**
- Script: `panelA/a11/a11_size.py`, which writes `a11_size.json` and `a11_size.log`.
- Checks: `a11_decodechk.py`, `a11_rotdiag*.py`.
- Figure: `panelA/A11_size_sweep.{png,pdf}`.

**Arms** (test only):
- MAV6D, 4,800 frames, rules `real_test`:
  - PST = `in2mav/pcmd_fused_test`
  - RGB-only = `in2mav_none/pcmd_fused_test`
  - Sim-only = `in2mav/srconly_test`
- AntiUAV3D v2, 2,624 frames, rules `outdoor_test`: the `out2antifs*` equivalents.

**Scoring.** All rows are re-scored with the pack `score.py::metrics`. The deployed rows reproduce 60.81 / 57.38 / 49.51 and 41.73 / 38.92 / 30.55.

## 0. How distance is decoded (verified)
The decoder computes z_det = f_x · span_x(s̄, R_det) / ŵ_2D. Here:
- span_x is the horizontal 8-corner projection at input-pixel K_in;
- s̄ is the size the detector self-calibrated on unlabeled val images (det1 for PST and RGB-only, det0 for sim-only);
- R_det is the detector's rotation.

The identity holds exactly: log span(s̄, R_det, t_det) / ŵ_2D = 0.0000 (p5–p95 ±0.0000) for every arm on both sets. PST and RGB-only then rescale z_det by the refiner and the fusion.

Per frame, at the GT centre, the log distance error of the detector splits into **three** terms, not two:

| Term | Definition |
|---|---|
| **size** | log span(s̄, R_det) / span(s_gt, R_det). This is the pure size error. |
| **rotation footprint** | log span(s_gt, R_det) / span(s_gt, R_gt) |
| **width read** | w2d = log span(GT box) / ŵ_2D. A negative value means the width is over-read. |

C1_REPORT's "size +0.22" is size + rotation footprint (`size_c1`).

**Medians on the deployed TP frames**

| Set | Arm | size (pure) | rotation footprint | size_c1 (= C1 "size") | width read w2d | final error |
|---|---|---|---|---|---|---|
| AntiUAV3D, M4E | PST | **+0.541** | −0.31 | +0.225 | **−0.302** | −0.054 |
| | RGB-only | +0.541 | −0.32 | +0.223 | −0.323 | −0.083 |
| | Sim-only | +0.542 | −0.31 | +0.227 | −0.282 | −0.057 |
| AntiUAV3D, M400 (47 TP) | PST | −0.610 | −0.12 | −0.734 | −0.238 | −0.835 |
| MAV6D, all | PST | +0.023 | −0.12 | −0.100 | +0.181 | +0.046 |
| | RGB-only | +0.022 | −0.12 | −0.101 | +0.222 | +0.052 |
| | Sim-only | +0.021 | −0.13 | −0.109 | +0.087 | −0.033 |
| MAV6D, Mavic 2 | PST | +0.209 | −0.25 | −0.045 | +0.133 | +0.074 |

On AntiUAV3D M4E, s̄ is **×1.72 too large** (+0.54 log). Two errors of about −0.30 each offset it:
- the width head over-reads ŵ by ×1.35;
- the detector's rotation gives a body footprint ×0.73 narrower than the GT pose.

The residual is −0.05 to −0.08. On MAV6D, s̄ ≈ the Phantom 4 box, so the pure size error is about 0. There the width is *under*-read and the rotation footprint partly offsets it. The rotation-footprint term is my addition; C1 folded it into "size".

## 1. Perfect size: GT size in the decoder, same apparent width and rotation (z · e^(−size))
| Set | Arm | as deployed: UDS / AP / ATE / med log / med \|rel\| | perfect size: UDS / AP / ATE / med log / med \|rel\| |
|---|---|---|---|---|
| MAV6D | PST | 60.81 / 40.93 / 0.306 m / +0.046 / 6.2 % | **61.83** / 42.48 / 0.286 m / −0.003 / 6.0 % |
| | RGB-only | 57.38 / 38.04 / 0.348 / +0.052 / 6.8 % | 59.40 / 40.98 / 0.304 / −0.000 / 6.4 % |
| | Sim-only | 49.51 / 26.87 / 0.283 / −0.033 / 6.8 % | 48.37 / 25.14 / 0.306 / −0.058 / 7.4 % |
| AntiUAV3D | PST | 41.73 / 27.89 / 9.9 % / −0.056 / 7.8 % | **19.99** / 0.00 / 43.4 % / −0.590 / 44.6 % |
| | RGB-only | 38.92 / 23.39 / 11.1 % / −0.086 / 9.7 % | 19.80 / 0.00 / 44.9 % / −0.617 / 46.0 % |
| | Sim-only | 30.55 / 16.04 / 13.2 % / −0.062 / 11.5 % | 13.32 / 0.00 / 40.8 % / −0.532 / 41.2 % |

**MAV6D.**
- A perfect size is worth about +1 UDS for PST and +2 for RGB-only.
- The PST − RGB-only gap shrinks from +3.43 to +2.43 overall. On Phantom 4 it grows from +2.52 to +3.14. On Mavic 2 it shrinks from +6.63 to +1.36: s̄ is 23 % too big for Mavic 2, which offsets its width under-read.

**AntiUAV3D.** A perfect size alone would be **catastrophic**: distance falls ×0.55 short and AP drops to 0. The constant oversized s̄ is what compensates the two −0.30 errors.

## 2. Size-bias sweep (decoder size = GT · (1 + b); positions scaled along the ray, predicted lwh kept), all frames
Each cell is UDS / median |rel| distance error, with the median taken on the variant's own TP frames. The figure is `A11_size_sweep.pdf`.

| b | MAV6D PST | MAV6D RGB-only | MAV6D Sim-only | AntiUAV3D PST | AntiUAV3D RGB-only | AntiUAV3D Sim-only |
|---|---|---|---|---|---|---|
| −40 % | 31.67 / 41.4 % | 30.16 / 41.2 % | 27.12 / 44.4 % | 8.32 / 54.2 % | 7.44 / 51.1 % | 7.19 / 59.0 % |
| −20 % | 40.09 / 21.2 % | 38.50 / 21.0 % | 29.17 / 25.3 % | 14.48 / 53.4 % | 14.01 / 54.6 % | 8.41 / 47.2 % |
| −10 % | 53.82 / 11.0 % | 51.60 / 11.0 % | 37.54 / 15.6 % | 18.74 / 49.5 % | 18.14 / 51.0 % | 10.74 / 44.0 % |
| 0 | 61.83 / 6.0 % | 59.40 / 6.4 % | 48.37 / 7.4 % | 19.99 / 44.6 % | 19.80 / 46.0 % | 13.32 / 41.2 % |
| +10 % | 49.02 / 11.6 % | 47.40 / 11.8 % | 48.56 / 7.2 % | 20.23 / 39.1 % | 20.22 / 40.8 % | 15.37 / 37.6 % |
| +20 % | 37.96 / 21.2 % | 36.10 / 21.5 % | 38.44 / 14.6 % | 20.26 / 33.6 % | 20.28 / 35.5 % | 16.16 / 33.1 % |
| +40 % | 31.70 / 42.9 % | 30.22 / 43.4 % | 27.21 / 34.9 % | 22.66 / 22.6 % | 21.00 / 24.8 % | 19.51 / 22.7 % |
| *as deployed* | *60.81 / 6.2 %* | *57.38 / 6.8 %* | *49.51 / 6.8 %* | *41.73 / 7.8 %* | *38.92 / 9.7 %* | *30.55 / 11.5 %* |

**Reading.**
- Distance error moves one-for-one with size error: ±10 % size gives about ±0.10 log distance. On MAV6D, ±10 % costs 8–13 UDS.
- PST stays ahead of RGB-only at every bias on MAV6D (+1.5 to +2.4 UDS). On AntiUAV3D, every swept size is far worse than deployed, because the width and rotation errors are no longer cancelled.
- The AntiUAV3D −40 % point is not monotone. At that bias most frames fall out of the TP set, and the error is computed on the variant's own TPs.

## 3. Removing the two cancelling AntiUAV3D biases (oracle)
Columns are UDS for PST / RGB-only / Sim-only, then the PST − RGB-only and PST − Sim-only gaps.
- "Size" = size_c1, as in C1, so it includes the rotation footprint.
- "Width" = each arm's global median w2d on its deployed TP frames.

| Correction | all 2,624 frames: P / R / S | P−R | P−S | M4E only (2,576): P / R / S | P−R | P−S |
|---|---|---|---|---|---|---|
| as deployed | 41.73 / 38.92 / 30.55 | **+2.81** | **+11.18** | 42.89 / 40.04 / 31.65 | +2.85 | +11.24 |
| remove size bias only (per frame) | 21.84 / 20.58 / 17.94 | +1.26 | +3.90 | 21.66 / 20.51 / 17.90 | +1.14 | +3.76 |
| remove width over-read only (global) | 20.54 / 20.85 / 17.46 | −0.31 | +3.08 | 20.60 / 20.92 / 17.47 | −0.32 | +3.13 |
| **remove both: per-frame size + global width** | 41.65 / 40.64 / 28.58 | **+1.01** | **+13.07** | 42.49 / 41.61 / 28.15 | +0.88 | +14.34 |
| remove both: global size + global width | 43.42 / 43.59 / 30.23 | −0.18 | +13.19 | 45.00 / 44.87 / 31.28 | +0.13 | +13.72 |
| ref.: remove each arm's own global distance bias | 44.25 / 43.86 / 30.13 | +0.39 | +14.12 | 45.77 / 45.24 / 31.14 | +0.53 | +14.63 |

**Median |rel| distance error after "both removed" (per frame / global).**
- PST 7.0 / 6.5 %
- RGB-only 7.3 / 6.5 %
- Sim-only 11.3 / 11.0 %

**Reading for the paper.**
- Removing either bias alone halves UDS. Removing both restores the deployed level.
- **PST − Sim-only survives and grows:** +11.2 → +13.1 to +14.1 UDS. The per-frame spread drops from 11 % to 6.5–7 %, and this is the refiner's per-frame correction.
- **PST − RGB-only on AntiUAV3D largely vanishes:** +2.81 → +1.01 (per-frame size), −0.18 (global), +0.39 (own bias). RGB-only over-reads width slightly more (−0.32 vs −0.30) and ends at −8.6 % vs −5.6 % final bias. So roughly two-thirds to all of the AntiUAV3D privileged-teacher gain is a global scale offset, not per-frame accuracy. Without labels the global scale is not identified (paper Sec. "Global scale").
- The same oracle on MAV6D keeps the gap: own-global-bias removal gives PST − RGB-only **+3.38** (deployed +3.43) and PST − Sim-only +14.92. On MAV6D the privileged gain is per-frame.

**M400 (48 frames).** Here s̄ is ×0.54 too small. "Remove size per frame" lifts PST from 19.3 to 43.5 UDS, and RGB-only to 51.9. Any global correction leaves it at about 19 (AP 0). Report it separately, never pooled.

## Caveats
- All corrections use test GT (oracle) and are diagnostics. The global medians are each arm's own, taken on its deployed TP frames.
- No intervals are given here. The AntiUAV3D test set has 2 sequences, so any interval would be within-flight (block bootstrap, cf. A3).
- The perfect-size row keeps the detector's own rotation and apparent width. The C1-style row (size_c1) additionally corrects the rotation footprint.
