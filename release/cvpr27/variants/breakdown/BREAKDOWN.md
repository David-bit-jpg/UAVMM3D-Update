# Per-aircraft breakdown and AntiUAV3D val as a second held-out set (11-13; CPU only; existing predictions only)

Scripts:
- `per_aircraft.py` → `per_aircraft.json`, `per_aircraft.log`. Subset GT metas are in `meta/`.
- `val_boot.py` → `val_boot.json`, `val_boot.log`.

Scoring uses `srcx/lib/score_x.py`, which calls the pack `score.metrics`:
- **Per aircraft:** `score_meta` on a GT meta restricted to that aircraft's frames.
- **Val:** `score_val`.
- **Check:** every all-frame row reproduces the stored score JSON exactly (asserted).

Units:
- **ATE:** MAV6D in metres. AntiUAV3D is relative, as a fraction of the distance.
- **ASE:** cm.
- **AOE:** degrees, with 4-fold symmetry.

**Distance bias** is the median of log(z_pred / z_gt), where z is the depth of the box centre. It is taken over TP frames (top-1 box, projected 2-D IoU ≥ 0.1). A negative value means the distance is underestimated. Over all predicted frames, and with range instead of z, the values are within about 0.01–0.03 of the TP values (see the JSON).

## B1. MAV6D test (4,800 frames: Mavic 2 1,290 · Phantom 4 3,510)
| Arm | Aircraft | AP | ATE (m) | ASE | AOE | mDR | UDS | bias log z |
|---|---|---|---|---|---|---|---|---|
| PST (in2mav) | all | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | **60.81** | +0.046 |
| | Mavic 2 | 30.97 | 0.408 | 3.18 | 21.50 | 86.51 | 50.04 | +0.075 |
| | Phantom 4 | 44.70 | 0.273 | 1.30 | 12.24 | 99.72 | 64.60 | +0.039 |
| RGB-only (in2mav_none) | all | 38.04 | 0.348 | 2.58 | 16.32 | 95.98 | 57.38 | +0.053 |
| | Mavic 2 | 25.16 | 0.494 | 5.08 | 22.32 | 85.81 | 43.41 | +0.101 |
| | Phantom 4 | 42.54 | 0.301 | 1.79 | 14.43 | 99.72 | 62.08 | +0.040 |
| Sim-only | all | 26.87 | 0.283 | 1.74 | 20.32 | 69.77 | 49.51 | −0.033 |
| | Mavic 2 | 0.00 | 0.386 | 5.12 | 21.94 | 6.51 | 22.31 | +0.088 (84 TP) |
| | Phantom 4 | 38.35 | 0.281 | 1.65 | 20.28 | 93.02 | 58.31 | −0.035 |

How to read it:
- **Mavic 2 drives most of the adaptation gain.** The sim-only detector almost never finds it: mDR is 6.5 % and AP is 0. Self-training lifts mDR to about 86 % for both teacher settings.
- **The PST vs RGB-only gap is also mostly on Mavic 2:** +6.6 UDS there, against +2.5 on Phantom 4. Mavic 2 accounts for +5.8 AP, −8.6 cm ATE and −1.9 cm ASE of that gap.
- **PST overestimates distance slightly,** by about 4 % on Phantom 4 and about 8 % on Mavic 2. RGB-only's Mavic 2 bias is larger, about 10 %.

## B2. AntiUAV3D v2 test (2,624 frames: Matrice 4E 2,576 · Matrice 400 48)
| Arm | Aircraft | AP | ATE (rel.) | ASE | AOE | mDR | UDS | bias log z |
|---|---|---|---|---|---|---|---|---|
| PST (out2antifs) | all | 27.89 | 0.099 | 16.30 | 17.11 | 91.92 | **41.73** | −0.056 |
| | Matrice 4E | 29.23 | 0.090 | 16.07 | 16.96 | 91.81 | 42.89 | −0.054 |
| | Matrice 400 | 0.00 | 0.571 | 27.81 | 24.75 | 97.92 | 19.34 | −0.835 |
| RGB-only (out2antifs_none) | all | 23.39 | 0.111 | 15.97 | 16.45 | 90.93 | 38.92 | −0.086 |
| | Matrice 4E | 24.68 | 0.102 | 15.74 | 16.24 | 90.80 | 40.04 | −0.083 |
| | Matrice 400 | 0.00 | 0.552 | 27.22 | 26.87 | 97.92 | 18.88 | −0.794 |
| Sim-only | all | 16.04 | 0.132 | 15.36 | 26.06 | 78.54 | 30.55 | −0.062 |
| | Matrice 4E | 17.20 | 0.120 | 15.00 | 26.12 | 78.18 | 31.65 | −0.057 |
| | Matrice 400 | 0.00 | 0.629 | 30.87 | 23.18 | 97.92 | 19.68 | −0.989 |

How to read it:
- **Matrice 400 is only 48 frames, from one sequence.** Every arm detects it but places it at about 0.4× its true distance: exp(−0.8 to −1.0). That fits a large aircraft decoded with a small constant size s_bar. So AP is 0 for all arms, and the slice is too small to rank arms.
- **The whole test score is essentially Matrice 4E.** There PST beats RGB-only by +2.85 UDS and +4.55 AP, and beats sim-only by +11.2 UDS. PST's depth bias is also smaller (−5.4 % against −8.3 % for RGB-only).

## C. AntiUAV3D v2 val as a second held-out evaluation (1,148 frames, 2 sequences, Matrice 4E only)
**What val is used for.** No arm ever trains on val labels, and no choice was made from them. But the val images (no labels) set deployment constants, so val is transductive-unlabeled, not fully untouched:
- for every arm, including sim-only: the detector's constant size s_bar, from unlabeled val images;
- for PST and RGB-only: the b_k offsets, also from unlabeled val images.

The val labels give only a sanity score, so this is a second held-out evaluation of the labels, not of the images.

**Point metrics.** These are from the stored `*.score_val.json`, recomputed and checked equal.

| Arm | AP | ATE (rel.) | ASE | AOE | mDR | UDS | bias log z |
|---|---|---|---|---|---|---|---|
| PST (out2antifs) | 20.30 | 0.097 | 15.89 | 20.83 | 73.78 | **34.98** | −0.040 |
| RGB-only (out2antifs_none) | 16.88 | 0.105 | 15.74 | 20.88 | 74.48 | 32.95 | −0.058 |
| Sim-only | 16.58 | 0.104 | 15.00 | 29.22 | 67.60 | 30.15 | −0.052 |

**Paired moving-block bootstrap.** The method is the one in `review1113/A3`:
- circular blocks within each sequence, with the sequence strata kept;
- frames in numeric frame order;
- B = 2,000 resamples, seed 0, with both arms on the same resample.

**Block length.** L is the first lag at which the ACF drops below 0.1, taking the maximum over arms and over {distance error, AOE} in the long sequence. On val that gives L = 53 (test gave 176). L = 50 and L = 100 are shown for sensitivity.

| Pair | Metric | Point | 95 % CI, L = 53 | frac ≤ 0 | CI, L = 50 | CI, L = 100 |
|---|---|---|---|---|---|---|
| PST − Sim-only | UDS | +4.83 | [+1.07, +8.78] | 0.003 | [+1.14, +8.56] | [+0.77, +9.09] |
| PST − Sim-only | AP | +3.72 | [−1.35, +9.47] | 0.077 | [−1.22, +9.22] | [−1.73, +9.82] |
| PST − RGB-only | UDS | +2.03 | [+0.31, +3.76] | 0.013 | [+0.20, +3.61] | [−0.11, +4.05] |
| PST − RGB-only | AP | +3.42 | [+0.83, +6.09] | 0.007 | [+0.67, +5.96] | [+0.26, +6.43] |

Caveats:
- **Only 2 sequences.** `test_seq_0003` has 831 frames and `test_seq_0005` has 317. The intervals cover within-sequence variation only, not sequence-to-sequence variation.
- **`test_seq_0005` is almost undetected.** mDR is 22 % for PST, 23 % for RGB-only and 3 % for sim-only. There PST's UDS (10.9) is below sim-only's (14.9) because of AOE 63° on the few frames it detects.
- **The sign of the val result matches test,** but it is weaker:
  - PST − RGB-only UDS is +2.0 on val against +2.8 on test. Its CI touches 0 at L = 100.
  - PST − Sim-only AP is not significant on val.
