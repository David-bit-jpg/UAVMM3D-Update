# N2 — chance-level and constant orientation baselines (CPU, 2026-11-13)

Script `n2_chance.py`, raw `n2_results.json`, log `n2_run.log`. GT is read only to score.

**Metric (same as the paper code).** AOE = min over q of the geodesic between Rp·Rz(90°q) and Rg, using the arccos-trace form from `cmpeval/paper_table.py` `sym`, which matches pack `score.py` `sym`. Yaw = min over q of |atan2(E10, E00)| with E = Rgᵀ·Rp·Rz(90°q), giving 0–45°, as in `strict/final_eval.yawerr` and `ou_eval.errs`. The label-error check uses the chordal form from `pseudo_build.py:34-35` and reproduces the reference exactly: **20.644 on all 4131 frames and 16.517 on 1491 passing frames** (set `t6il`). Medians on those frames: AOE 17.76 / 12.91, yaw 12.42 / 8.14.

**Caveat.** These baselines are averaged over *all GT frames*. Paper AOE is averaged over true positives only (2D IoU ≥ 0.1), so the frame sets differ slightly.

**Reference numbers.** Teacher labels on MAV6D train: 20.64 (all) / 16.52 (passing). PST: AOE 13.54 on MAV6D test and 17.1 on AntiUAV3D test.

| split | predictor | AOE mean | AOE median | yaw mean | yaw median | n |
|---|---|---|---|---|---|---|
| MAV6D test | Random rotation (Haar) | 93.65 | 92.86 | 26.87 | 24.22 | 240000 |
| MAV6D test | Random yaw about body z (GT tilt kept) | 22.48 | 22.50 | 22.48 | 22.50 | 240000 |
| MAV6D test | Best constant (oracle, fit on this split) | 19.78 | 17.23 | 14.51 | 9.17 | 4800 |
| MAV6D test | Constant fit on sim train labels | 19.99 | 16.60 | 14.48 | 9.74 | 4800 |
| MAV6D test | Constant fit on MAV6D real train GT | 26.09 | 26.86 | 22.84 | 24.18 | 4800 |
| MAV6D test | Identity in camera frame | 91.16 | 90.87 | 31.52 | 16.52 | 4800 |
| MAV6D test | Level, body x toward camera (-z_cam) | 19.85 | 17.83 | 14.70 | 8.65 | 4800 |
| MAV6D test | Level, facing camera along GT line of sight | 20.45 | 18.59 | 15.35 | 11.64 | 4800 |
| AntiUAV3D test | Random rotation (Haar) | 93.70 | 93.05 | 26.98 | 24.40 | 131200 |
| AntiUAV3D test | Random yaw about body z (GT tilt kept) | 22.46 | 22.47 | 22.46 | 22.47 | 131200 |
| AntiUAV3D test | Best constant (oracle, fit on this split) | 18.38 | 17.03 | 14.62 | 13.08 | 2624 |
| AntiUAV3D test | Constant fit on sim train labels | 20.43 | 18.93 | 15.58 | 13.71 | 2624 |
| AntiUAV3D test | Constant fit on MAV6D real train GT | 30.21 | 30.81 | 27.03 | 28.57 | 2624 |
| AntiUAV3D test | Identity in camera frame | 83.70 | 84.57 | 27.59 | 19.91 | 2624 |
| AntiUAV3D test | Level, body x toward camera (-z_cam) | 20.15 | 19.10 | 14.64 | 14.39 | 2624 |
| AntiUAV3D test | Level, facing camera along GT line of sight | 19.60 | 19.42 | 14.25 | 11.12 | 2624 |
| MAV6D train (all GT frames) | Random rotation (Haar) | 93.68 | 93.00 | 26.87 | 24.22 | 643300 |
| MAV6D train (all GT frames) | Random yaw about body z (GT tilt kept) | 22.48 | 22.48 | 22.48 | 22.48 | 643300 |
| MAV6D train (all GT frames) | Best constant (oracle, fit on this split) | 24.95 | 25.32 | 19.50 | 21.08 | 12866 |
| MAV6D train (all GT frames) | Constant fit on sim train labels | 27.60 | 27.33 | 22.88 | 21.98 | 12866 |
| MAV6D train (all GT frames) | Constant fit on MAV6D real train GT (in-sample) | 24.95 | 25.32 | 19.50 | 21.08 | 12866 |
| MAV6D train (all GT frames) | Identity in camera frame | 93.21 | 92.55 | 32.26 | 28.90 | 12866 |
| MAV6D train (all GT frames) | Level, body x toward camera (-z_cam) | 27.68 | 29.69 | 23.20 | 26.17 | 12866 |
| MAV6D train (all GT frames) | Level, facing camera along GT line of sight | 27.50 | 27.69 | 22.77 | 22.92 | 12866 |
| MAV6D train, teacher-label frames (all) | Random rotation (Haar) | 93.69 | 92.96 | 26.85 | 24.18 | 206550 |
| MAV6D train, teacher-label frames (all) | Random yaw about body z (GT tilt kept) | 22.48 | 22.48 | 22.48 | 22.48 | 206550 |
| MAV6D train, teacher-label frames (all) | Best constant (oracle, fit on this split) | 24.82 | 25.23 | 19.49 | 20.93 | 4131 |
| MAV6D train, teacher-label frames (all) | Constant fit on sim train labels | 27.61 | 27.24 | 23.00 | 22.02 | 4131 |
| MAV6D train, teacher-label frames (all) | Constant fit on MAV6D real train GT (in-sample) | 24.82 | 25.26 | 19.49 | 20.88 | 4131 |
| MAV6D train, teacher-label frames (all) | Identity in camera frame | 93.15 | 92.57 | 32.18 | 28.80 | 4131 |
| MAV6D train, teacher-label frames (all) | Level, body x toward camera (-z_cam) | 27.69 | 29.68 | 23.32 | 26.18 | 4131 |
| MAV6D train, teacher-label frames (all) | Level, facing camera along GT line of sight | 27.45 | 27.91 | 22.83 | 22.98 | 4131 |
| MAV6D train, teacher-label frames (passing) | Random rotation (Haar) | 93.45 | 92.75 | 26.86 | 24.11 | 74550 |
| MAV6D train, teacher-label frames (passing) | Random yaw about body z (GT tilt kept) | 22.53 | 22.56 | 22.53 | 22.56 | 74550 |
| MAV6D train, teacher-label frames (passing) | Best constant (oracle, fit on this split) | 24.76 | 26.09 | 20.21 | 21.65 | 1491 |
| MAV6D train, teacher-label frames (passing) | Constant fit on sim train labels | 26.85 | 26.38 | 22.70 | 21.36 | 1491 |
| MAV6D train, teacher-label frames (passing) | Constant fit on MAV6D real train GT (in-sample) | 24.81 | 25.49 | 20.25 | 21.85 | 1491 |
| MAV6D train, teacher-label frames (passing) | Identity in camera frame | 91.63 | 91.82 | 28.81 | 21.15 | 1491 |
| MAV6D train, teacher-label frames (passing) | Level, body x toward camera (-z_cam) | 26.77 | 28.99 | 22.82 | 25.71 | 1491 |
| MAV6D train, teacher-label frames (passing) | Level, facing camera along GT line of sight | 26.73 | 27.34 | 22.62 | 22.36 | 1491 |

**Notes**
- Haar: 50 random draws per frame (n counts draws). The 90° fold only removes yaw ambiguity, so random tilts and flips leave the error near 93°.
- Random yaw: GT rotation times Rz(ψ) about the body z axis, with ψ uniform. After folding the error is uniform on [0, 45°], so mean = median = 22.5° analytically; the simulation confirms this.
- Oracle constant: start from 400 GT rotations as candidates, then refine the top 5 with Nelder–Mead on a rotation vector, minimising the mean AOE on the evaluated split. It is an upper bound for any constant.
- Sim constant: fitted the same way on `E:/mmcache/full_v2/train` (19490 frames). It uses no real labels. Euler xyz (deg) = [75.1, 85.5, -14.9].
- Real-train constant: fitted on MAV6D train GT. It transfers *badly* to MAV6D test (26.1) because the orientation distributions of the train and test sequences differ.
- "Level": body z = camera up (−y_cam). "Line of sight" sets body x to the horizontal direction toward the camera, computed from the GT position.

**Reading.** On MAV6D test, PST 13.54 beats the oracle constant 19.78 by 6.2° and the non-oracle sim constant 19.99 by 6.5°. On AntiUAV3D test, PST 17.1 is only **1.3° below the oracle constant 18.38**, and 3.3° below the sim constant 20.43 and 2.5° below line-of-sight 19.60. On MAV6D train, teacher labels at 20.64 (all) / 16.52 (passing) beat the oracle constant on the same frames, 24.82 / 24.76, and random yaw at 22.5 by more. All learned numbers are far below Haar chance (~93.7°).
