# E1geo: deploy-only variant of the sensor-free control E1 (MAV6D)

E1geo = E1 with two DEPLOYMENT-only changes; every network, pseudo-label, w and stage-3 target is E1's own (read only).
- **(a) two brightness classes**: PST's rule (1-D two-means of the silhouette body brightness `lum` over unlabeled train crops, nearest centre at
  deployment), applied to E1's own hull-supervised region head; c_k (refiner offset, train) and beta'_k (fusion offset, unlabeled val images)
  recomputed per class exactly as `chain_x deploy`. **Training stays one-class**: the stage-3 fused targets (beta_k inside `chain_x targets`) were
  built by E1 with one class and are NOT recomputed (that would need a stage-3 retrain).
- **(b) per-class height ratio r_k** from box-corner PnP of the RGB-only KZ teachers (KZs0/1 after E1's AdaBN; H6 refit rule, l, w fixed at s_bar),
  median over unlabeled train crops per class; h = r_k (l + w) / 2 as in `chain_x deploy`. Student-derived r_k is a diagnostic only (students are
  trained toward s_bar-shaped pseudo boxes, so their aspect collapses to the constant). No IR/LiDAR-trained network is used.
- Faithfulness: the patched chain copy with both switches off reproduces E1's own test predictions exactly (asserted per run).
- Test scored once per run (pack score.py, real_test). Paired sequence bootstrap (10 test sequences, B = 2000) via heads/heads_boot.py.

## Per run (test)

| run | E1 UDS | E1geo UDS | PST UDS | E1geo - E1 UDS [95% CI] | E1geo - PST UDS [95% CI] | E1geo - E1 ASE | E1geo - E1 ATE [95% CI] |
|---|---|---|---|---|---|---|---|
| mav_s0 | 53.48 | 52.54 | 60.81 | -0.94 [-2.03, +0.73] | -8.28 [-10.60, -6.45] | -0.12 | +0.012 [-0.004, +0.025] |
| mav_s10 | 61.70 | 61.36 | 59.28 | -0.34 [-1.13, +0.68] | +2.08 [-2.39, +5.52] | -0.10 | +0.004 [-0.003, +0.010] |
| mav_s20 | 54.93 | 53.94 | 62.70 | -1.00 [-2.12, +0.68] | -8.76 [-10.00, -6.71] | -0.01 | +0.010 [-0.007, +0.023] |

## Pooled over finished seed pairs (mean of per-pair differences, pooled 95% CI, P(diff <= 0))

- **E1geo - E1** (3 pairs): arm mean UDS 55.95, ref mean UDS 56.71 | UDS -0.76 [-1.72, +0.68] (P<=0 0.866) | AP -1.48 [-2.26, -0.29] (P<=0 0.991) | ATE +0.009 [-0.004, +0.019] (P<=0 0.089) | ASE -0.08 [-0.74, +0.42] (P<=0 0.570) | AOE -0.00 [-0.01, +0.00] (P<=0 0.764) | mDR -0.04 [-0.11, +0.00] (P<=0 1.000)
- **E1geo - PST** (3 pairs): arm mean UDS 55.95, ref mean UDS 60.93 | UDS -4.98 [-6.45, -3.67] (P<=0 1.000) | AP -6.00 [-8.30, -4.03] (P<=0 1.000) | ATE +0.059 [+0.036, +0.089] (P<=0 0.000) | ASE +0.82 [+0.57, +1.10] (P<=0 0.000) | AOE +1.12 [+0.07, +2.15] (P<=0 0.018) | mDR +0.21 [+0.00, +0.52] (P<=0 0.028)

## Deployment constants

- mav_s0: E1 one class c_k {-1: 0.100, 0: 0.100} r_k {0: 0.561} beta'_k {0: 0.328} | E1geo centres [84.3, 138.4] (train bright 0.43) c_k {-1: 0.100, 0: 0.100, 1: 0.101} r_k {0: 0.402, 1: 0.543} beta'_k {0: 0.363, 1: 0.267} | student r_k (diag) {0: 0.524, 1: 0.548} | test frames without class 75
- mav_s10: E1 one class c_k {-1: 0.088, 0: 0.088} r_k {0: 0.561} beta'_k {0: 0.373} | E1geo centres [86.0, 139.0] (train bright 0.44) c_k {-1: 0.088, 0: 0.084, 1: 0.090} r_k {0: 0.405, 1: 0.546} beta'_k {0: 0.394, 1: 0.326} | student r_k (diag) {0: 0.534, 1: 0.556} | test frames without class 148
- mav_s20: E1 one class c_k {-1: 0.106, 0: 0.106} r_k {0: 0.561} beta'_k {0: 0.336} | E1geo centres [89.3, 151.1] (train bright 0.39) c_k {-1: 0.106, 0: 0.107, 1: 0.105} r_k {0: 0.412, 1: 0.541} beta'_k {0: 0.363, 1: 0.295} | student r_k (diag) {0: 0.534, 1: 0.550} | test frames without class 1

## Reading

PST - E1 (the reported sensor-free gap) splits into PST - E1geo (what remains once E1 gets two brightness classes and a
sensor-free per-class height ratio at deployment) plus E1geo - E1 (what those two non-sensor deployment pieces were worth inside E1).
Caveat: E1geo cannot reproduce two-class stage-3 TRAINING targets (beta_k) or any upstream use; only deployment constants change.
