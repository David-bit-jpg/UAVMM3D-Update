# MonoDGP (CVPR 2025) → cmpfull UAV data — port notes

Source: official repo copied from `E:/ext_repos/monodgp` (original untouched). Every changed line is tagged `[UAV port]`.
Spec: `../SPEC.md`.

## Entry points
- Train: `bash train.sh --labels D:/cmpfull/sim_train/labels.pkl [pseudo.pkl[@img_root] ...] --out <dir> --seed 444 --epochs 48 --batch-size 2`
  (= `python tools/train_uav.py ...`; `GPU_FRAC` env caps GPU memory, default 0.2; `--max-iters` for smoke only;
  `--keep-every 4` also writes model-only `model_epoch_N.pth` snapshots for selection on sim_val.)
  Pseudo-label pkls in another folder: `pseudo.pkl@D:/cmpfull/real_train` (image root; default = pkl folder; or an `img_root` key per record).
- Predict: `python predict.py --ckpt <dir>/checkpoint.pth --split real_test|sim_val|real_train --out pred.pkl`
  → list in labels.pkl order, `dict(seq, frame, boxes9d (n×9: center, lwh, euler_xyz, our definition), scores (n,))`, n ≤ 10.
- Self-check: `python predict.py --roundtrip --split sim_val --limit 50` (GT → training encoder → decoder → boxes9d).
- Original KITTI launcher kept as `train_kitti.sh`; KITTI dataset code untouched (just no longer imported eagerly).

## What changed and why
| file | change |
|---|---|
| `lib/datasets/uav/uav_dataset.py` (new) | Replaces `kitti_dataset.py`. `__getitem__` kept step by step: photometric distortion, flip 0.5, crop/scale 0.05/shift 0.05 affine (prob 0.5), mixup3d 0.5 (same K + same size), 3D-center-inside-image filter, l/r/t/b ≥ 0 filter (clip_2d False), depth target = z·crop_scale, same target dict. KITTI semantics replaced (below). |
| `lib/models/monodgp/monodgp.py` | `num_classes` 1 (config). Angle head 24 → 72 (3 Euler angles × original 12-bin cls + 12 residuals); `loss_angles` applies the original per-angle loss (CE on bin + L1 on residual of GT bin) to each angle and sums them (equal weight, same `angle_loss_coef`=1). Depth-map loss: hard-coded `[80, 24, 80, 24]` (stride-16 map of 1280×384) → actual map size. `.cuda()` → `device=`. |
| `lib/helpers/decode_helper.py` | `gather` of the heading uses its real width (72). |
| `lib/models/monodgp/ops/functions/ms_deform_attn_func.py` | CUDA op not compiled → falls back to the repo's own `ms_deform_attn_core_pytorch` (same operator: bilinear `grid_sample`, zero padding, align_corners=False, weighted sum; = the CUDA kernel's sampling). Affects speed/memory only, not results. If `MultiScaleDeformableAttention` is ever built it is used automatically. |
| `lib/models/monodgp/ops/modules/ms_deform_attn.py` | torch-version check mis-parses torch 2.x (precedence bug) → try/except import. |
| `lib/models/monodgp/region_seg_head.py` | `sklearn` import made lazy (only used by `cluster_regions`, never called). |
| `lib/models/monodgp/backbone.py` | `pretrained=True` → `weights='IMAGENET1K_V1'` (the identical legacy checkpoint; torchvision 0.26 has no `pretrained=`). `MONODGP_NO_PRETRAIN=1` = random init, smoke test only. |
| `lib/helpers/dataloader_helper.py` | `type: UAV` branch; no test loader. |
| `lib/helpers/trainer_helper.py` | absolute `--out`, loss jsonl log, `max_iters` stop (smoke), optional model-only snapshots. Optimisation untouched (no grad clipping, as official). |
| `configs/monodgp_uav.yaml` | copy of `monodgp.yaml`; only dataset type/labels, `num_classes: 1`, `writelist: [drone]`, filter knobs, mean size (unused). All model/loss/matcher/optimizer/augmentation values official. |
| `tools/train_uav.py`, `train.sh`, `predict.py` (new) | entries. |

### Label semantics (KITTI → ours)
- **Center**: KITTI `pos` is the bottom center and the code itself converts to the geometric center (`pos + [0,-h/2,0]`) for the projected 3D center; we use the geometric center directly. Decoding drops the `locations[1] += h/2` (bottom-center shift). Depth target = z of the geometric center (KITTI: z of bottom center = same z for upright boxes).
- **2D box**: KITTI uses annotated 2D boxes (`bbox2d_type: anno`); we have none → tight box of the 8 projected corners, clipped to the 960×540 image.
- **Filters**: KITTI drops `level == UnKnown` (box height < 25 px or truncation > 0.5 or occlusion > 2) and `z < 2 m`, and masks truncation > 0.5 / occlusion > 2. We keep truncation ≤ 0.5 (truncation = 1 − clipped/unclipped projected-box area; occlusion unavailable = 0). **The 25-px height rule and z ≥ 2 m are off** (`min_box_height: 0`, `min_depth: 0`): they are KITTI evaluation-protocol / road-scene thresholds; with them 45 % of sim_train drones (h2d < 25 px at 960 width) and the near drones (real test 1.6–5.6 m) would be discarded. The 65 m cap stays (no effect).
- **Size**: original `[h, w, l]` order with h = `lwh[2]` (object z = drone up axis; verified on sim_train: object z ↦ camera −y, |R[:,2]·ŷ| = 0.97), w = `lwh[1]`, l = `lwh[0]`. Official config `meanshape: False` ⇒ absolute sizes (no class mean); sim_train single-class mean `[0.251, 0.620, 0.646]` is filled in but only used if `meanshape` is switched on.
- **Rotation (revised 10-29, coordinator decision)**: KITTI-style `ypr = [yaw, pitch, roll]` from `../angles.py` (R = Ry(yaw)·Rx(pitch)·Rz(roll)·R0; labels.pkl has `ypr`, otherwise computed from R). **yaw is the KITTI ry** and gets the full original treatment: `alpha = ry − atan2(u2d − cu, fu)`, 12-bin + residual, decoded with `alpha2ry`. pitch and roll (±30° in sim) use the same 12-bin + residual on the absolute angle. 3 angle losses summed with equal weight. predict converts ypr → euler_xyz with `angles.euler_xyz_from_ypr`. (The earlier euler_xyz encoding had a branch jump at |yaw| > 90°; gone now.)
- **Flip augmentation**: KITTI `ry → π − ry`. The nose/up-preserving mirror R' = S·R·D (S = diag(−1,1,1) camera-x mirror, D = diag(1,−1,1) flips the lateral axis) is, in this ypr convention, exactly **(yaw, pitch, roll) → (π − yaw, −pitch, roll)** (checked on 500 random R: 1e-15). Note `pitch` here = rotation about the level nose axis (Rx), `roll` = about the lateral axis (Rz), so it is pitch that flips sign. The coordinator's suggested "(π − yaw, pitch, −roll)" is **not** a proper mirror in this convention (R error up to 2.0); the code uses the verified one. Checked in the pipeline: decoded R of flipped samples vs S·R_gt·D max 9e-6°, corner reprojection vs mirrored GT 0.25 px (mirror about W/2 instead of cu, same approximation as the original).
- **Geometry-error depth (MonoDGP's core)**: unchanged code: `depth = h3d · fx / h2d + Δ_geo`, h3d = predicted size[0] (our drone height `lwh[2]`), h2d = predicted 2D box height (t + b). For KITTI h2d ≈ projected car height because cars are upright and pitch ≈ 0. **For drones the 2D box height also contains the projection of the tilted top/bottom faces and of l, w** (pitch/roll up to ±90° in sim; drones are flat, h ≈ 0.25 m vs l, w ≈ 0.63 m), so h3d·f/h2d is systematically smaller than z for tilted drones and the learned Δ_geo has to absorb it. This is MonoDGP's own design (its "geometry error" is exactly meant to absorb such deviations); we did not replace h with a rotation-aware projected height, to avoid changing the method.
- **Matching**: Hungarian costs unchanged (class 2, 3D-center 10, 2D l/r/t/b 5, GIoU 2), 11 query groups × 50 queries in training, 1 group at test.

### Input resolution
SPEC says 960×540 → zero-pad bottom to 960×544. **MonoDGP needs H divisible by 64**: `RegionSegHead` upsamples the stride-64 level ×2 and adds it to the stride-32 level; 544/32 = 17 → 9·2 ≠ 17 → crash. We therefore pad the bottom to **960×576** (same 960×540 content and same scaled K, 32 more zero rows; official KITTI input 1280×384 is also a multiple of 64). No model code changed for this. (Alternative would be editing the upsample to `size=`, i.e. a model change; we preferred padding.)

## Decoding / output
Exactly `decode_helper.py`: top-50 of sigmoid class probs, score = prob · exp(−log σ) (depth uncertainty), center = back-projection of predicted projected 3D center with the predicted depth, yaw via `alpha2ry` with the 2D-box center x. DETR has no NMS; we keep the 10 highest final scores and apply **no** score threshold (official tester filters prob < 0.2 before writing KITTI txt; left to the evaluator).

## Epochs (recommendation: 48)
Official KITTI: 250 epochs × 3,712 images, batch 8 → 464 it/epoch, **116,000 iterations, 928k samples**; lr 2e-4 AdamW, step ×0.5 at [85,125,165,205]; best val epoch in the 3 official logs 177/177/201 (selected on KITTI val each epoch).
sim_train: 19,490 frames → 48 epochs × 19,490 = 935k samples = 117k iterations at batch 8. `--epochs 48` therefore matches both the official iteration count (at batch 8) and the number of samples seen. Decay epochs are rescaled proportionally → [16, 24, 32, 39].
If batch < 8 the lr is scaled linearly (batch 2 → 5e-5) and epochs stay 48 (same samples; 4× more iterations).
Model selection: the official code picks the best epoch on val; we save `checkpoint.pth` (latest) + model-only `model_epoch_N.pth` every 4 epochs so the unified evaluator can select on **sim_val** (no real labels).

## Cost (RTX, 24 GB, pure-PyTorch deformable attention)
- batch 2: peak `max_memory_allocated` 4.94 GB, reserved 5.29 GB → **does not fit `GPU_FRAC=0.2` (4.77 GB; OOM at iteration 2)**; needs **`GPU_FRAC=0.25`** (≈ 6 GB). Smoke run used 0.35 as cap, actual use 5.3 GB.
- speed 3.4 it/s at batch 2 → 9,745 it/epoch ≈ **47 min/epoch** → 48 epochs ≈ **38 h** (one GPU, workers 2).
- batch 8 (official) would need roughly 4× activations (~16–20 GB) — not attempted.
- Checkpoint (with optimizer) 496 MB; model-only snapshot ≈ 165 MB.

## Smoke test after the ypr change (2026-10-29, GPU_FRAC 0.25, batch 2, 100 it, random-init backbone)
loss_detr 71.1 → 52.2 (first/last 20 it), ce 1.61 → 0.70, depth 2.51 → 1.87, angle 7.24 → 4.39, center 1.02 → 0.73, depth_map 1.49 → 0.65; peak alloc 4.94 GB / reserved 5.29 GB (fits 0.25). Round trip sim_val 50 frames: 5e-7 m / 5e-5 px / 6e-6°; sim_train 1000 frames: 9e-7 m / 7e-5 px / 1e-5°. predict on sim_val/real_test 50: order + format OK, (10, 9) float32 + (10,).

## Earlier smoke test (2026-10-29, euler_xyz encoding, random-init backbone because the ImageNet weights are not on disk)
- 200 iterations, batch 2, sim_train: loss_detr mean of first 20 it 72.8 → last 20 it 54.6; loss_ce 1.70 → 0.68, depth 2.50 → 1.69, depth_map 1.46 → 0.57, angle 7.73 → 6.29, center 1.00 → 0.68, giou 2.66 → 1.94 (region dice ≈ flat 3.8, bbox l1 ≈ flat).
- Round trip (GT → encoder → decoder, sim_val 50 frames, 50 boxes): max corner error 5e-7 m, max corner reprojection error 5e-5 px, max rotation error 6e-6°. Flip round trip: 0.25 px max (see above).
- `predict.py` on sim_val 50 and real_test 50: same order/seq/frame as labels.pkl, boxes9d (10, 9) float32, scores (10,). (Smoke model is untrained, so the boxes themselves are meaningless.)

## Open items / fidelity caveats
1. **ImageNet ResNet-50 weights must be downloaded** before formal training (torchvision `resnet50-0676ba61.pth`, ~98 MB, from download.pytorch.org, into `%USERPROFILE%/.cache/torch/hub/checkpoints/`). Without them the baseline is badly weakened — do not train with `MONODGP_NO_PRETRAIN=1`.
2. 576 instead of 544 padding height (architectural constraint, content identical) — accepted by coordinator.
   Decisions accepted: filters off, GPU_FRAC 0.25, batch 2 + linear lr, 48 epochs; pretrained-weight download pending user.
3. KITTI 25-px / 2-m training filters disabled (see above).
4. (resolved) angles now KITTI-style ypr; flip uses (π − yaw, −pitch, roll), verified against R.
5. Geometry depth uses the drone height vs a 2D box height inflated by tilt (method kept as is; Δ_geo absorbs it).
6. Batch 2 + linear lr instead of batch 8 (memory). Group-DETR matching and the 11 query groups are per image, so batch size only affects optimisation noise.

## 10-31 续训学习率修正
`lib/helpers/trainer_helper.py` 第 72 行：官方续训设 `lr_scheduler.last_epoch = epoch - 1`，但存档时已调用 step() epoch 次，续训后每个衰减点会晚一轮；改为 `= epoch`（与 MonoDETR 移植一致）。monodgp_s0 在此之前从未从存档续训过（所有重启都早于 10-30 12:44 第一个 checkpoint），不受影响。
