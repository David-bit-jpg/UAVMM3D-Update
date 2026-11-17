# GUPNet + classic sim→real UDA baselines (11-14)

Two standard unsupervised-domain-adaptation baselines on top of the GUPNet port, for the comparison table.
New code only; `tools/train_uav.py`, the dataset, the model, and the loss are unchanged. All new code is marked `[UDA]`.

| file | what it is |
|---|---|
| `code/tools/uda_train.py` | training entry for both baselines (`--method mt` / `--method dann`) |
| `../run1110/gupnet_uda_run.sh` | shared launcher: skip-if-done, resume, OOM→halve batch, predict, score, `.done` |
| `../run1110/gupnet_uda_mt.sh`, `../run1110/gupnet_uda_dann.sh` | the two queue entries (`gproto/local_queue_1113.txt`) |

Outputs go to `D:/cmpfull_runs/gupnet_uda_mt_s0` and `D:/cmpfull_runs/gupnet_uda_dann_s0`: `train.log`, `run_info.json`, `tau.json` (MT only), `checkpoints/checkpoint_latest.pth` (saved every epoch, used for resume), `checkpoint_epoch_{10,20,30,40}.pth`, `checkpoint_final.pth`, `pred_real_test.pkl`, `rescore_local.json`, `.done`. A one-line summary (AP ATE ASE AOE mDR UDS) goes to `run1110/gupnet_uda_<m>.log` and the end of `train.log`.

## Shared setup (both baselines)
- **Initialisation**: the source-only model `D:/cmpfull_runs/gupnet_s0/checkpoints/checkpoint_final.pth` (140 epochs on sim_train, real_test UDS 22.13).
- **Data**: source = `D:/cmpfull/sim_train/labels.pkl` with GT boxes and the dataset's usual augmentation (flip 0.5, crop 0.5, scale 0.4, shift 0.1). Target = `D:/cmpfull/real_train/labels.pkl`, 12,866 unlabeled MAV6D frames. When the target records are loaded, every field except `file/root/K/W/H/seq/frame` is dropped and `boxes` is forced to `[]`, with an assert. **No real label is read for training, thresholds, or selection.** real_test is predicted once with the final checkpoint and scored once by `score_x.py`.
- **Budget**: 40 epochs × 3,712 source samples. Each iteration uses one source batch and one target batch of the same size. Batch 8 gives 464 iterations per epoch and 18,560 in total, which is 28.6% of the source-only schedule.
- **Optimiser**: Adam, weight decay 1e-5 (the official values), constant lr **3.125e-5** = source lr 3.125e-4 × 0.1. lr is scaled linearly with batch size if an OOM forces a smaller batch. No warmup or decay. **No model selection**: the final checkpoint is used.
- **Detection loss**: the unchanged `GupnetLoss` with every term weighted 1. Hierarchical Task Learning (HTL) is switched off. At the end of source training its weights were already ≈1 (depth 0.98, all other terms ≥0.995; see `gupnet_s0/train.log`), so this keeps the end-of-training objective.
- **BatchNorm statistics frozen** (`--bn frozen`, the default): BN layers run in eval mode with the source running statistics. Their affine weights still train. Reason, measured on 24 random real_train frames with the source model: the input was the *identical, unaugmented* image and the pseudo box was re-encoded through the dataset. With BN in train mode (single-domain target batch), the student's depth differed from the eval-mode teacher's by a median of 2.56 m (max 10.8 m), and the depth loss was 27.7 instead of −0.86. So the pseudo-label loss would mostly be fitting BN-statistics noise. Frozen BN is also the standard setting for small-batch detector fine-tuning (Detectron2 FrozenBN, DA-Faster R-CNN implementations). `--bn train` restores train-mode BN.
- Seed 0. No worker processes (spawn-process limit on Windows): images are loaded by a 6-thread prefetching loader in the same process. The thread interleaving makes augmentation randomness not bit-reproducible.
- GPU: `GPU_FRAC=0.35` (8.4 GB cap). Peak allocation is about 3.5 GB at batch 4 and is expected to be about 7 GB at batch 8. Launched through `gproto/gpu_slot.sh light`. The entry file name contains `train.py`, so the shared schedulers count it as a training process.

## 1. Mean-teacher self-training (STMono3D-style)
Li et al., "Unsupervised Domain Adaptation for Monocular 3D Object Detection via Self-Training", ECCV 2022.
- Student and teacher are both initialised from the source model. Teacher = EMA of the student, momentum **0.999**, updated after every optimiser step (parameters and floating-point buffers). The teacher has no gradients and is always in eval mode.
- Each iteration:
  1. Source batch: GT detection loss, backward.
  2. Target batch, **weak view** (no flip, no crop, no jitter: the dataset's test path): the teacher predicts with the normal GUPNet decoding (heatmap NMS, top-50, score = heatmap × depth confidence). The **top-1 box per image** is kept if its score ≥ τ. Images without a kept box are left out of the student batch: they are not trained as all-negative images, because almost every MAV6D frame contains a drone.
  3. **Strong view** of the same frames: the dataset's own train path (flip 0.5, crop 0.5 / scale 0.4 / shift 0.1), plus colour jitter (brightness, contrast, saturation 0.4, hue 0.05) on the 960×540 image region. The pseudo box is passed in as the record's label, so flip, crop, heatmap, offsets, depth, and angle encoding go through exactly the code used for GT. The weak image tensor is converted back to the identical uint8 image, so there is no second disk read. Student loss = normal `GupnetLoss` × target weight **1.0**, backward. Then the optimiser step and the EMA update.
- **τ is fixed once** before training: τ = the 20th percentile of the *source model's* top-1 score over all 12,866 real_train frames, computed from images only. This keeps 80% of frames, the same rule as the +ST baseline (`make_pseudo.py --keep 0.8`). The value is saved in `tau.json`. In the smoke run with 80 frames, τ was 0.055 and the median top-1 score was 0.088; source-model scores on real images are low, so a fixed 0.3 would have kept almost nothing.
- **Prediction model = EMA teacher.** `checkpoint_final.pth`'s `model_state` is the teacher, and the student is stored as `student_state`. `predict.py` works unchanged.
- **Taken from STMono3D**: the EMA mean teacher, weak→strong teacher/student views, confidence-thresholded pseudo-labels, and training the target with the detector's own losses alongside the source GT loss.
- **Not taken**: their quality-aware supervision (score-weighted loss; available as `--quality-weight`, off), the geometry-aligned multi-scale training, and the positive/negative/ignore threshold split. GUPNet's crop already maps ROI boxes back to original-image pixels through `coord_ranges`, so geometric depth stays consistent under scaling. We also use top-1 per image instead of all boxes above the threshold (one drone per frame; same as +ST), and frozen BN (see above).

## 2. Adversarial feature alignment (DANN)
Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015.
- Domain classifier on the **DLAUp neck output** (stride 4, 64 channels; this is the map every GUPNet head reads). Architecture: 3 × [conv3×3 stride 2, ReLU] (64→128→128→128) → global average pool → Linear 128→64 → ReLU → Dropout 0.5 → Linear 64→1. It has no BN, because each batch is single-domain. It is trained with binary cross-entropy: source = 0, target = 1.
- **Gradient reversal layer** between the neck and the classifier, with the DANN schedule λ_p = λ_max · (2 / (1 + exp(−10 p)) − 1), where p = training progress in [0, 1]. **λ_max = 0.1** is the image-level adversarial weight usually used for detection (DA-Faster R-CNN). The domain classifier itself trains at full rate, with lr = **10 ×** the network lr, the DANN convention for newly added layers.
- Each iteration: source batch → GT detection loss + domain loss (label 0), backward. Target batch → backbone + neck only → domain loss (label 1), backward. One optimiser step. No target detection loss.
- Prediction model = the fine-tuned detector. The domain classifier is stored in the checkpoint (`disc_state`) but is not used at test time.
- **Deviations**: λ_max is 0.1 instead of the paper's 1.0 (the image-level weight used in DA-Faster R-CNN; fixed a priori and not tuned, because no real-data selection is allowed). The paper used SGD with an annealed lr; we use Adam with a constant small lr from the source checkpoint. BN is frozen.

## Smoke tests (scratch: `C:/Users/jiang/AppData/Local/Temp/claude/0fa2f3b7e932577a/62c031f1-65b4-4079-9760-1d45e27ef298/scratchpad/uda_smoke`)
The GPU was shared during these runs: 14–19 GB was already in use by other jobs at 90–99% utilisation.
- **MT**, batch 4, 32 iterations: losses finite, no skipped steps. Source loss stayed at the end-of-training level (src_det −0.9 to 0.0). Target loss went from 11.4 to 6–13 (seg 2.1 → 0.7–1.5). On average 3.3 of 4 target frames got a pseudo box. Peak memory 3.52 GB, about 1.2–1.4 it/s.
- **DANN**, batch 4, 32 iterations: losses finite. λ ramped to 0.1, and the domain loss moved (source 0.67 → 0.46). Peak memory 3.38 GB, about 1.4 it/s.
- **Resume**: continued from `checkpoint_latest` at the correct epoch and iteration.
- **OOM path**: with `GPU_FRAC=0.02`, the run hit exit 3 → halved batch → hit exit 3 again → reported failure, as designed.
- **End to end**: launcher → train → `predict.py` (real_test, 4,800 frames) → `score_x.py` → `rescore_local.json` → `.done` → summary line. The 16-iteration MT smoke scored UDS 22.13, the same as the source model; this is expected, because an EMA teacher with momentum 0.999 is still ≈ the source model. It confirms the predict/score path.
- **Data loading**: about 80 ms per item (decode + affine). 6 threads give 0.31 s per batch of 8, against 0.65 s with a single thread.

**Estimated full run** (batch 8, 18,560 iterations; depends heavily on GPU contention): MT about 6–8 h (≈0.7 it/s), DANN about 4–6 h. Prediction plus scoring adds about 10 min.
