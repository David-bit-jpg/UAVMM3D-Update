# N1clean: does the self-trained detector copy its pseudo labels? Clean MAV6D chain srcx/in2mav (11-13)

**Short answer: the effect holds on the clean chain, but it is weaker than the 0.95 from nl075.**
- **Held-out val (16 sequences, 14 of which have scored frames):** the student's distance error correlates **r = 0.83** [0.73, 0.90] with the pseudo-label error. After per-sequence demeaning, r = **0.85** [0.76, 0.91].
- **Training frames:** r = **0.93** [0.90, 0.94]; demeaned 0.91.
- **Error sizes on val:** the student is 3.0% from the pseudo label, against 5.0% from GT. The pseudo label itself is 5.3% from GT.

**Suggested wording:** "error correlation 0.83 on held-out validation sequences (0.93 on its training frames)". An alternative is to say "closely" rather than "almost frame by frame".

## What was measured
- **Pseudo label:** the rule in `srcx/lib/make_pseudo_x.py`.
  - The box is det0's (the stage-1 3-seed ensemble with flip), kept when conf ≥ 0.5.
  - Depth: t ← t · exp(δ̂ − 0.75 · c̄) when |δ̂| ≤ log 1.4.
  - δ̂ is the mean of the stage-2 refiners `ref_st2_s5` and `ref_st2_s6`, averaged with the `avg_inf.py` rule.
  - **c̄ = 0.09925 is the TRAIN constant**, computed from the chain's own `inf_st2_det0_train.pkl`. It was not re-estimated on val. The val δ̂ median is 0.0999, so the choice barely matters.
- **Val δ̂:** new inference of the same two stage-2 refiners on det0's val detections, run with `n1c_refiner_infer.py`. The code path is identical to `refiner_x.cmd_infer`, with memory fraction 0.15 and batch 4, launched through `gpu_slot.sh light`. Crops were produced for 2336 of 2336 frames, and all 835 conf ≥ 0.5 frames were corrected.
- **Student:** det1, the self-trained 3-seed ensemble with flip (`recs/det1.pkl`), on the same frames.
- **Error:** e = log(z / z_GT). GT comes from `E:/mmcache/mav6d_cn_obj/{train,val}/index.pkl`.
- **Frames scored:** det0 conf ≥ 0.5, and both the pseudo-label box and the student box within 1 m of GT, as in N1.
- **Confidence intervals:** sequence cluster bootstrap with 2000 resamples. Statistics code is copied from N1.

## Checks
- **The train reconstruction reproduces `index_pseudo_srcx_in2mav.pkl` exactly:**
  - same 5227 frames;
  - max |Δ log z| = 5.9e-8 and max |Δt| = 2.4e-7 m (float32 rounding);
  - c̄ matches the index's stored value.
- **Batch-4 re-inference reproduces the chain's stage-2 train inference.** On the first 300 train frames, max |Δδ̂| is 5.8e-5 for s5 and 3.6e-5 for s6.
- **No overlap:** val and train share 0 of their sequences (61 train, 16 val).

## Main table

| Frames | n | n seq | Pearson raw [95% CI] | demeaned [CI] | Spearman | median \|rel err\| student vs GT | PL vs GT | student vs PL |
|---|---|---|---|---|---|---|---|---|
| **det1, val (held-out)** | 803 | 14 | **0.833** [0.726, 0.901] | **0.846** [0.757, 0.906] | 0.761 | 5.0% | 5.3% | **3.0%** |
| det1, train (fitted) | 5137 | 48 | 0.925 [0.903, 0.940] | 0.914 [0.891, 0.927] | 0.904 | 5.4% | 5.8% | 1.8% |

Median signed bias is +2.7% for the student and +2.8% for the pseudo label on val; on train it is +3.1% and +3.3%.

## Control: copying vs shared difficulty

| | n | Pearson raw | demeaned | median \|Δ\| |
|---|---|---|---|---|
| val: student vs pseudo label | 803 | **0.833** | **0.846** | **3.0%** |
| val: student vs raw det0 (before refiner correction) | 779 | 0.817 | 0.821 | 9.9% |
| val: raw det0 vs pseudo label | 775 | 0.812 | 0.845 | 9.7% |
| train: student vs raw det0 | 5078 | 0.794 | 0.815 | 9.9% |

- **Magnitude separates the two cleanly.** The student sits 3.0% from the pseudo label but 9.9% from raw det0. det0 is biased by −6.5%, and the student inherited the IR-refiner correction (+2.8%), not det0's scale.
- **Correlation separates them only weakly on val:** 0.83 against 0.82. On val, shared per-image difficulty explains most of the correlation. Copying shows up mainly in the gap size and bias, plus the in-sample rise to 0.93 (against 0.79 for raw det0 on train).

## Comparison with N1 (nl075 chain)
| | nl075 val | clean val | nl075 train | clean train |
|---|---|---|---|---|
| r raw | 0.945 | 0.833 | 0.970 | 0.925 |
| r demeaned | 0.938 | 0.846 | 0.966 | 0.914 |
| student vs PL | 1.9% | 3.0% | 1.2% | 1.8% |
| n frames | 1966 | 803 | 11893 | 5137 |

The clean chain trains on fewer pseudo labels: 5227 train frames pass conf ≥ 0.5, against 12,257 for nl075. Its student therefore tracks the labels less tightly.

## Caveats
- **The val set is small and skewed.** Only 835 of 2336 val frames pass det0 conf ≥ 0.5, and 795 of those are phantom4.
  - mavic2 sequences contribute 0–27 frames each.
  - Two sequences (mavic2/02/0202 and mavic2/02/0214) contribute none, so the bootstrap runs over 14 sequences.
  - The held-out figure is therefore mostly a phantom4 statement.
- **Held out by sequence, not by scene.** All val scenes also appear in training.
- **The student is the detector stage (det1), not the fused final chain output.**
- **Inference only.** No training was run, and nothing outside this folder was written.

## Files (this folder)
- `n1c_refiner_infer.py` and `run_infer.sh` produce:
  - `st2_s{5,6}_det0_val.pkl`: val δ̂;
  - `chk_s{5,6}_train300.pkl`: the reproduction check;
  - `infer.log`.
- `n1c_analyze.py` produces `n1c_results.json` and `n1c_analyze.log`.
