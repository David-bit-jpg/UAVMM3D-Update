# N1: does the student copy its pseudo labels on held-out sequences? (11-13)

**Short answer: yes, the claim holds on held-out sequences, though it is a little weaker than on training frames.** On the 16 MAV6D validation sequences, which the student never trained on, the final detector's distance error correlates **0.945** with the error of the pseudo label the teacher would produce on the same frames. On its own training frames the figure is 0.970. After subtracting each sequence's mean error, the held-out figure is **0.938**. The student is 1.9% away from the pseudo label but 6.4% away from the truth, so it sits about 3× closer to the pseudo label than to the truth. The intro's "almost frame by frame" is therefore supported out of sample. The paper should quote the held-out figure, not the in-sample 0.97.

## What was measured
- **Student (final label-free chain):** detector nl075, a 3-seed ensemble with flip test-time augmentation and no temporal background.
  - Val predictions: `geo1004/nl075_ens/ens_val.pkl`, the existing chain evaluation file.
  - Train predictions: `nolabel/nl075/recs/nl075_train.pkl`.
- **Teacher's pseudo label on val:** produced with the same networks and the same rule as `make_nolabel_pseudo.py`.
  - First the ST2 5-model ensemble runs with flip and no temporal background. This was a new inference-only run on val (`real_val_preds_st2nb.pkl`). Its self-calibrated size matches the train run.
  - Then the first-generation second stage (IR-silhouette-distilled ranging network, `V2Pp_s0..4`) gives δ̂.
  - Pseudo depth = t_det · exp(δ̂ − 0.75 · c), with |δ̂| ≤ log 1.4. The constant **c = 0.1263 is the training-set constant**; it was not re-estimated on val.
  - Frames are kept when confidence ≥ 0.5, using the top-confidence box.
  - Check: on the training frames, the reconstructed pseudo labels match the index actually trained on (`index_pseudo_nl075.pkl`, 12,257 frames) to within 6e-8 in log z.
- **Error:** e = log(z / z_GT). The correction rescales the whole translation, so the z ratio equals the distance ratio.
- **Frames used:** the teacher's confidence is ≥ 0.5, and both the teacher's and the student's top boxes are within 1 m of GT (same drone, no false positive). GT is used only to score; nothing is selected with it.
- **Confidence intervals:** 95% intervals come from a sequence-level cluster bootstrap with 2,000 resamples.

## Main table

| Detector / frames | n frames | n seq | Pearson raw [95% CI] | Pearson, per-seq demeaned [CI] | median \|rel err\| student vs GT | pseudo label vs GT | student vs pseudo label |
|---|---|---|---|---|---|---|---|
| **Final nl075, val (held-out sequences)** | 1,966 | 16 | **0.945** [0.92, 0.96] | **0.938** [0.92, 0.95] | 6.4% | 5.4% | **1.9%** |
| Final nl075, train (fitted frames) | 11,893 | 61 | 0.970 [0.97, 0.97] | 0.966 [0.96, 0.97] | 5.6% | 5.3% | 1.2% |
| 11-07 wa (temporal bg), val held-out | 2,094 | 16 | 0.908 [0.89, 0.93] | 0.901 [0.87, 0.93] | 6.5% | 6.3% | 2.2% |
| 11-07 wa, no-bg inference, val held-out | 2,079 | 16 | 0.890 [0.86, 0.92] | 0.879 [0.83, 0.91] | 6.5% | 6.3% | 2.3% |
| 11-07 wa, train, seed 0 / 1 / 2 (`diag_trainfit` repro) | 11,906 | 61 | 0.971 / 0.969 / 0.968 | 0.967 / 0.966 / 0.965 | 5.8–5.9% | 5.5% | 1.7–1.8% |
| 11-07 wa, train, 3-seed log-mean | 11,902 | 61 | 0.979 | 0.977 | 5.7% | 5.5% | 1.5% |

The per-seed wa rows reproduce the original in-sample 0.97 exactly.

Spearman and the old clipped ±0.4 metric agree to within 0.01–0.03 of the raw Pearson values. All numbers are in `n1_results.json`.

For **wa on val**, the teacher is the 11-07 st2plus rule: ST2 ensemble with temporal background (`final/ens_val.pkl`), plus the V2Pp δ̂, minus 0.75 × the causal global rolling median, seeded with c0 = 0.1308, the val δ̂ median, as in `make_st2plus_dump.py`. wa's student predictions were already on disk (`wa_ens`, `wanb_ens`).

## Control: is it copying, or just shared difficulty?
Any two detectors that look at the same image share some error. The check is whether the student tracks its *pseudo label* more closely than it tracks the *raw teacher detector* the label was derived from.

| Val, held-out | Pearson raw | demeaned | median \|Δ\| |
|---|---|---|---|
| student vs pseudo label | **0.945** | **0.938** | **1.9%** |
| student vs raw ST2 teacher detector (before IR correction) | 0.861 | 0.907 | 5.7% |
| raw ST2 detector vs pseudo label (shares the image and the box by construction) | 0.893 | 0.920 | 5.6% |

The student follows the corrected pseudo label, not just "what any detector sees". It differs from it by 1.9%, against 5.7% from the raw teacher. The training frames show the same pattern: student vs raw ST2 gives r = 0.864.

Some of the correlation is still shared structure. Between-sequence variance is about 25% of the pseudo-label error variance on val. Demeaning by sequence barely changes r (0.945 → 0.938), so the copying happens within sequences, frame by frame, not only through per-sequence offsets.

## Plain-language conclusion
- **It holds out of sample.** On sequences the student never saw, its distance error still moves almost in lockstep with the error its teacher's pseudo label would have:
  - r = 0.945 for the final chain;
  - r = 0.91 for the 11-07 wa detector.
  - Each sequence-level confidence interval stays above 0.87 (above 0.83 for wa's no-background variant).
- **Memorisation adds only a little.** In-sample, r rises to 0.97–0.98 and the student–label gap shrinks from 1.9% to 1.2%.
- **The mechanism is not a tautology.** The student learned the teacher's mapping from image to distance, so it reproduces the teacher's mistakes on new images. That is why its accuracy is capped by the pseudo-label error: 6.4% vs 5.4% on val.

**Suggested wording:** "…reproduces its pseudo labels almost frame by frame (error correlation 0.95 on held-out validation sequences, 0.97 on its training frames)."

## Caveats
- **Held out by sequence, not by scene.** The 16 val sequences share no sequence with the 61 training sequences, but all 4 val scenes (mavic2/02, mavic2/03, phantom4/01, phantom4/02) also appear in training. This is held out by sequence (no neighbouring frames), not by environment.
- **Not all frames are scored.** The frame set requires both detectors to be within 1 m of GT. On val, 1,966 of the 2,110 frames with teacher confidence ≥ 0.5 qualify; misses and false positives are excluded from the correlation.
- **The student is the detector, not the whole chain.** It is the self-trained detector stage, as in the intro claim, not the fused final output with the second stage.
- **Inference only.** The only GPU work was inference: the ST2 ensemble on val (memory fraction 0.15, batch ≤ 4) and the V2Pp δ̂ on val. No training was run and no existing files were modified.

## Files (this folder)
- `run_dump_st2nb_val.py` → `real_val_preds_st2nb.pkl`: teacher detector on val (contains GT, which is used for diagnosis only)
- `val_deltas.py` → `val_deltas.pkl`: V2Pp δ̂ on val for both teacher variants
- `n1_analyze.py` → `n1_results.json`, `n1_analyze.log`
