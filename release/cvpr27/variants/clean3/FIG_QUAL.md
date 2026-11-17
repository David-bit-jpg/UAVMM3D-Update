# Fig. 3: qualitative results (clean MAV6D chain + AntiUAV3D main run)

- PDF: `D:/CVPR27_UAV/figures/fig_qual.pdf`. Single column, 3.33 × 3.80 in. The 18 raster crops are embedded; boxes, text and legend are vector.
- PNG copy: `review1113/clean3/fig_qual.png`, 400 dpi.
- Scripts: `qual/select_qual.py`, which writes `selection.json`, and `qual/make_fig_qual.py`.
- Colours: magenta = PST prediction, cyan = ground-truth 3D box. Both are the 8 projected corners and 12 edges, drawn with K_raw on the undistorted raw frame.
- Each panel shows the airframe, the GT range, the distance error Δd = ‖t_pred − t_gt‖ and AOE. AOE is the quadrotor-symmetric geodesic error used by the scorer. AntiUAV3D panels also give Δd relative to range, which is the scorer's ATE unit.
- AntiUAV3D panels carry a full-frame thumbnail. The yellow rectangle marks the crop, so the drone's true size is visible.

**How the examples were picked.** The rule was fixed in code before rendering, and no image was looked at.
- Pool: TP frames of the test set only (top-scoring box, projected 2D IoU ≥ 0.1, as in the scorer). MAV6D has 4,616 / 4,800; AntiUAV3D has 2,412 / 2,624. Missed frames have no error to annotate.
- Position error and AOE are each converted to percentile ranks p_pos and p_aoe.
- Success: p_pos ≤ 0.15 and p_aoe ≤ 0.15. Three frames are taken at the 1/6, 1/2 and 5/6 range quantiles of that pool.
- Typical: the frame with both ranks closest to 0.5, i.e. it minimises max |p − 0.5|.
- Position failure: the range-median frame with p_pos ≥ 0.95.
- Orientation failure: the range-median frame with p_aoe ≥ 0.95.
- Sequences: MAV6D prefers distinct sequences. AntiUAV3D test has only 2 sequences (Matrice 4E 2,576 frames, Matrice 400 48 frames), so the rule there only keeps picks ≥ 200 frames apart.
- For reference, the TP medians are 0.20 m / 11.7° on MAV6D and 7.8 % of range / 15.2° on AntiUAV3D. The typical panels (d) and (j) sit at these medians.

## Panel descriptions
**MAV6D test (indoor)**
- (a) Success. MAV6D Phantom 4 at 2.5 m: position error 0.05 m, AOE 4° (seq phantom4/01/0116, frame 232).
- (b) Success. MAV6D Phantom 4 at 3.0 m: position error 0.01 m, AOE 5° (phantom4/01/0123, frame 23).
- (c) Success. MAV6D Phantom 4 at 3.7 m: position error 0.03 m, AOE 6° (phantom4/02/0201, frame 15).
- (d) Typical (median errors). MAV6D Mavic 2 at 3.5 m: position error 0.21 m, AOE 12° (mavic2/02/0206).
- (e) Position failure. MAV6D Mavic 2 at 4.1 m: position error 0.90 m (22 % of range), AOE 39°. The 2D box overlaps the drone but the distance is badly overestimated, and the orientation is also off (mavic2/02/0211).
- (f) Orientation failure. MAV6D Phantom 4 at 3.5 m, banked and motion-blurred: position error 0.18 m, AOE 44° (phantom4/02/0210, frame 161).

**AntiUAV3D test (outdoor)**
- (g) Success. AntiUAV3D Matrice 4E at 19 m: position error 0.33 m (1.7 % of range), AOE 8° (train_seq_0010, frame 011142).
- (h) Success. AntiUAV3D Matrice 4E at 38 m: position error 0.51 m (1.3 %), AOE 8° (train_seq_0010, 007822).
- (i) Success. AntiUAV3D Matrice 4E at 46 m: position error 0.97 m (2.1 %), AOE 8° (train_seq_0010, 008079).
- (j) Typical (median errors). AntiUAV3D Matrice 4E at 46 m: position error 3.6 m (7.8 %), AOE 16°. The predicted box is also visibly too large, consistent with the outdoor ASE of 16 cm (train_seq_0010, 000420).
- (k) Position failure. AntiUAV3D Matrice 400 at 38 m: position error 21.6 m (58 % of range), AOE 22°. This airframe, from the 48-frame sequence, is about 2.5× larger than the predicted size (GT 0.98 × 0.76 m vs predicted 0.56 m). As a result, the range comes out at 15.9 m instead of 37.5 m (train_seq_0022, 000115).
- (l) Orientation failure. AntiUAV3D Matrice 4E at 46 m: position error 3.1 m (6.7 %), AOE 43° (train_seq_0010, 003791).

Airframe names: AntiUAV3D codes M4E and M400 are written as DJI Matrice 4E and Matrice 400. The GT box of M4E, 0.307 × 0.3875 × 0.1495 m, matches the Matrice 4E's published dimensions.
