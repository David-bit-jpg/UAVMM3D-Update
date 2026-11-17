#!/bin/bash
# 11-11 11:5x: out det s1,s2 waited 5 h for a heavy slot; run them directly with --workers 0 (no spawn dataloader -> safe as 3rd job)
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx
bash $X/stage1_det_w0.sh cfgs/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out.yaml 1
bash $X/stage1_det_w0.sh cfgs/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out.yaml 2
