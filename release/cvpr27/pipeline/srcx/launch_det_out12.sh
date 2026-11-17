#!/bin/bash
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx
bash $X/../gpu_slot.sh heavy bash $X/stage1_det.sh cfgs/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out.yaml 1
bash $X/../gpu_slot.sh heavy bash $X/stage1_det.sh cfgs/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out.yaml 2
