#!/bin/bash
# [11-17] RealDrone3D (antifs) lane B: H4 s0,s10,s20 (via run_arm_out.sh)
cd /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
for s in 0 10 20; do bash run_arm_out.sh H4 antifs $s; done
