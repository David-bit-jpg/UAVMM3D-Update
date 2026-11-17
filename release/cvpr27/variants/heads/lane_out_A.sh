#!/bin/bash
# [11-17] RealDrone3D (antifs) lane A: H6 s0,s10,s20 then H5 s0,s10,s20 (via run_arm_out.sh)
cd /e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
for x in "H6 0" "H6 10" "H6 20" "H5 0" "H5 10" "H5 20"; do set -- $x; bash run_arm_out.sh $1 antifs $2; done
