#!/bin/bash
# [review1113 sensorfree] wrapper for queue lines that must wait for the sensor-free block: waits for sensorfree/logs/E.done (poll 10 min),
# then runs its arguments unchanged.   usage: bash sf_wait_then.sh <command ...>
# Prepared for heads_seeds / heads_anti. They were already started (waiting) before this block was built, so they are NOT wrapped yet:
# re-queueing them through this wrapper needs their waiting PIDs stopped first (see sensorfree/README.md), which this block never does.
E=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/logs/E.done
n=0; until [ -f $E ]; do [ $n = 0 ] && echo "$(date '+%m-%d %H:%M:%S') waiting for $E: $*"; n=1; sleep 600; done
exec "$@"
