#!/usr/bin/env bash
# camnorm-v1 缓存：MAV6D（去畸变 + 512x288，train/val/test）与新室内仿真（RGB-only，真内参，机体 z 朝上）。
# 顺序执行（两个都写 E 盘，并发只会让机械盘来回寻道）。每步有产物即跳过，可重跑续做。
set -u
PY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
cd /e/Open3DUAVDet/tools || exit 1
LOG=/e/Open3DUAVDet/output/camnorm/logs
mkdir -p "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/caches.log"; }

MAV=E:/mmcache/mav6d_cn
SIM=E:/mmcache/indoor8cn

if [ ! -f "$MAV/test/index.pkl" ] || [ ! -f "$MAV/norm.txt" ]; then
  say "建 MAV6D 缓存 -> $MAV"
  "$PY" build_mav6d_cache.py --root E:/MAV6D --out "$MAV" --workers 8 > "$LOG/cache_mav6d.log" 2>&1
  say "  rc=$? $(grep -E '^\[(train|val|test)\]' $LOG/cache_mav6d.log | tr '\n' ' ')"
fi

for sp in train test; do
  if [ ! -f "$SIM/$sp/index.pkl" ]; then
    say "建仿真缓存 $sp -> $SIM"
    "$PY" build_mm_cache.py --list cfgs/subsets/indoor8/near_$sp.txt --split $sp --root D:/data_collect \
        --out "$SIM" --every 3 --max-label-range 40 --workers 8 --rgb-only --intrinsic auto \
        > "$LOG/cache_sim_$sp.log" 2>&1
    say "  rc=$? $(grep -E '^完成' $LOG/cache_sim_$sp.log)"
  fi
done

if [ ! -f "$SIM/train/vis_score.npy" ] || [ ! -f "$SIM/test/vis_score.npy" ]; then
  say "仿真可见度分"
  "$PY" mm_vis_score.py --cache "$SIM" --splits train test > "$LOG/vis_sim.log" 2>&1
  say "  rc=$? $(head -3 $LOG/vis_sim.log | tr '\n' ' ')"
fi

if [ ! -f "$SIM/norm.txt" ]; then
  say "仿真归一化统计量"
  "$PY" - > "$SIM/norm.txt" <<'PYEOF'
import numpy as np, pickle
a = np.load('E:/mmcache/indoor8cn/train/rgb.npy', mmap_mode='r')
idx = pickle.load(open('E:/mmcache/indoor8cn/train/index.pkl', 'rb'))['valid_idx']
sel = idx[np.linspace(0, len(idx) - 1, min(600, len(idx))).astype(int)]
s1 = np.zeros(3); s2 = np.zeros(3); n = 0
for i in sel:
    v = a[i].astype(np.float64).reshape(-1, 3) / 255.0
    s1 += v.sum(0); s2 += (v * v).sum(0); n += v.shape[0]
m = s1 / n; sd = np.sqrt(np.maximum(s2 / n - m * m, 0))
print('NORM_MEAN: [%.4f, %.4f, %.4f]' % tuple(m))
print('NORM_STD: [%.4f, %.4f, %.4f]' % tuple(sd))
print('frames %d' % len(sel))
PYEOF
fi
say "MAV6D $(tr '\n' ' ' < $MAV/norm.txt)"
say "SIM   $(tr '\n' ' ' < $SIM/norm.txt)"
touch "$LOG/CACHES_READY"
say "ALL DONE"
