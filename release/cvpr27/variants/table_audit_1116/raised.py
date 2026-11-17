# Threshold-raised rows of tab:ablation row 7 / tab:matched: keep the top x% of TEST frames by PST's own confidence.
import sys, json, numpy as np
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('der', 'derived_lib.py')
from derived_lib import load, met
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'
P0 = G + 'srcx/in2mav/preds/pcmd_fused_test.pkl'; A0 = G + 'srcx/out2antifs/preds/pcmd_fused_test.pkl'
out = {}
for name, pred, split, fracs in [('MAV', P0, 'real_test', (0.696, 0.697, 0.698, 0.699, 0.85)), ('RD', A0, 'outdoor_test', (0.793, 0.794, 0.795, 0.875))]:
    P, M = load(pred, split); c = P['conf']; N = len(M['gt']); qs = np.sort(c[np.isfinite(c)])[::-1]
    for fr in fracs:
        for n in (int(np.floor(fr * N)), int(round(fr * N))):
            th = qs[n - 1]; Q = {k: v.copy() for k, v in P.items()}; drop = ~(c >= th)
            for k in ('pt', 'plwh', 'peul'): Q[k][drop] = np.nan
            Q['conf'][drop] = np.nan; r = met(Q, M, split)
            key = '%s keep %.1f%% (n=%d)' % (name, 100 * fr, n); out[key] = {k: round(v, 3) for k, v in r.items()}
            print(key, out[key], flush=True)
json.dump(out, open('raised.json', 'w'), indent=1)
