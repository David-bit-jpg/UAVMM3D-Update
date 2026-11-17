# [review1113 copy of srcx/lib/avg_inf.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s10/lib] sys.path -> redirected sx; code unchanged
# [outdoor orig] average per-crop refiner outputs of several seeds (indoor final_unc.py: delta_hat and log b = mean over nets; lum = mean)
import sys, pickle, numpy as np
out, ins = sys.argv[1], sys.argv[2:]
U = [pickle.load(open(f, 'rb')) for f in ins]; R = {}
for k in U[0]:
    if not all(k in u for u in U):
        continue
    lum = [u[k]['lum'] for u in U if u[k]['lum'] is not None]
    R[k] = dict(d=float(np.mean([u[k]['d'] for u in U])), lb=float(np.mean([u[k]['lb'] for u in U])), lum=float(np.mean(lum)) if lum else None)
pickle.dump(R, open(out, 'wb')); print('avg_inf %s: %d keys from %d files' % (out, len(R), len(U)))
