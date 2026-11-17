# -*- coding: utf-8 -*-
"""N1clean: stage-2 depth refiner (ref_st2_s5 / s6 of srcx/in2mav, the networks that produced the chain's pseudo labels) on det0 VAL detections.
Same crop / scalars / forward as srcx/lib/refiner_x.cmd_infer, only: memory fraction 0.15 and batch 4 (eval mode -> batch size does not change outputs).
Also re-infers the first N train frames as a reproduction check against ref/st2_s{5,6}_det0_train.pkl.
Copied 11-14 to review1113/figdata/testgap for TEST split; writes only to the given out path. Inference only.
usage: python n1c_refiner_infer.py <seed 5|6> <split> <out.pkl> [n_first]"""
import os, sys, pickle
os.environ['SX_SRC'] = 'in'; os.environ['SX_TGT'] = 'mav'; os.environ['SX_TEACH'] = 'full'
import numpy as np, torch, cv2
torch.cuda.set_per_process_memory_fraction(0.15, 0)
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import sx  # noqa
import refiner_x as RX  # noqa
BS = 4
seed, split, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
nfirst = int(sys.argv[4]) if len(sys.argv) > 4 else None
ck = sx.O + '/ref/ref_st2_s%d.pt' % seed
dev = torch.device('cuda')
net = RX.Net().to(dev); net.load_state_dict(torch.load(ck, map_location=dev)); net.eval()
recs = pickle.load(open(sx.O + '/recs/det0.pkl', 'rb'))['recs'][split]
if nfirst:
    recs = recs[:nfirst]
res = {}; buf = []


def flush():
    if not buf:
        return
    X = RX.dt.prep(np.stack([b[1] for b in buf]), None, False).to(dev)
    s = torch.from_numpy(np.array([b[2] for b in buf], np.float32)).to(dev)
    with torch.no_grad():
        dh, lb, lg, _ = net.fwd(X, s)
        p = torch.sigmoid(lg.float()).cpu().numpy()
    for (k, v, _), a, b, pp in zip(buf, dh.float().cpu().numpy(), lb.float().cpu().numpy(), p):
        gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)
        msk = pp > 0.5
        res[k] = dict(d=float(a), lb=float(b), lum=float(gray[msk].mean()) if msk.sum() >= 10 else None)
    buf.clear()


for r, m in RX.real_iter(recs, split):
    if r['box'] is None:
        continue
    K = np.asarray(m['K_raw'], np.float64)
    c = RX.det_crop(sx.imread_raw(m), r['box'], K, sx.T.get('wscale', 1.0))
    if c is None:
        continue
    buf.append(((r['seq'], r['frame']), c[0], RX.scal(c[1], K[0, 0], r['box'])))
    if len(buf) == BS:
        flush()
flush()
pickle.dump(res, open(out, 'wb'))
d = np.array([v['d'] for v in res.values()])
print('n1c infer s%d %s: %d crops / %d frames, delta_hat median %+.4f' % (seed, split, len(res), len(recs), np.median(d)), flush=True)
