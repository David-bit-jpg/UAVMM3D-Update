# -*- coding: utf-8 -*-
"""[review1113 C1b] re-run (read-only copy of) wf23/mirror/boot_ci.py for the paper's MAV6D UDS intervals, writing only here;
reports #sequences / #frames resampled and which B (300 or 500) reproduces the stored interval."""
import sys, pickle, json
import numpy as np
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; HERE = G + '/gproto/wf23/mirror'
sys.path.insert(0, G + '/cmpeval')
import paper_table as PT  # noqa
PAIRS = [('pos_rek_w00', 'pos_z3S6Ik_w00'), ('lad1_w00', 'lad2_w00'), ('lad2_w00', 'simpk_w00'), ('simpk_w00', 'simpk_w15')]
out = {}
for a, b in PAIRS:
    stored = json.load(open('%s/boot_%s__%s.json' % (HERE, a, b)))
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
        Pa = pickle.load(open('%s/fe_%s_%s_P.pkl' % (HERE, a, sp), 'rb')); Pb = pickle.load(open('%s/fe_%s_%s_P.pkl' % (HERE, b, sp), 'rb'))
        valid = Pa['valid']; seq = np.array([str(s) for s in M['seq']])[valid]; us = np.unique(seq); groups = [valid[seq == s] for s in us]
        res = {}
        for B in (300, 500):
            rng = np.random.default_rng(0); d = []
            for _ in range(B):
                pick = rng.integers(0, len(us), len(us)); sel = np.concatenate([groups[i] for i in pick])
                d.append(PT.metrics(Pb['P'], M, sel)['UDS'] - PT.metrics(Pa['P'], M, sel)['UDS'])
            res[B] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        st = [stored[sp]['UDS']['lo'], stored[sp]['UDS']['hi']]
        match = [B for B, r in res.items() if np.allclose(r, st, atol=1e-6)]
        out['%s->%s %s' % (a, b, sp)] = dict(n_seq=len(us), n_frames=int(len(valid)), stored=st, B300=res[300], B500=res[500], B_matching=match)
        print('%s -> %s %s: seq %d frames %d stored [%+.3f,%+.3f] B300 [%+.3f,%+.3f] B500 [%+.3f,%+.3f] match %s' % (a, b, sp, len(us), len(valid), *st, *res[300], *res[500], match), flush=True)
json.dump(out, open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/c1b_boot_check.json', 'w'), indent=1)
