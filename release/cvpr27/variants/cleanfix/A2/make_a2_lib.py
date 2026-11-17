# -*- coding: utf-8 -*-
"""[cleanfix A2] copy srcx/lib/chain_x.py -> cleanfix/A2/lib/chain_a2.py with ONE change: beta'_k (b_k) estimated from the unlabeled TRAINING
images (tr1 student readings + train refiner outputs) instead of the unlabeled calibration (val) images. Outputs go to cleanfix/A2/<combo>/.
sx.py is imported from srcx/lib read-only (only constants / paths; its makedirs targets exist)."""
S = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/chain_x.py'
D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A2/lib/chain_a2.py'
t = open(S, encoding='utf-8').read()
def rep(a, b, n=1):
    global t
    assert t.count(a) == n, (a, t.count(a)); t = t.replace(a, b)
rep("O = sx.O\nV = O\n", "O = sx.O\nV = O\nOUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A2/' + sx.COMBO   # [A2] writes only here\nos.makedirs(OUT + '/preds', exist_ok=True)\n")
rep("    ZL = {'val': zL_table('val1', V + '/ou/read_val1_stu.pkl'), 'test': zL_table('te1', V + '/ou/read_te1_stu.pkl')}",
    "    ZL = {'train': zL_table('tr1', V + '/ou/read_tr1_stu.pkl'), 'val': zL_table('val1', V + '/ou/read_val1_stu.pkl'), 'test': zL_table('te1', V + '/ou/read_te1_stu.pkl')}\n    BSRC = os.environ.get('A2_BSRC', 'train')   # [A2] 'val' = deployed rule (reproduction check), 'train' = the A2 variant")
rep("        kpk = {'val': 'val1', 'test': 'te1'}[sp]", "        kpk = {'train': 'tr1', 'val': 'val1', 'test': 'te1'}[sp]")
rep("    for r in D['recs']['val']:\n        if r['box'] is None:\n            continue\n        q, c = base('val', r); z = ZL['val'].get((r['seq'], r['frame']))",
    "    for r in D['recs'][BSRC]:\n        if r['box'] is None:\n            continue\n        q, c = base(BSRC, r); z = ZL[BSRC].get((r['seq'], r['frame']))")
rep("res = dict(w=w, s_bar=sbar.tolist(), col_centres=cc.tolist(), c_k=C, r_k=rk, b_k_val_images=Bv, alpha=ALPHA)",
    "res = dict(w=w, s_bar=sbar.tolist(), col_centres=cc.tolist(), c_k=C, r_k=rk, b_k_source=BSRC, b_k=Bv, n_b=n_b, alpha=ALPHA)")
rep("    Bv = {c: float(np.median(v)) for c, v in Bv.items()}", "    n_b = {c: len(v) for c, v in Bv.items()}\n    Bv = {c: float(np.median(v)) for c, v in Bv.items()}")
rep("    for sp in ('val', 'test'):\n        for tag, ww in (('fused', w), ('nofuse', 0.0)):", "    for sp in ('test',):\n        for tag, ww in (('fused', w),):")
rep("            fn = '%s/preds/%s_%s_%s.pkl' % (V, prefix, tag, sp)", "            fn = '%s/preds/%s_%s_%s.pkl' % (OUT, prefix, tag, sp)")
rep("    json.dump(res, open('%s/deploy_%s.json' % (V, prefix), 'w'), indent=1, default=float)", "    json.dump(res, open('%s/deploy_%s.json' % (OUT, prefix), 'w'), indent=1, default=float)")
open(D, 'w', encoding='utf-8').write('# [cleanfix A2 copy of srcx/lib/chain_x.py] only change: b_k from $A2_BSRC images (default train); writes under cleanfix/A2\n' + t)
print('ok', D)
