import sys, pickle, numpy as np
exec(open('a11_size.py', encoding='utf-8').read().split('res = {}')[0])
for dn in DS:
    M = np.load(DS[dn]['meta']); g = M['gt'].astype(float); key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    combo, pred, det = DS[dn]['arms']['PST']; P = load(X + combo + '/preds/' + pred + '.pkl', M, key)
    Dd = pickle.load(open(X + combo + '/recs/' + det + '.pkl', 'rb'))
    Rr = {(r['seq'], r['frame']): r for r in Dd['recs']['test'] if r['box'] is not None}
    a, b, ry, rr = [], [], [], []
    for i in range(0, len(g), 3):
        r = Rr.get((str(M['seq'][i]), str(M['frame'][i])))
        if r is None: continue
        Kin = np.asarray(r['K_in'], float); t = g[i, :3]; Rd = np.asarray(r['box'], float)[6:9]
        sg = span(t, g[i, 3:6], g[i, 6:9], Kin)
        a.append(np.log(span(t, g[i, 3:6], Rd, Kin) / sg)); b.append(np.log(span(t, g[i, 3:6], P['peul'][i], Kin) / sg))
        E = Rot.from_euler('xyz', g[i, 6:9]).as_matrix().T @ Rot.from_euler('xyz', Rd).as_matrix()
        ry.append(np.degrees(np.arctan2(E[1, 0], E[0, 0])) % 90)
        # GT rotated to a "view-aligned" yaw: span of GT box with GT yaw replaced by det yaw only? report GT span vs l,w extent
        rr.append(np.log(sg / (Kin[0, 0] * np.hypot(*g[i, 3:5]) / t[2])))
    ry = np.array(ry)
    print(dn, 'rot term det R %.3f | final (student) R %.3f | rel yaw mod 90 hist(0-15,15-30,30-45,45-60,60-75,75-90):' % (np.median(a), np.median(b)),
          np.histogram(ry, bins=[0, 15, 30, 45, 60, 75, 90])[0], '| log span_gt / (fx*diag/z) median %.3f' % np.median(rr))
