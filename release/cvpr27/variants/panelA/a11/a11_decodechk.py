# does the detector's depth satisfy span(det box: s_bar, R_det, t_det; K_in) == w2d_hat ?  (decode identity)
import pickle, numpy as np
exec(open('a11_size.py', encoding='utf-8').read().split('res = {}')[0])
for dn in DS:
    for an in ('PST', 'Sim-only'):
        combo, pred, det = DS[dn]['arms'][an]
        Dd = pickle.load(open(X + combo + '/recs/' + det + '.pkl', 'rb')); sb = np.asarray(Dd['size'], float)
        v = []; v2 = []
        for r in Dd['recs']['test'][::5]:
            if r['box'] is None: continue
            b = np.asarray(r['box'], float); Kin = np.asarray(r['K_in'], float)
            v.append(np.log(span(b[:3], sb, b[6:9], Kin) / r['w2d'])); v2.append(np.log(span(b[:3], b[3:6], b[6:9], Kin) / r['w2d']))
        print(dn, an, 'log span(s_bar,R_det,t_det)/w2d_hat: median %.4f p5 %.4f p95 %.4f | with box lwh %.4f' % (np.median(v), *np.percentile(v, [5, 95]), np.median(v2)))
