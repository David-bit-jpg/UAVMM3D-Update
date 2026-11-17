# diagnostic: why size_c1 (+0.22) != size_pure (+0.54) on AntiUAV3D -> rotation term log(span(s_gt,R_det)/span(s_gt,R_gt))
import sys, pickle, numpy as np
sys.argv = ['x']; exec(open('a11_size.py', encoding='utf-8').read().split('res = {}')[0])
for dn in DS:
    M = np.load(DS[dn]['meta']); g = M['gt'].astype(float)
    combo, pred, det = DS[dn]['arms']['PST']
    Dd = pickle.load(open(X + combo + '/recs/' + det + '.pkl', 'rb'))
    Rr = {(r['seq'], r['frame']): r for r in Dd['recs']['test'] if r['box'] is not None}
    rot, ang_gt, ang_det, wr = [], [], [], []
    for i in range(len(g)):
        r = Rr.get((str(M['seq'][i]), str(M['frame'][i])))
        if r is None: continue
        Kin = np.asarray(r['K_in'], float); t = g[i, :3]; Rd = np.asarray(r['box'], float)[6:9]
        rot.append(np.log(span(t, g[i, 3:6], Rd, Kin) / span(t, g[i, 3:6], g[i, 6:9], Kin)))
        # tilt of body z axis from camera -y (up) for GT and det
        zg = Rot.from_euler('xyz', g[i, 6:9]).as_matrix()[:, 2]; zd = Rot.from_euler('xyz', Rd).as_matrix()[:, 2]
        ang_gt.append(np.degrees(np.arccos(abs(zg @ np.array([0, -1, 0]))))); ang_det.append(np.degrees(np.arccos(abs(zd @ np.array([0, -1, 0])))))
        wr.append(np.log(span(t, g[i, 3:6], g[i, 6:9], Kin) / (S.box2d(t, g[i, 3:6], g[i, 6:9], Kin)[2] - S.box2d(t, g[i, 3:6], g[i, 6:9], Kin)[0]) * Kin[0, 0] / M['K'][i][0, 0]))
    print(dn, 'rot term median %.3f IQR %s | body-z vs camera-up tilt: GT median %.1f deg, det median %.1f deg | span(K_in)/box2d(K) ratio log median %.3f'
          % (np.median(rot), np.percentile(rot, [25, 75]).round(3), np.median(ang_gt), np.median(ang_det), np.median(wr)))
