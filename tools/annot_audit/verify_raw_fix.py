import sys, os, glob, pickle, zipfile, numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/tools'); sys.path.insert(0, 'E:/Open3DUAVDet')
import build_mm_cache as BMC
from scipy.spatial.transform import Rotation as R
data_root, backup_root = sys.argv[1], sys.argv[2]
combos = sys.argv[3:]
idx = {}
for split in ('train', 'test'):
    for name in ('index_nose.pkl',):
        d = pickle.load(open('E:/mmcache/indoor8jpg/%s/%s' % (split, name), 'rb'))
        for m in d['metas']:
            if m: idx[(m['seq'].replace(chr(92), '/'), m['frame'])] = m
for combo in combos:
    rel = os.path.relpath(combo, data_root).replace(chr(92), '/')
    z = zipfile.ZipFile(os.path.join(backup_root, rel + '.zip'))
    perm_err, n = 0.0, 0
    for arc in z.namelist():
        if not arc.endswith('.pkl') or '/' not in arc: continue
        old = pickle.loads(z.read(arc)); new = pickle.load(open(os.path.join(combo, arc), 'rb'))
        for o, nn in zip(old, new):
            assert o[0] == nn[0]
            co = np.array(o[1:]); cn = np.array(nn[1:])
            perm_err = max(perm_err, np.abs(cn - co[[1, 2, 3, 0, 5, 6, 7, 4]]).max()); n += 1
    di_old = pickle.loads(z.read('drone_info.pkl')); di_new = pickle.load(open(os.path.join(combo, 'drone_info.pkl'), 'rb'))
    # 与 index_nose 对比（缓存按 every 抽帧，只比有的帧）
    worst, cnt = 0.0, 0
    for f in sorted(glob.glob(os.path.join(combo, 'boxes_rgb', '*.pkl'))):
        key = (rel, os.path.basename(f)[:-4] + '.png')
        if key not in idx: continue
        m = idx[key]
        rows = pickle.load(open(f, 'rb'))
        b_raw = {BMC.class_of(r[0]): BMC.corners_to_9params(np.array(r[1:], float).reshape(8, 3)) for r in rows}
        for b, nm in zip(m['boxes9d'], m['names']):
            if nm not in b_raw: continue
            br = b_raw[nm]
            Ri = R.from_euler('xyz', np.asarray(b, float)[6:9]).as_matrix(); Rr = R.from_euler('xyz', br[6:9]).as_matrix()
            worst = max(worst, np.abs(np.asarray(b, float)[:6] - br[:6]).max(), np.abs(Ri - Rr).max()); cnt += 1
    print('%s\n  角点重排误差 %.1e（%d 行）| drone_info 旧 %r -> 新 %r | 与 index_nose 对比 %d 框，最大差 %.1e' % (rel, perm_err, n, di_old.split(chr(10))[0], di_new.split(chr(10))[0], cnt, worst))
