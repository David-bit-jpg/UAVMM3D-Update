# -*- coding: utf-8 -*-
"""给仿真缓存写修正后的索引（不动图像、不覆盖原 index.pkl）。

    cd E:/Open3DUAVDet/tools && python fix_sim_labels.py --roots E:/mmcache/indoor8jpg E:/mmcache/indoor8cn

每个 split 目录写两份：
    index_nose.pkl       只修机头偏航（R·Rz(-90°)，l/w 互换）
    index_nosescale.pkl  机头 + 物理尺度（t/k、lwh/k）
数据集用 DATA_CONFIG.INDEX_FILE 选。修正表与证据在 uavdet3d/utils/sim_asset_fix.py。
写完做自检：旋转行列式、机头方向 = 旧 -y、l/w 互换、缩放前后 8 角点投影逐像素一致。
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.utils import sim_asset_fix as F   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def corners(b):
    return (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def selfcheck(old, new_n, new_ns, name):
    Ro = R.from_euler('xyz', old[6:9]).as_matrix()
    Rn = R.from_euler('xyz', new_n[6:9]).as_matrix()
    assert abs(np.linalg.det(Rn) - 1) < 1e-4
    assert np.abs(Rn[:, 0] - (-Ro[:, 1])).max() < 1e-4, '新 x 应等于旧 -y'
    assert np.abs(Rn[:, 1] - Ro[:, 0]).max() < 1e-4, '新 y 应等于旧 x'
    assert np.abs(Rn[:, 2] - Ro[:, 2]).max() < 1e-4
    assert abs(new_n[3] - old[4]) < 1e-5 and abs(new_n[4] - old[3]) < 1e-5 and abs(new_n[5] - old[5]) < 1e-5
    # 机头修正不改变 8 角点集合（只是换了哪条边叫 x）
    co, cn = corners(old), corners(new_n)
    d = max(np.linalg.norm(co - cn[k], axis=1).min() for k in range(8))
    assert d < 1e-3, '机头修正后角点集合变了 %.2e' % d
    # 缩放：投影逐像素一致（归一化平面坐标）
    cs = corners(new_ns)
    k = F.SCALE_SIM_OVER_REAL[F.base_name(name)]
    assert np.abs(cs * k - cn).max() < 1e-3
    pn = cn[:, :2] / cn[:, 2:3]
    ps = cs[:, :2] / cs[:, 2:3]
    return float(np.abs(pn - ps).max()), float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', default=['E:/mmcache/indoor8jpg', 'E:/mmcache/indoor8cn'])
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    args = ap.parse_args()
    for root in args.roots:
        for split in args.splits:
            p = os.path.join(root, split, 'index.pkl')
            idx = pickle.load(open(p, 'rb'))
            # 2026-09-15 起原始数据已就地修正机头（tools/fix_raw_nose_labels.py），index.pkl 也换成了机头修正版：
            # 再在这上面套一次 −90° 会转两次
            if idx.get('label_fix') or str(idx.get('raw_label_fix', '')).startswith('nose-x'):
                raise SystemExit('%s 的标签已经是机头修正版（label_fix=%s raw_label_fix=%s），拒绝重复旋转'
                                 % (p, idx.get('label_fix'), idx.get('raw_label_fix')))
            out = {'nose': copy.deepcopy(idx), 'nosescale': copy.deepcopy(idx)}
            worst_px, worst_c, n = 0.0, 0.0, 0
            for i, m in enumerate(idx['metas']):
                if not m or len(m['boxes9d']) == 0:
                    continue
                old = np.asarray(m['boxes9d'])
                bn = F.apply_fix(old, m['names'], nose=True, scale=False)
                bns = F.apply_fix(old, m['names'], nose=True, scale=True)
                out['nose']['metas'][i]['boxes9d'] = bn
                out['nosescale']['metas'][i]['boxes9d'] = bns
                for j in range(len(old)):
                    px, c = selfcheck(old[j].astype(np.float64), bn[j].astype(np.float64), bns[j].astype(np.float64), m['names'][j])
                    worst_px, worst_c, n = max(worst_px, px), max(worst_c, c), n + 1
            for tag, d in out.items():
                d['label_fix'] = {'version': F.FIX_VERSION, 'nose': True, 'scale': tag == 'nosescale',
                                  'nose_yaw_deg': dict(F.NOSE_YAW_DEG),
                                  'scale_sim_over_real': dict(F.SCALE_SIM_OVER_REAL) if tag == 'nosescale' else None,
                                  'source_index': p}
                dst = os.path.join(root, split, 'index_%s.pkl' % tag)
                with open(dst + '.tmp', 'wb') as f:
                    pickle.dump(d, f, protocol=4)
                os.replace(dst + '.tmp', dst)
            print('%s/%s: %d 框；机头修正角点集合最大差 %.1e m；缩放前后归一化投影最大差 %.1e -> index_nose.pkl / index_nosescale.pkl'
                  % (root, split, n, worst_c, worst_px), flush=True)


if __name__ == '__main__':
    main()
