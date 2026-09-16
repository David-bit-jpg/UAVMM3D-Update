# -*- coding: utf-8 -*-
"""给「从已修正原始数据建的」仿真缓存生成旧约定标签索引 index_labelx.pkl（逆变换 R·Rz(+90°)，l/w 互换），做对照实验用。

并与正式缓存 E:/mmcache/indoor8jpg 的原版索引（index_labelx_v1.pkl，修正前建的）按 (seq, frame) 逐框核对：
  - 本缓存 index.pkl  == indoor8jpg index_nose.pkl
  - 本缓存 index_labelx.pkl == indoor8jpg index_labelx_v1.pkl
    python tools/annot_audit/make_labelx_index.py --cache E:/mmcache/tiny_nosex
"""
import argparse
import copy
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R


def inv_fix(b):
    b = np.asarray(b, np.float64).copy()
    Rm = R.from_euler('xyz', b[6:9]).as_matrix() @ R.from_euler('z', 90, degrees=True).as_matrix()
    b[6:9] = R.from_matrix(Rm).as_euler('xyz')
    b[3], b[4] = b[4], b[3]
    return b


def box_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    Ra, Rb = R.from_euler('xyz', a[6:9]).as_matrix(), R.from_euler('xyz', b[6:9]).as_matrix()
    return max(np.abs(a[:6] - b[:6]).max(), np.abs(Ra - Rb).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ref', default='E:/mmcache/indoor8jpg')
    args = ap.parse_args()
    for split in ('train', 'test'):
        idx = pickle.load(open(os.path.join(args.cache, split, 'index.pkl'), 'rb'))
        assert str(idx.get('raw_label_fix', '')).startswith('nose-x'), '缓存不是从已修正原始数据建的：%s' % idx.get('raw_label_fix')
        old = copy.deepcopy(idx)
        for m in old['metas']:
            if m and len(m['boxes9d']):
                m['boxes9d'] = np.stack([inv_fix(b) for b in m['boxes9d']]).astype(np.float32)
        old['raw_label_fix'] = 'inverted-to-labelx (对照实验用)'
        old['label_fix'] = {'version': 'labelx-inverse-of-nose-x-v1', 'nose': False, 'scale': False}
        with open(os.path.join(args.cache, split, 'index_labelx.pkl'), 'wb') as f:
            pickle.dump(old, f, protocol=4)
        # 核对
        refs = {}
        for name, key in (('index_nose.pkl', 'nose'), ('index_labelx_v1.pkl', 'labelx')):
            p = os.path.join(args.ref, split, name)
            if os.path.exists(p):
                refs[key] = {(m['seq'], m['frame']): m for m in pickle.load(open(p, 'rb'))['metas'] if m}
        for key, mine in (('nose', idx), ('labelx', old)):
            if key not in refs:
                print('%s: 参照 %s 不存在，跳过核对' % (split, key))
                continue
            worst, n, nframes = 0.0, 0, 0
            for m in mine['metas']:
                if not m:
                    continue
                r = refs[key].get((m['seq'], m['frame']))
                if r is None or len(r['boxes9d']) != len(m['boxes9d']):
                    continue
                nframes += 1
                for a, b in zip(m['boxes9d'], r['boxes9d']):
                    worst = max(worst, box_diff(a, b)); n += 1
            print('%s: 本缓存 %s 标签 vs 正式缓存 %s：%d 帧 %d 框，最大差 %.1e' % (split, key, key, nframes, n, worst))


if __name__ == '__main__':
    main()
