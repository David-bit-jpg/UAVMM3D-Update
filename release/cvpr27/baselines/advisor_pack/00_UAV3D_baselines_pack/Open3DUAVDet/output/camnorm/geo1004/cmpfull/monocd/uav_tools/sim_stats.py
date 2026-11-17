# -*- coding: utf-8 -*-
"""Class statistics (single class 'drone') from sim_train, in MonoFlex internal order (l, h, w) = (lwh0, lwh2, lwh1).
EL = center_y + h/2 (see data/datasets/uav.py). Output is pasted into runs/monocd_uav.yaml."""
import pickle, sys
import numpy as np
d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else '@@ROOT@@/data/cmpfull/sim_train/labels.pkl', 'rb'))
B = [b for r in d for b in r['boxes']]
L = np.array([b['lwh'] for b in B]); C = np.array([b['center'] for b in B])
dims = L[:, [0, 2, 1]]
EL = C[:, 1] + dims[:, 1] / 2
print('n_boxes', len(B))
print('DIMENSION_MEAN', dims.mean(0).round(4).tolist(), 'DIMENSION_STD', dims.std(0).round(4).tolist())
print('Y_MEAN', round(EL.mean(), 4), 'Y_STD', round(EL.std(), 4), 'Y3D_RANGE', [round(EL.min(), 2), round(EL.max(), 2)])
print('depth range', C[:, 2].min(), C[:, 2].max())
