# -*- coding: utf-8 -*-
"""对手方法自训练用（可选步骤）：把某方法在目标域训练图（无标签）上的预测变成伪标签 labels.pkl（格式同 sim_train/labels.pkl）。
所有对手方法用同一条单帧规则，不读任何真实标签：
  1. 每帧只取分数最高的一个框；
  2. 分数排在该方法所有帧前 KEEP（默认 80%）的才留下（各方法分数刻度不同，所以按比例而不是按绝对阈值）。
（11-10 起不再用前后帧一致性筛选：与我们方法一样，训练素材只看本帧。）
用法：python make_pseudo.py <pred_train.pkl> <out_labels.pkl> --labels <目标域训练集 labels.pkl> [--keep 0.8]"""
import argparse, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from angles import ypr_from_R

ap = argparse.ArgumentParser()
ap.add_argument('pred'); ap.add_argument('out')
ap.add_argument('--labels', required=True)
ap.add_argument('--keep', type=float, default=0.8)
ap.add_argument('--smoke', action='store_true', help='冒烟测试：预测只做了前几帧，标签表截成同长')
a = ap.parse_args()

L = pickle.load(open(a.labels, 'rb'))
P = pickle.load(open(a.pred, 'rb'))
if a.smoke:
    L = L[:len(P)]
assert len(P) == len(L), 'predict.py 的输出应与 labels.pkl 同序同长'
top = []
for r in P:
    sc = np.asarray(r['scores'], np.float64).reshape(-1)
    b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9)
    top.append((b[int(np.argmax(sc))], float(sc.max())) if len(sc) else (None, -np.inf))
scores = np.array([s for _, s in top])
thr = np.quantile(scores[np.isfinite(scores)], 1 - a.keep)
keep = np.array([b is not None and s >= thr for b, s in top])
out = []      # 只写留下的帧；file 写成绝对路径，各方法的数据读取都直接能找到目标域训练图
img_root = os.path.dirname(os.path.abspath(a.labels))
for i, l in enumerate(L):
    if keep[i]:
        b = top[i][0]
        R = Rot.from_euler('xyz', b[6:9]).as_matrix()
        d = dict(l)
        d['file'] = os.path.join(img_root, l['file']).replace('\\', '/'); d['root'] = img_root
        d['boxes'] = [dict(center=b[:3].copy(), lwh=b[3:6].copy(), euler_xyz=b[6:9].copy(), R=R, cls='drone', ypr=ypr_from_R(R))]
        out.append(d)
pickle.dump(out, open(a.out, 'wb'))
print('伪标签：%d / %d 帧（分数阈值 %.4f，保留前 %.0f%%，单帧规则）' % (keep.sum(), len(L), thr, 100 * a.keep))
