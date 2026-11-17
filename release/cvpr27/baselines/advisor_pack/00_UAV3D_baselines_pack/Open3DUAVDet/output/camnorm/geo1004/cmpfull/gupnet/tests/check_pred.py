# predict.py 输出格式自检 + 把 boxes9d 角点投影回原图，与 labels.pkl 真值角点投影比较（冒烟模型不准，只看量级/约定）
import sys, pickle, numpy as np
from scipy.spatial.transform import Rotation as Rot
P = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
pred = pickle.load(open(sys.argv[1], 'rb')); lab = pickle.load(open(sys.argv[2], 'rb'))
assert all(p['seq'] == l['seq'] and p['frame'] == l['frame'] for p, l in zip(pred, lab)), '顺序不一致'
print('frames', len(pred), 'keys', sorted(pred[0].keys()), 'boxes9d', pred[0]['boxes9d'].shape, 'scores', pred[0]['scores'].shape)
print('每帧框数', sorted(set(len(p['scores']) for p in pred)), '分数降序', all(np.all(np.diff(p['scores']) <= 0) for p in pred))
def proj(b, K):
    c = (P * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    u = c @ K.T; return u[:, :2] / u[:, 2:]
dc, dpx, dz = [], [], []
for p, l in zip(pred, lab):
    if not l['boxes'] or not len(p['scores']): continue
    g = l['boxes'][0]; b = p['boxes9d'][0]
    gb = np.concatenate([g['center'], g['lwh'], g['euler_xyz']])
    dpx.append(np.abs(proj(b, l['K']).mean(0) - proj(gb, l['K']).mean(0)).max()); dz.append(b[2] - gb[2])
print('top1 投影中心像素差(原图) 中位 %.1f px；深度差 中位 %.2f m' % (np.median(dpx), np.median(dz)))
print('例：pred', np.round(pred[0]['boxes9d'][0], 3), ' gt', np.round(np.concatenate([lab[0]['boxes'][0]['center'], lab[0]['boxes'][0]['lwh'], lab[0]['boxes'][0]['euler_xyz']]), 3))
