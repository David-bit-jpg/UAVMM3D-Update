# -*- coding: utf-8 -*-
"""仿真 LiDAR → 每个机型的机体系点云模板 + 结构关键点 + 实测尺寸，对比 CARLA 标注框（老师建议：LiDAR 抽关键点提升 size → 朝向）。
每帧：tag/depth 边车（原图分辨率，5×5 补洞过）先 5×5 腐蚀回到原打点位置附近 → 像素反投影（平面深度）→ 真值位姿转到机体系（x 前 / y 左 / z 上，米）。
只用单机帧、离该机中心深度 0.75·最大边 + 0.3 m 以内的点。输出 templates.npz + templates.png。"""
import pickle, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']; plt.rcParams['axes.unicode_minus'] = False
B = '@@MMCACHE@@/full_v2/train'
O = '@@ROOT@@/output/camnorm/geo1004/lidar/kpsize'
d = pickle.load(open(B + '/index.pkl', 'rb'))
metas = d['metas']
idx_d, idx_t = np.load(B + '/depth_png_index.npy'), np.load(B + '/tag_png_index.npy')
mm_d = np.memmap(B + '/depth_png.bin', dtype=np.uint8, mode='r')
mm_t = np.memmap(B + '/tag_png.bin', dtype=np.uint8, mode='r')
ker = np.ones((5, 5), np.uint8)
pts, box = {}, {}
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for i in range(0, len(metas), STEP):
    m = metas[i]
    if len(m['names']) != 1 or idx_t[i][1] == 0:
        continue
    t = cv2.imdecode(np.asarray(mm_t[int(idx_t[i][0]):int(idx_t[i][0]) + int(idx_t[i][1])]), cv2.IMREAD_UNCHANGED)
    if t is None or t.sum() == 0:
        continue
    dp = cv2.imdecode(np.asarray(mm_d[int(idx_d[i][0]):int(idx_d[i][0]) + int(idx_d[i][1])]), cv2.IMREAD_UNCHANGED).astype(np.float64) / 100.0
    t = cv2.erode((t > 0).astype(np.uint8), ker) > 0
    g = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)[0]
    L = g[3:6].max()
    vs, us = np.nonzero(t & (dp > 0.05) & (np.abs(dp - g[2]) < 0.75 * L + 0.3))
    if len(us) < 5:
        continue
    K = np.asarray(m['K_raw'], np.float64)
    z = dp[vs, us]
    P = (np.linalg.inv(K) @ np.stack([us + 0.5, vs + 0.5, np.ones(len(us))])) * z
    Xb = Rot.from_euler('xyz', g[6:9]).as_matrix().T @ (P - g[:3, None])
    nm = m['names'][0]
    pts.setdefault(nm, []).append(Xb.T.astype(np.float32))
    box.setdefault(nm, []).append(g[3:6])
out = {}
fig, ax = plt.subplots(2, len(pts), figsize=(3.2 * len(pts), 6.4))
DIAG = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
for c, nm in enumerate(sorted(pts, key=lambda n: np.median(np.array(box[n])[:, 0]))):
    X = np.concatenate(pts[nm])
    lwh = np.median(np.array(box[nm]), 0)
    X = X[np.linalg.norm(X, axis=1) < 0.75 * np.linalg.norm(lwh)]          # 去离群点：离机体中心 > 1.5 × 框半对角线
    bl = np.array(box[nm]); assert np.abs(bl - lwh).max() < 1e-3, (nm, np.abs(bl - lwh).max())   # 同机型标注框尺寸是常数（训练时按尺寸认机型）
    lo, hi = np.percentile(X, 1, 0), np.percentile(X, 99, 0)
    ext = hi - lo
    # 结构关键点：4 个臂端（xy 平面沿对角线最远的 2% 点的质心）、机头/机尾（x 最大/最小 2%）、顶/底
    kp = {}
    for j, dvec in enumerate(DIAG):
        s = X[:, :2] @ dvec
        kp['arm%d' % j] = X[s >= np.percentile(s, 98)].mean(0)
    for key, ax_, sg in (('nose', 0, 1), ('tail', 0, -1), ('top', 2, 1), ('bottom', 2, -1)):
        s = sg * X[:, ax_]
        kp[key] = X[s >= np.percentile(s, 98)].mean(0)
    out[nm] = dict(n_pts=len(X), n_frames=len(pts[nm]), box_lwh=lwh, lidar_ext=ext, lidar_center=(lo + hi) / 2, kp=kp,
                   kp8=np.stack([kp[k] for k in ('arm0', 'arm1', 'arm2', 'arm3', 'nose', 'tail', 'top', 'bottom')]), pts=X[np.random.default_rng(1).choice(len(X), min(len(X), 60000), replace=False)])
    print('%-18s 帧 %4d 点 %7d | 标注框 lwh %.3f %.3f %.3f | LiDAR 实测外廓(1–99%%) %.3f %.3f %.3f | 比值 %.2f %.2f %.2f | 外廓中心偏 %+.3f %+.3f %+.3f' % (
        nm, len(pts[nm]), len(X), *lwh, *ext, *(ext / lwh), *((lo + hi) / 2)))
    sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)]
    for r, (a0, a1, lab) in enumerate(((1, 0, '俯视：横轴 y 左(m)，纵轴 x 前(m)'), (1, 2, '正视：横轴 y 左(m)，纵轴 z 上(m)'))):
        A = ax[r, c]
        A.scatter(sub[:, a0], sub[:, a1], s=0.3, c='#2e86ab', alpha=0.3)
        A.add_patch(plt.Rectangle((-lwh[a0] / 2, -lwh[a1] / 2), lwh[a0], lwh[a1], fill=False, ec='#d1495b', lw=1.2))
        for k, p in kp.items():
            if (r == 0 and k.startswith(('arm', 'nose', 'tail'))) or (r == 1 and k in ('top', 'bottom')):
                A.plot(p[a0], p[a1], 'o', ms=5, mfc='#f18f01', mec='k', mew=0.5)
        A.set_aspect('equal'); A.set_title(('%s\n' % nm) if r == 0 else '', fontsize=9); A.tick_params(labelsize=7)
        if c == 0:
            A.set_ylabel(lab, fontsize=8)
fig.suptitle('仿真 LiDAR 打在机身上的点（蓝，按真值位姿转到机体系）vs 标注框（红）；橙点 = 从点云抽的结构关键点（臂端、机头机尾、顶底）', fontsize=10)
plt.tight_layout(); plt.savefig(O + '/templates.png', dpi=110)
np.save(O + '/templates.npy', out, allow_pickle=True)
