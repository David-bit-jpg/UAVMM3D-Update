# -*- coding: utf-8 -*-
"""自训练（PREREG_selftrain.md）第 2 步：真实训练集无标签裁窗。P0_s{j} 每帧最高分检测，置信度 ≥ 0.5、投影宽 ∈ [6, 170]，
与 dc / softkd 同一取窗 → 128×128 彩色裁窗 + 5 个标量。不读任何标签。输出 selftrain/real_s{j}_{rgb,scal}.npy。"""
import os
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import cropw_extract as ce  # noqa: E402
import silw_train as sw  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))


def main():
    b = '@@MMCACHE@@/mav6d_cn_obj/train'
    arr = np.load(b + '/rgb.npy', mmap_mode='r')
    d = pickle.load(open(b + '/index.pkl', 'rb'))
    key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
    os.makedirs(G + '/selftrain', exist_ok=True)
    for j in [int(a) for a in (sys.argv[1:] or ['3', '4'])]:
        recs = pickle.load(open('%s/irloc_train_recs/train_P0_s%d.pkl' % (G, j), 'rb'))['records']
        views, scal = [], []
        for r in recs:
            if len(r['pred']) == 0:
                continue
            pi = int(np.argmax(r['conf']))
            if r['conf'][pi] < 0.5:
                continue
            i = key[(r['seq_id'], r['frame_id'])]
            K = np.asarray(d['metas'][i]['K_in'], np.float64)
            g = r['pred'][pi]
            if g[2] <= 1e-3:
                continue
            c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (6.0 <= w <= 170.0):
                continue
            img = np.ascontiguousarray(arr[i])
            views.append(sw.view_img(ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
        np.save('%s/selftrain/real_s%d_rgb.npy' % (G, j), np.stack(views))
        np.save('%s/selftrain/real_s%d_scal.npy' % (G, j), np.array(scal, np.float32))
        print('j=%d：%d 帧 → %d 个裁窗' % (j, len(recs), len(views)), flush=True)


if __name__ == '__main__':
    main()
