# -*- coding: utf-8 -*-
"""裁窗宽度蒸馏实验（PREREG_cropw.md）第 1 步：把每架无人机周围的方形大窗口裁出来存盘。
窗口边长 L = 5 × 真值画面宽度 w，中心 = 真值 3D 框 8 角点投影的 2D 外接框中心，缩放到 192×192（所以无人机宽度在大窗口里恒为 38.4 像素）。
训练 / 测试时再从大窗口里随机取 s × w（s 随机）的子窗口，网络要回答「无人机宽度占子窗口的几分之几」——窗口尺度随机，答案只能靠看。
    仿真：原分辨率 1280×720 JPEG 缓存（@@MMCACHE@@/full_v2），彩色 + 同位红外；train 每 2 帧取 1、test 全部（4646 帧）
    真实：MAV6D test 缓存（512×288，已去畸变），只有彩色
只保留 8 角点都在画面内、深度 > 0、w ≥ 6 像素的目标。写 cropw/{sim_train,sim_test,real_test}_{rgb,ir}.npy + _meta.pkl。"""
import os
import pickle
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = '@@ROOT@@'
sys.path.insert(0, ROOT)
os.chdir(ROOT + '/tools')
import logging  # noqa: E402
from easydict import EasyDict  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402

OUT = ROOT + '/output/camnorm/geo1004/cropw'
P = 192          # 大窗口边长（像素）
KW = 5.0         # 大窗口边长 = KW × 真值宽度
MIN_W = 6.0


def build(cfgp, split):
    c = EasyDict()
    cfg_from_yaml_file(cfgp, c)
    dc = c.DATA_CONFIG
    dc.DATA_SPLIT['test'] = split
    dc.SAMPLED_INTERVAL['test'] = 1
    ds, _, _ = build_dataloader(dataset_cfg=dc, batch_size=1, dist=False, workers=0, logger=logging.getLogger('x'), training=False)
    return ds


def window(img, cx, cy, L):
    """img (H,W[,C]) 上以 (cx,cy) 为中心、边长 L 的方形窗口 → P×P。缩小时先整体 INTER_AREA 降采样防混叠。"""
    A = P / L
    src = img
    sx = 1.0
    if A < 1.0:
        H, W = img.shape[:2]
        nw, nh = max(8, int(round(W * A))), max(8, int(round(H * A)))
        src = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        sx = nw / float(W)
        cx, cy, L = (cx + 0.5) * sx - 0.5, (cy + 0.5) * (nh / float(H)) - 0.5, L * sx
        A = P / L
    M = np.array([[A, 0, (0.5 - cx + L / 2) * A - 0.5], [0, A, (0.5 - cy + L / 2) * A - 0.5]], np.float64)
    return cv2.warpAffine(src, M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def boxes_2d(boxes, K, seq):
    out = []
    for b in boxes:
        if b[2] <= 1e-6:
            out.append(None)
            continue
        c = (pe.PROTO8 * b[3:6]) @ R.from_euler(seq, b[6:9]).as_matrix().T + b[:3]
        p = (K @ c.T).T
        if (p[:, 2] <= 1e-6).any():
            out.append(None)
            continue
        uv = p[:, :2] / p[:, 2:3]
        out.append((uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()))
    return out


def run(tag, cfgp, split, every, with_ir):
    t0 = time.time()
    ds = build(cfgp, split)
    mm = ds._open()
    seq = getattr(ds, 'euler_seq', 'xyz')
    idx = ds.valid_idx[::every]
    rgbs, irs, meta = [], [], []
    for n, i in enumerate(idx):
        i = int(i)
        m = ds.metas[i]
        K = ds.cache_intrinsic(m)
        if ds.store == 'jpeg':
            o, ln = mm['rgb_jpg_index'][i]
            img = cv2.imdecode(np.asarray(mm['rgb_jpg'][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
            ir = None
            if with_ir:
                o2, l2 = mm['ir_jpg_index'][i]
                ir = cv2.imdecode(np.asarray(mm['ir_jpg'][int(o2):int(o2) + int(l2)]), cv2.IMREAD_UNCHANGED)
                if ir.ndim == 3:
                    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        else:
            img = np.ascontiguousarray(mm['rgb'][i])
            ir = None
        H, W = img.shape[:2]
        boxes = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)
        for k, bb in enumerate(boxes_2d(boxes, K, seq)):
            if bb is None or np.abs(boxes[k]).sum() == 0:
                continue
            u0, u1, v0, v1 = bb
            w, h = u1 - u0, v1 - v0
            if w < MIN_W or u0 < 0 or v0 < 0 or u1 > W - 1 or v1 > H - 1:
                continue
            cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            L = KW * w
            rgbs.append(window(img, cx, cy, L))
            if with_ir:
                irs.append(window(ir, cx, cy, L))
            meta.append(dict(seq=str(m['seq']), frame=str(m['frame']), k=k, w=float(w), h=float(h), z=float(boxes[k, 2]),
                             name=str(m['names'][k]) if k < len(m['names']) else '', W=W, H=H))
        if n % 2000 == 0:
            print(tag, n, '/', len(idx), len(meta), '%.0fs' % (time.time() - t0), flush=True)
    np.save('%s/%s_rgb.npy' % (OUT, tag), np.stack(rgbs))
    if with_ir:
        np.save('%s/%s_ir.npy' % (OUT, tag), np.stack(irs))
    with open('%s/%s_meta.pkl' % (OUT, tag), 'wb') as f:
        pickle.dump(meta, f)
    print(tag, '完成：%d 帧 → %d 个目标，%.0fs' % (len(idx), len(meta), time.time() - t0), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    SIM = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_kdir.yaml'
    REAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
    which = sys.argv[1:] or ['sim_test', 'real_test', 'sim_train']
    if 'sim_test' in which:
        run('sim_test', SIM, 'test', 1, True)
    if 'real_test' in which:
        run('real_test', REAL, 'test', 1, False)
    if 'sim_train' in which:
        run('sim_train', SIM, 'train', 2, True)
