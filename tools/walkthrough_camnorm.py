# -*- coding: utf-8 -*-
"""手动走一遍 camnorm 训练链路：每一步用【独立重算】对比管线里的值，打印 PASS/FAIL 与数字。

    cd E:/Open3DUAVDet/tools && python walkthrough_camnorm.py

A  仿真 原始标注(boxes_rgb 8 角点) -> 缓存 boxes9d：手向 det(R)=+1、机体 z·世界上>0、由 9 参数重建的角点集合==原始角点、
   图像像素==原图缩放、K_in==真内参缩放、JPEG 缓存解码==原图
B  MAV6D 原始标签(txt 四元数) -> 缓存：R,t==read_truth_Rt、官方 util.py 的畸变投影关键点 vs 我们的投影（含中心偏移量化）、
   去畸变图像像素==缓存、机体 z 朝上
C  数据集取样 -> 监督图：热力图格子/亚像素偏移/虚拟深度/尺寸/视线相对旋转 全部独立重算；解码器还原真值
D  真 DataLoader spawn worker（训练增广开）：worker 产出的监督图解码 == 同一 worker 给出的 gt；评测 loader 与主进程逐位一致
E  单批过拟合（真实训练用的 bf16 autocast + Adam）：300 步后解码 == 真值 -> 损失/解码/梯度链路正确
F  checkpoint：train_utils.save_checkpoint 存 -> load_params_from_file 载 -> 输出逐位一致；权重 float32；best.json 的轮次 == 验证分数最高轮
G  评测：把真值当预测送进 eval_camnorm.ads_eval -> 6dof 误差 0、ADS 100
H  训练实际用到的帧：SAMPLED_INTERVAL/TRAIN_REPEAT 的抽样是否均匀覆盖所有序列
"""
import glob
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import camera_geometry as cg, common_utils   # noqa: E402
from uavdet3d.utils.object_encoder_mav6d import center_point_decoder   # noqa: E402
from uavdet3d.utils.rotation_repr import euler_to_vec   # noqa: E402
import build_mm_cache as BMC   # noqa: E402
import build_mav6d_cache as BMV   # noqa: E402

OUT = os.path.join(ROOT, 'output', 'camnorm', 'walkthrough')
os.makedirs(OUT, exist_ok=True)
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
C2O = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
SIM_CFG = os.environ.get('WALK_SIM_CFG', 'cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml')
MAV_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
RESULTS = []
rng = np.random.RandomState(0)


def check(name, ok, msg):
    RESULTS.append((name, bool(ok)))
    print('[%s] %-32s %s' % ('PASS' if ok else 'FAIL', name, msg), flush=True)


def corners_of(b):
    return (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def load_cfg(path, sets=None):
    cfg = EasyDict()
    cfg_from_yaml_file(path, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    return cfg


# =============================================================================
def stage_A():
    idx = pickle.load(open('E:/mmcache/indoor8cn/train/index.pkl', 'rb'))
    rgb = np.load('E:/mmcache/indoor8cn/train/rgb.npy', mmap_mode='r')
    jidx = pickle.load(open('E:/mmcache/indoor8jpg/train/index.pkl', 'rb'))
    jbin = np.memmap('E:/mmcache/indoor8jpg/train/rgb_jpg.bin', dtype=np.uint8, mode='r')
    jix = np.load('E:/mmcache/indoor8jpg/train/rgb_jpg_index.npy')
    jmap = {(m['seq'], m['frame']): i for i, m in enumerate(jidx['metas']) if m}
    picks = rng.choice(idx['valid_idx'], 16, replace=False)
    d_p9, d_corner, dets, zups, d_img, d_K, d_jpg, d_jK, n_box = [], [], [], [], [], [], [], [], 0
    for i in picks:
        m = idx['metas'][int(i)]
        base = os.path.join('D:/data_collect', m['seq'])
        raw = pickle.load(open(os.path.join(base, 'boxes_rgb', os.path.splitext(m['frame'])[0] + '.pkl'), 'rb'))
        info = pickle.load(open(os.path.join(base, 'im_info.pkl'), 'rb'))
        K_leg = np.array(info['rgb']['intrinsic'], dtype=np.float64)
        K_true, fixed = cg.legacy_sim_intrinsic_fix(K_leg, 1280, 720)
        assert fixed
        E = np.array(info['rgb']['extrinsic'], dtype=np.float64)
        up_cam = (C2O @ np.linalg.inv(E) @ np.array([0, 0, 1.0, 0]))[:3]
        rows = []
        for row in raw:
            name = row[0] if isinstance(row[0], str) else '?'
            if BMC.class_of(name) is None:
                continue
            c = np.array(row[1:], dtype=np.float64).reshape(8, 3)
            rows.append((BMC.class_of(name), c, BMC.corners_to_9params(c)))
        for b in m['boxes9d']:
            b = b.astype(np.float64)
            # 按中心匹配原始行
            j = int(np.argmin([np.linalg.norm(p9[:3] - b[:3]) for _, _, p9 in rows]))
            cls, c, p9 = rows[j]
            d_p9.append(np.abs(p9 - b).max())
            rec = corners_of(b)
            d_corner.append(max(np.linalg.norm(c - rec[k], axis=1).min() for k in range(8)))   # 集合意义上的角点重合
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            dets.append(np.linalg.det(Rm))
            zups.append(float(Rm[:, 2] @ up_cam))
            # 原始角点三条边是左手系：检查我们翻的是宽度轴（机体 y），x 仍是 c1-c0 方向
            ex = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
            d_corner.append(0.0 if abs(ex @ Rm[:, 0] - 1) < 1e-4 else 9.0)
            n_box += 1
        img = cv2.imread(os.path.join(base, 'images_rgb', m['frame']), cv2.IMREAD_COLOR)
        small = cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA)
        d_img.append(np.abs(small.astype(int) - np.asarray(rgb[int(i)]).astype(int)).max())
        d_K.append(np.abs(cg.scale_K(K_true, 512 / 1280, 288 / 720) - np.asarray(m['K_in'])).max())
        jj = jmap[(m['seq'], m['frame'])]
        o, ln = jix[jj]
        dec = cv2.imdecode(np.asarray(jbin[int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        d_jpg.append(np.abs(dec.astype(int) - img.astype(int)).mean())
        d_jK.append(np.abs(np.asarray(jidx['metas'][jj]['K_in']) - K_true).max())
    check('A1 sim 9参数==重算', max(d_p9) < 1e-5, '%d 框，最大差 %.1e' % (n_box, max(d_p9)))
    check('A2 sim 角点集合重合', max(d_corner) < 1e-3, '由 9 参数重建的 8 角点与原始角点最大距离 %.1e m（含 x=c1-c0 方向检查）' % max(d_corner))
    check('A3 sim 右手系', min(dets) > 0.999, 'det(R) 范围 %.6f~%.6f' % (min(dets), max(dets)))
    check('A4 sim 机体z朝上', min(zups) > 0.5, 'z·世界上 范围 %.3f~%.3f' % (min(zups), max(zups)))
    check('A5 sim 缓存图像==原图缩放', max(d_img) == 0, '16 帧最大像素差 %d' % max(d_img))
    check('A6 sim K_in==真内参缩放', max(d_K) < 1e-9, '最大差 %.1e' % max(d_K))
    check('A7 sim JPEG缓存==原图', max(d_jpg) < 6.0 and max(d_jK) < 1e-9, 'JPEG(q95, 有损) 平均像素差 最大 %.2f 灰度级；K_in 差 %.1e' % (max(d_jpg), max(d_jK)))


# =============================================================================
def stage_B():
    from uavdet3d.datasets.mav6d.mav6d_utils import read_truth_Rt
    K, D = BMV.K_CALIB, BMV.D_CALIB
    m1, m2, K_pin = cg.undistort_maps(K, D, (1920, 1080))
    idx = pickle.load(open('E:/mmcache/mav6d_cn/test/index.pkl', 'rb'))
    rgb = np.load('E:/mmcache/mav6d_cn/test/rgb.npy', mmap_mode='r')
    up_cam = BMV.CAMERA2VICON[:3, :3] @ np.array([0, 0, 1.0])
    util = None
    try:
        spec = importlib.util.spec_from_file_location('mav6d_util', 'E:/MAV6D/util.py')
        util = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(util)
    except Exception as e:
        print('   （官方 util.py 无法导入：%r，官方投影用自己复现的公式）' % (e,))
    picks = rng.choice(idx['valid_idx'], 12, replace=False)
    d_Rt, dets, zups, d_img, d_K, off_px, off_body, d_proj = [], [], [], [], [], [], [], []
    for i in picks:
        m = idx['metas'][int(i)]
        cls, scene, sq = m['seq'].split('/')
        lp = os.path.join('E:/MAV6D', cls, 'labels', scene, sq, os.path.splitext(m['frame'])[0] + '.txt')
        Rm9, t = read_truth_Rt(lp)
        Rm = np.asarray(Rm9).reshape(3, 3)
        b = m['boxes9d'][0].astype(np.float64)
        d_Rt.append(max(np.abs(t - b[:3]).max(), np.abs(R.from_euler('xyz', b[6:9]).as_matrix() - Rm).max()))
        dets.append(np.linalg.det(Rm))
        zups.append(float(Rm[:, 2] @ up_cam))
        # 官方角点（含中心偏移）投到【原始畸变图】：官方函数 vs cv2.projectPoints
        corners_off = np.array([[-0.18, -0.16, -0.17], [-0.18, -0.16, 0.06], [-0.18, 0.18, -0.17], [-0.18, 0.18, 0.06],
                                [0.16, -0.16, -0.17], [0.16, -0.16, 0.06], [0.16, 0.18, -0.17], [0.16, 0.18, 0.06]])
        pc = corners_off @ Rm.T + t
        uv_cv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        uv_cv = uv_cv.reshape(-1, 2)
        if util is not None and hasattr(util, 'projection_in_distort_image'):
            try:
                # 官方：VICON 系齐次点 + vicon2camera_T
                pose = list(map(float, open(lp).readline().split()))[9:]
                r = R.from_quat([pose[-4], pose[-3], pose[-2], pose[-1]])
                cv_ = r.apply(corners_off) + np.array(pose[:3])
                hom = np.concatenate((cv_.T, np.ones((1, 8))), axis=0)
                uv_off = np.transpose(util.projection_in_distort_image(hom, BMV.CAMERA2VICON, D))[:, :2]
                d_proj.append(np.abs(uv_off - uv_cv).max())
            except Exception as e:
                print('   （官方 projection_in_distort_image 调用失败：%r）' % (e,))
        # 中心偏移：官方框几何中心 vs 我们用的 VICON 原点，在缓存图上差多少像素
        c_off = np.array([-0.01, 0.01, -0.055]) @ Rm.T + t
        Kin = np.asarray(m['K_in'])
        u0 = Kin @ t
        u1 = Kin @ c_off
        off_px.append((u1[:2] / u1[2] - u0[:2] / u0[2]))
        off_body.append(np.linalg.norm(c_off - t))
        img = cv2.imread(os.path.join('E:/MAV6D', cls, 'JPEGImages', scene, sq, m['frame']), cv2.IMREAD_COLOR)
        und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        small = cv2.resize(und, (512, 288), interpolation=cv2.INTER_AREA)
        d_img.append(np.abs(small.astype(int) - np.asarray(rgb[int(i)]).astype(int)).max())
        d_K.append(np.abs(cg.scale_K(K_pin, 512 / 1920, 288 / 1080) - Kin).max())
    off_px = np.array(off_px)
    check('B1 mav R,t==read_truth_Rt', max(d_Rt) < 1e-5, '12 帧最大差 %.1e' % max(d_Rt))
    check('B2 mav 右手系', min(dets) > 0.999, 'det(R) %.6f~%.6f' % (min(dets), max(dets)))
    check('B3 mav 机体z朝上', min(zups) > 0.5, 'z·世界上 %.3f~%.3f' % (min(zups), max(zups)))
    if d_proj:
        check('B4 官方投影==cv2投影', max(d_proj) < 1.0, '官方 projection_in_distort_image 与 cv2.projectPoints 最大差 %.3f px' % max(d_proj))
    check('B5 mav 缓存图像==去畸变缩放', max(d_img) == 0, '12 帧最大像素差 %d' % max(d_img))
    check('B6 mav K_in==针孔内参缩放', max(d_K) < 1e-9, '最大差 %.1e' % max(d_K))
    print('   [记录] 官方框几何中心相对 VICON 原点：机体系偏移 %.3f m；在 512x288 输入上 (dx,dy) 中位 (%.1f, %.1f) px，|d| 中位 %.1f px'
          % (np.median(off_body), np.median(off_px[:, 0]), np.median(off_px[:, 1]), np.median(np.linalg.norm(off_px, axis=1))))


# =============================================================================
def independent_targets(b, K, seq, f_ref, max_dis, max_size, stride=8):
    f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
    u = K @ b[:3]
    u = u[:2] / u[2]
    uh = u / stride
    cell = np.floor(uh).astype(int)
    zv = b[2] * f_ref / f_in
    d = b[:3] / np.linalg.norm(b[:3])
    axis = np.array([-d[1], d[0], 0.0])
    s = np.linalg.norm(axis)
    theta = np.arctan2(s, d[2])
    R_ray = R.from_rotvec(axis / s * theta).as_matrix() if s > 1e-12 else np.eye(3)
    R_allo = R_ray.T @ R.from_euler(seq, b[6:9]).as_matrix()
    a = R.from_matrix(R_allo).as_euler(seq)
    rot6 = np.array([np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1]), np.cos(a[2]), np.sin(a[2])])
    return cell, uh - cell, zv / max_dis, b[3:6] / max_size, rot6


def stage_C():
    for name, cfg_path in (('mav', MAV_CFG), ('sim', SIM_CFG)):
        cfg = load_cfg(cfg_path)
        dc = cfg.DATA_CONFIG
        ds = MMCache_Det_Dataset(dc, training=False)
        errs = {'img_mean': [], 'cell': 0, 'res': 0.0, 'dis': 0.0, 'dim': 0.0, 'rot': 0.0, 'dec_pos': 0.0, 'dec_ang': 0.0, 'n': 0}
        kw = encoder_geometry_kwargs(dc, center_point_decoder, 'dec')
        for it in rng.choice(len(ds), 6, replace=False):
            d = ds[int(it)]
            img = d['image']
            assert img.shape == (1, 3, 288, 512) and d['raw_im_size'].tolist() == [512, 288]
            errs['img_mean'].append((float(img.mean()), float(img.std())))
            K = d['intrinsic'][0]
            hm, res, dis, dim, rot = d['hm'][0], d['center_res'][0], d['center_dis'][0], d['dim'][0], d['rot'][0]
            assert hm.shape == (1, 36, 64), hm.shape
            for b in d['gt_box9d'].astype(np.float64):
                cell, r_, zv_n, dim_n, rot6 = independent_targets(b, K, dc.EULER_SEQ, dc.DEPTH_F_REF, dc.MAX_DIS, dc.MAX_SIZE)
                cx, cy = int(cell[0]), int(cell[1])
                if not (0 <= cx < 64 and 0 <= cy < 36):
                    continue
                errs['n'] += 1
                if hm[0, cy, cx] < 0.999:
                    errs['cell'] += 1
                errs['res'] = max(errs['res'], np.abs(np.array([res[0, cy, cx], res[1, cy, cx]]) - r_).max())
                errs['dis'] = max(errs['dis'], abs(dis[0, cy, cx] - zv_n))
                errs['dim'] = max(errs['dim'], np.abs(dim[:, cy, cx] - dim_n).max())
                errs['rot'] = max(errs['rot'], np.abs(rot[:, cy, cx] - rot6).max())
            dec, conf = center_point_decoder(torch.from_numpy(hm[None]).float(), torch.from_numpy(res[None]).float(),
                                             torch.from_numpy(dis[None] * dc.MAX_DIS).double(), torch.from_numpy(dim[None] * dc.MAX_SIZE).double(),
                                             torch.from_numpy(rot[None]).float(), np.array([K]), np.array([np.eye(4)]), np.zeros((1, 5)),
                                             512, 288, 512, 288, 8, 1, max(len(d['gt_box9d']), 1), **kw)
            for b in d['gt_box9d'].astype(np.float64):
                j = int(np.argmin(np.linalg.norm(dec[:, :3] - b[:3], axis=1)))
                errs['dec_pos'] = max(errs['dec_pos'], np.linalg.norm(dec[j, :3] - b[:3]))
                errs['dec_ang'] = max(errs['dec_ang'], np.degrees((R.from_euler('xyz', dec[j, 6:9]).inv() * R.from_euler('xyz', b[6:9])).magnitude()))
        mm = np.array(errs['img_mean'])
        check('C1 %s 图像归一化' % name, abs(mm[:, 0].mean()) < 0.5 and 0.5 < mm[:, 1].mean() < 2.0,
              '归一化后 均值 %.2f 标准差 %.2f（应≈0/≈1）' % (mm[:, 0].mean(), mm[:, 1].mean()))
        check('C2 %s 热力图格子' % name, errs['cell'] == 0, '%d 个目标，中心格热力图<0.999 的 %d 个' % (errs['n'], errs['cell']))
        check('C3 %s 监督图==独立重算' % name, errs['res'] < 1e-5 and errs['dis'] < 1e-6 and errs['dim'] < 1e-6 and errs['rot'] < 1e-5,
              '亚像素偏移 %.1e 虚拟深度 %.1e 尺寸 %.1e 视线相对旋转 euler6 %.1e' % (errs['res'], errs['dis'], errs['dim'], errs['rot']))
        check('C4 %s 解码还原真值' % name, errs['dec_pos'] < 1e-3 and errs['dec_ang'] < 0.05,
              '位置 %.1e m 角度 %.3f°' % (errs['dec_pos'], errs['dec_ang']))


# =============================================================================
def stage_D():
    for name, cfg_path, match in (('mav', MAV_CFG, 'top1'), ('sim', SIM_CFG, 'greedy')):
        cfg = load_cfg(cfg_path)
        dc = cfg.DATA_CONFIG
        kw = encoder_geometry_kwargs(dc, center_point_decoder, 'dec')
        torch.manual_seed(3)
        ds, loader, _ = build_dataloader(dc, batch_size=4, dist=False, workers=2, logger=common_utils.create_logger(), training=True, seed=3)
        worst_p, worst_a, n, nb = 0.0, 0.0, 0, 0
        for batch in loader:
            for b in range(batch['batch_size']):
                gt = batch['gt_box9d'][b]
                gt = gt[np.abs(gt).sum(1) > 0].astype(np.float64)
                if len(gt) == 0:
                    continue
                K = batch['intrinsic'][b][0]
                dec, _ = center_point_decoder(torch.from_numpy(batch['hm'][b]).float(), torch.from_numpy(batch['center_res'][b]).float(),
                                              torch.from_numpy(batch['center_dis'][b] * dc.MAX_DIS).double(), torch.from_numpy(batch['dim'][b] * dc.MAX_SIZE).double(),
                                              torch.from_numpy(batch['rot'][b]).float(), np.array([K]), np.array([np.eye(4)]), np.zeros((1, 5)),
                                              512, 288, 512, 288, 8, 1, len(gt), **kw)
                for g in gt:
                    j = int(np.argmin(np.linalg.norm(dec[:, :3] - g[:3], axis=1)))
                    worst_p = max(worst_p, np.linalg.norm(dec[j, :3] - g[:3]))
                    worst_a = max(worst_a, np.degrees((R.from_euler('xyz', dec[j, 6:9]).inv() * R.from_euler('xyz', g[6:9])).magnitude()))
                    n += 1
            nb += 1
            if nb >= 3:
                break
        check('D1 %s spawn worker 增广自洽' % name, worst_p < 1e-3 and worst_a < 0.05 and n > 0,
              'workers=2 训练 loader 3 个 batch %d 个目标：监督图解码 vs 同 worker 的 gt 位置 %.1e m 角度 %.3f°' % (n, worst_p, worst_a))
        # 评测 loader：worker 输出 vs 主进程 __getitem__ 逐位一致（确定性路径）
        dse, le, _ = build_dataloader(dc, batch_size=4, dist=False, workers=2, logger=common_utils.create_logger(), training=False)
        batch = next(iter(le))
        worst = 0.0
        for b in range(batch['batch_size']):
            d0 = dse[b]
            worst = max(worst, np.abs(batch['image'][b] - d0['image']).max(), np.abs(batch['intrinsic'][b] - d0['intrinsic']).max(),
                        np.abs(batch['center_dis'][b] - d0['center_dis']).max(), np.abs(batch['rot'][b] - d0['rot']).max())
        check('D2 %s 评测 loader==主进程' % name, worst == 0.0, 'worker 与主进程同索引样本最大差 %.1e' % worst)


# =============================================================================
def overfit(cfg_path, n_items, steps, tag):
    cfg = load_cfg(cfg_path)
    dc = cfg.DATA_CONFIG
    ds = MMCache_Det_Dataset(dc, training=False)
    items = [ds[int(i)] for i in rng.choice(len(ds), n_items, replace=False)]
    batch = ds.collate_batch(items)
    model = build_network(cfg.MODEL, ds).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    t0 = time.time()
    hist = []
    for s in range(steps):
        for g in opt.param_groups:                     # 余弦退火到 1e-5：单批过拟合要看能否收敛到厘米级，恒定 lr 会在后期抖动
            g['lr'] = 1e-5 + 0.5 * (5e-4 - 1e-5) * (1 + np.cos(np.pi * s / steps))
        bd = dict(batch)
        load_data_to_gpu(bd)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(bd)['loss']
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
        opt.step()
        if s % 100 == 0 or s == steps - 1:
            hist.append((s, float(loss.detach()), dict(model.dense_head_2d.loss_terms)))
    model.eval()
    bd = dict(batch)
    load_data_to_gpu(bd)
    with torch.no_grad():
        out = model(bd)
    wp, wa, wd, n = 0.0, 0.0, 0.0, 0
    for b in range(out['batch_size']):
        gt = np.asarray(batch['gt_box9d'][b])
        gt = gt[np.abs(gt).sum(1) > 0].astype(np.float64)
        pred = np.asarray(out['pred_boxes9d'][b])
        for g in gt:
            if len(pred) == 0:
                wp = 9.0
                continue
            j = int(np.argmin(np.linalg.norm(pred[:, :3] - g[:3], axis=1)))
            wp = max(wp, np.linalg.norm(pred[j, :3] - g[:3]))
            wd = max(wd, np.abs(pred[j, 3:6] - g[3:6]).max())
            wa = max(wa, np.degrees((R.from_euler('xyz', pred[j, 6:9]).inv() * R.from_euler('xyz', g[6:9])).magnitude()))
            n += 1
    print('   [%s] %d 步 %.0f s；损失 %s' % (tag, steps, time.time() - t0, ' -> '.join('%d:%.3f' % (s, l) for s, l, _ in hist)))
    print('   [%s] 末步分项 %s' % (tag, {k: round(v, 4) for k, v in hist[-1][2].items()}))
    return model, batch, out, wp, wa, wd, n


def stage_EF():
    from tools.train_utils.train_utils import checkpoint_state, save_checkpoint
    model, batch, out, wp, wa, wd, n = overfit(MAV_CFG, 4, 800, 'mav 单批过拟合')
    check('E1 mav 单批过拟合', wp < 0.05 and wa < 5.0 and n == 4, '%d 目标：位置最大 %.3f m 角度 %.2f° 尺寸 %.3f m' % (n, wp, wa, wd))
    # F: 保存 -> 载入 -> 逐位一致
    ck = os.path.join(OUT, 'ckpt_walkthrough')
    save_checkpoint(checkpoint_state(model, None, 1, 300), filename=ck)
    sd = torch.load(ck + '.pth', map_location='cpu', weights_only=False)
    dtypes = set(str(v.dtype) for v in sd['model_state'].values() if torch.is_tensor(v) and v.is_floating_point())
    cfg = load_cfg(MAV_CFG)
    ds2 = MMCache_Det_Dataset(cfg.DATA_CONFIG, training=False)
    model2 = build_network(cfg.MODEL, ds2)
    n_l, n_t = model2.load_params_from_file(ck + '.pth', to_cpu=False)
    model2.cuda().eval()
    bd = dict(batch)
    load_data_to_gpu(bd)
    with torch.no_grad():
        out2 = model2(bd)
    dmax = max(np.abs(np.asarray(out['pred_boxes9d'][b]) - np.asarray(out2['pred_boxes9d'][b])).max() for b in range(out['batch_size']))
    check('F1 checkpoint 存/载', n_l == n_t and dmax == 0.0 and dtypes == {'torch.float32'},
          '载入 %d/%d 张量，重载后预测最大差 %.1e，权重 dtype %s' % (n_l, n_t, dmax, sorted(dtypes)))
    del model, model2
    torch.cuda.empty_cache()
    model, batch, out, wp, wa, wd, n = overfit(SIM_CFG, 4, 800, 'sim 单批过拟合(多目标)')
    check('E2 sim 单批过拟合', wp < 0.08 and wa < 6.0 and n >= 4, '%d 目标：位置最大 %.3f m 角度 %.2f° 尺寸 %.3f m' % (n, wp, wa, wd))
    del model
    torch.cuda.empty_cache()
    # F2: 实际训练产出的 best.json / best.pth / val_history 是否自洽
    bad = []
    for run in ('sim_indoor8_mz/S1', 'mav6d/C_p025_s0', 'mav6d/T_p025_s0', 'mav6d/C_p001_s0', 'mav6d/T_p005_s1'):
        d = os.path.join(ROOT, 'output', 'models', 'uavdet_3d', 'camnorm', run)
        if not os.path.exists(os.path.join(d, 'ckpt', 'best.pth')):
            continue
        h = json.load(open(os.path.join(d, 'val_history.json'), encoding='utf-8'))
        bj = json.load(open(os.path.join(d, 'ckpt', 'best.json'), encoding='utf-8'))
        best = max(h, key=lambda x: x['score'])
        ep_in_ckpt = torch.load(os.path.join(d, 'ckpt', 'best.pth'), map_location='cpu', weights_only=False).get('epoch')
        ok = bj['epoch'] == best['epoch'] == ep_in_ckpt and abs(bj['score'] - best['score']) < 1e-9
        if not ok:
            bad.append('%s: best.json ep%s / 历史最高 ep%s / ckpt ep%s' % (run, bj['epoch'], best['epoch'], ep_in_ckpt))
        print('   [F2] %-22s best.json 第 %2d 轮 score %.4f | 验证曲线最高 第 %2d 轮 %.4f | best.pth 记录 epoch %s' %
              (run, bj['epoch'], bj['score'], best['epoch'], best['score'], ep_in_ckpt))
    check('F2 按验证集选轮记录自洽', not bad, '; '.join(bad) if bad else '5 个 run 的 best.json == 验证曲线最高轮 == best.pth 内 epoch')


# =============================================================================
def stage_G():
    import eval_camnorm as ec
    from uavdet3d.utils import frame_convention
    cfg = load_cfg(MAV_CFG)
    ds = MMCache_Det_Dataset(cfg.DATA_CONFIG, training=False)
    picks = rng.choice(len(ds), 150, replace=False)
    recs, raw_whs = [], []
    for it in picks:
        m = ds.metas[int(ds.valid_idx[int(it)])]
        g = m['boxes9d'].astype(np.float64)
        p_ = g.copy()
        p_[:, :3] += 1e-3 * rng.randn(*p_[:, :3].shape)     # 加 1 mm 扰动：仓库的 ADS 把「误差全为 0」当成「没匹配」(accuracy 记 0)，见报告
        recs.append({'gt': g, 'pred': p_, 'conf': np.array([0.5 + 0.4 * rng.rand()]), 'K': np.asarray(m['K_in'], np.float64),
                     'seq_id': m['seq'], 'frame_id': m['frame']})
        raw_whs.append(tuple(m['raw_wh']))
    frame_convention.set_default_euler_seq('zyx')          # 故意先设错，看 ads_eval 会不会自己纠正成 'xyz'
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep = ec.ads_eval(recs, raw_whs, 512, 288, 'xyz', ['laa', 'indoor'], os.path.join(OUT, 'ads_gt_as_pred'), 'gt')
    vals = {}
    for prof, txt in rep.items():
        g = lambda key: float([ln for ln in txt.splitlines() if key in ln][0].split(':')[-1])   # noqa: E731
        vals[prof] = (g('orin_error_median(degree)'), g('posi_error_median(m)'), g('size_error_median(m)'),
                      float(txt.split('LAA3D_ADS_drone (%) : ')[1].split()[0]))
    ok = all(v[0] < 1e-3 and v[1] < 3e-3 and v[2] < 1e-4 and v[3] >= 99.5 for v in vals.values())
    check('G1 评测：真值(+1mm)当预测', ok, ' | '.join('%s: 角度 %.1e° 位置 %.1e m 尺寸 %.1e m ADS %.2f' % (p, *v) for p, v in vals.items())
          + '（150 帧；调用前故意把全局欧拉顺序设成 zyx）')
    frame_convention.set_default_euler_seq('xyz')


# =============================================================================
def stage_H():
    cfg = load_cfg(MAV_CFG, ['DATA_CONFIG.SAMPLED_INTERVAL.train', '4', 'DATA_CONFIG.TRAIN_REPEAT', '2'])
    ds = MMCache_Det_Dataset(cfg.DATA_CONFIG, training=True)
    uniq = sorted(set(ds.valid_idx.tolist()))
    seqs = {}
    for i in uniq:
        s = ds.metas[i]['seq']
        seqs[s] = seqs.get(s, 0) + 1
    full = {}
    idx = pickle.load(open('E:/mmcache/mav6d_cn/train/index.pkl', 'rb'))
    for i in idx['valid_idx']:
        s = idx['metas'][int(i)]['seq']
        full[s] = full.get(s, 0) + 1
    frac = np.array([seqs.get(s, 0) / full[s] for s in full])
    check('H1 抽帧覆盖所有序列', len(seqs) == len(full) and frac.min() > 0.2 and frac.max() < 0.3,
          '25%% 档：%d 唯一帧 x 重复 2 = %d，覆盖 %d/%d 序列，每序列抽样比例 %.2f~%.2f' % (len(uniq), len(ds), len(seqs), len(full), frac.min(), frac.max()))


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    for fn in (stage_A, stage_B, stage_C, stage_D, stage_EF, stage_G, stage_H):
        if only and fn.__name__ not in only:
            continue
        print('=' * 24, fn.__name__, '=' * 24, flush=True)
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            check(fn.__name__ + ' 异常', False, repr(e))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print('\n==== %d 项检查，失败 %d ====' % (len(RESULTS), n_fail))
    for name, ok in RESULTS:
        if not ok:
            print('   FAIL:', name)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
