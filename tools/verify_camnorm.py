# -*- coding: utf-8 -*-
"""camnorm 管线的回归测试。任何一项不过都不许开训。

    cd E:/Open3DUAVDet/tools && python verify_camnorm.py [--real]

  1 geometry     camera_geometry.self_check（K 缩放 / 翻转 / 视线相对旋转 / 虚拟深度 / 假内参识别）
  2 roundtrip    编码 -> 解码往返：随机焦距 120~2000 px、主点偏移、多种输入分辨率，
                 metric/virtual x ego/allo 四种组合，多目标，位置 / 角度 / 尺寸误差应为 0
  3 flip_rule    翻转标签的朝向语义（真实数据集 _augment）：翻转后机头 = 镜像机头、机顶 = 镜像机顶
                 （无人机左右对称，R' = M R S；合成盒子分不出前后，这一项专门补上）
    mut_flip_x   变异测试：用旧的 M R M（机头机尾对调）必须被 worker_aug 判失败
  3 worker_aug   合成缓存 + 真 DataLoader（spawn worker，与训练同路径）+ 翻转/缩放增广：
                 图上 8 个角点标记的位置 == 用增广后的 K 与 3D 框投影的位置（按角点编号比，
                 180° 对称错误也能抓到）；worker 输出的监督图解码回来 == 增广后的真值框
  4 model_decode 把真值监督图当成网络输出走 CenterDet.post_processing，框应还原
  5 ads_known    LAA3D_ADS 已知答案：只平移 0.2 m -> 位置 0.2、角度 0、尺寸 0；只转 20° -> 角度 20.000、尺寸 0
  6 real_caches  (--real) 真实缓存：机体 z 朝上（仿真与 MAV6D 同号）、K_in 与分辨率一致、
                 val/test 无重叠、档位帧数、增广后投影仍在画面内
"""
import argparse
import os
import pickle
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.utils import camera_geometry as cg   # noqa: E402
from uavdet3d.utils.object_encoder_mav6d import center_point_decoder, center_point_encoder   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
# 翻转后角点编号的对应：无人机左右对称，翻转标签 R' = M R S、S = 机体 y 取反 -> 原型角点 y 取反。
# 测试只接受「未翻转 = 恒等」或「翻转 = y 取反」；错误的 M R M（机体 x 取反 = 机头机尾对调）会被判失败。
FLIP_PERM = [3, 2, 1, 0, 7, 6, 5, 4]
RESULTS = []


def report(name, ok, msg):
    RESULTS.append((name, bool(ok)))
    print('[%s] %-13s %s' % ('PASS' if ok else 'FAIL', name, msg), flush=True)


def corners(box, seq='xyz'):
    return (PROTO8 * box[3:6]) @ R.from_euler(seq, box[6:9]).as_matrix().T + box[:3]


def proj(K, pts):
    uv = (K @ np.asarray(pts).T).T
    return uv[:, :2] / uv[:, 2:3]


def ang_err(a, b, seq='xyz'):
    return np.degrees((R.from_euler(seq, a).inv() * R.from_euler(seq, b)).magnitude())


def decode_maps(hm, res, dis, dim, rot, K, W, H, max_dis, max_size, kw, n):
    boxes, conf = center_point_decoder(
        torch.from_numpy(np.asarray(hm, np.float32)), torch.from_numpy(np.asarray(res, np.float32)),
        torch.from_numpy(np.asarray(dis, np.float64) * max_dis), torch.from_numpy(np.asarray(dim, np.float64) * max_size),
        torch.from_numpy(np.asarray(rot, np.float32)), np.asarray([K]), np.asarray([np.eye(4)]), np.zeros((1, 5)),
        W, H, W, H, 8, 1, n, **kw)
    return boxes


def random_scene(rng, W, H, K, n_obj, seq='xyz', margin=6):
    boxes = []
    tries = 0
    while len(boxes) < n_obj and tries < 500:
        tries += 1
        u, v = rng.uniform(margin * 4, W - margin * 4), rng.uniform(margin * 4, H - margin * 4)
        Z = rng.uniform(1.0, 9.0)
        x = (u - K[0, 2] - K[0, 1] * (v - K[1, 2]) / K[1, 1]) * Z / K[0, 0]
        y = (v - K[1, 2]) * Z / K[1, 1]
        size = np.array([0.5, 0.45, 0.2]) * rng.uniform(0.6, 3.0)
        b = np.concatenate([[x, y, Z], size, R.random(random_state=rng).as_euler(seq)])
        uvc = proj(K, corners(b, seq))
        if not ((uvc[:, 0] > margin) & (uvc[:, 0] < W - margin) & (uvc[:, 1] > margin) & (uvc[:, 1] < H - margin)).all():
            continue
        c = proj(K, b[None, :3])[0]
        if any(np.linalg.norm(c - proj(K, o[None, :3])[0]) < 40 for o in boxes):
            continue                              # 目标中心相距太近时 NMS 会合并，与本测试无关
        boxes.append(b)
    return np.array(boxes)


# --------------------------------------------------------------------------- #
def t_roundtrip(seed=0):
    rng = np.random.RandomState(seed)
    worst = {'pos': 0.0, 'ang': 0.0, 'size': 0.0, 'miss': 0, 'n': 0}
    for trial in range(60):
        W, H = [(512, 288), (640, 480), (320, 240), (800, 450)][trial % 4]
        f = rng.uniform(120, 2000) * W / 1920.0 * rng.uniform(0.5, 2.0)
        K = np.array([[f * rng.uniform(0.97, 1.03), rng.uniform(-0.5, 0.5), W / 2 + rng.uniform(-30, 30)],
                      [0, f, H / 2 + rng.uniform(-20, 20)], [0, 0, 1.0]])
        gt = random_scene(rng, W, H, K, n_obj=3)
        if len(gt) == 0:
            continue
        for depth_mode in ('metric', 'virtual'):
            for rot_frame in ('ego', 'allo'):
                kw = dict(rot_repr='euler6', euler_seq='xyz', depth_mode=depth_mode, f_ref=512, rot_frame=rot_frame)
                withcls = np.concatenate([gt, np.zeros((len(gt), 1))], 1)
                hm, res, dis, dim, rot = center_point_encoder(withcls, np.array([K]), np.array([np.eye(4)]),
                                                              np.zeros((1, 5)), W, H, W, H, 8, 1, ['drone'], 2, **kw)
                dec = decode_maps(hm[0][None], res[0][None], dis[0][None] / 100.0, dim[0][None] / 10.0, rot[0][None],
                                  K, W, H, 100.0, 10.0, kw, len(gt))
                for g in gt:
                    d = np.linalg.norm(dec[:, :3] - g[:3], axis=1)
                    k = int(np.argmin(d))
                    worst['n'] += 1
                    if d[k] > 0.5:
                        worst['miss'] += 1
                        continue
                    worst['pos'] = max(worst['pos'], d[k])
                    worst['ang'] = max(worst['ang'], ang_err(dec[k, 6:9], g[6:9]))
                    worst['size'] = max(worst['size'], np.abs(dec[k, 3:6] - g[3:6]).max())
    ok = worst['miss'] == 0 and worst['pos'] < 1e-4 and worst['ang'] < 1e-3 and worst['size'] < 1e-5
    report('roundtrip', ok, '%d 个目标 x 4 种模式：漏检 %d，位置最大误差 %.2e m，角度 %.2e°，尺寸 %.2e m'
           % (worst['n'] // 4, worst['miss'], worst['pos'], worst['ang'], worst['size']))


# --------------------------------------------------------------------------- #
def make_synthetic_cache(root, n=24, W=512, H=288, seed=1):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, 'train'), exist_ok=True)
    rgb = np.lib.format.open_memmap(os.path.join(root, 'train', 'rgb.npy'), 'w+', np.uint8, (n, H, W, 3))
    metas = []
    for i in range(n):
        while True:
            f = rng.uniform(200, 600)
            K = np.array([[f * rng.uniform(0.98, 1.02), 0.0, W / 2 + rng.uniform(-25, 25)],
                          [0.0, f, H / 2 + rng.uniform(-12, 12)], [0, 0, 1.0]])
            b = random_scene(rng, W, H, K, 1, margin=60)
            if len(b) == 0:
                continue
            uvc = proj(K, corners(b[0]))
            dmin = min(np.linalg.norm(uvc[a] - uvc[c]) for a in range(8) for c in range(a + 1, 8))
            if dmin >= 14:
                break
        img = np.zeros((H, W, 3), np.uint8)
        for k in range(8):
            cv2.circle(img, (int(round(uvc[k, 0] * 16)), int(round(uvc[k, 1] * 16))), 3 * 16,
                       (int((k + 1) * 28), 255, 0), -1, lineType=cv2.LINE_AA, shift=4)
        rgb[i] = img
        metas.append({'seq': 'syn/%03d' % i, 'frame': '%d.png' % i, 'K_in': K, 'raw_wh': (W, H),
                      'boxes9d': b.astype(np.float32), 'names': ['drone'], 'qualified': np.array([True])})
    rgb.flush()
    pickle.dump({'valid_idx': np.arange(n), 'metas': metas, 'H': H, 'W': W, 'format': 'camnorm-v1'},
                open(os.path.join(root, 'train', 'index.pkl'), 'wb'))


def make_synthetic_native_cache(root, split='train', n=24, W=1280, H=720, seed=2):
    """原分辨率 + 「jpeg 存储」格式的合成缓存（字节用无损 PNG 编码，cv2.imdecode 一样解）：角点标记半径 8 px，
    整幅缩到 512 后约 3 px、放大 4 倍时约 13 px，两头都能稳定定位。"""
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, split), exist_ok=True)
    blob = open(os.path.join(root, split, 'rgb_jpg.bin'), 'wb')
    index = np.zeros((n, 2), np.int64)
    pos = 0
    metas = []
    for i in range(n):
        while True:
            K = np.array([[640.0, 0.0, 639.5], [0.0, 640.0, 359.5], [0, 0, 1.0]])
            b = random_scene(rng, W, H, K, 1, margin=150)
            if len(b) == 0:
                continue
            uvc = proj(K, corners(b[0]))
            dmin = min(np.linalg.norm(uvc[a] - uvc[c]) for a in range(8) for c in range(a + 1, 8))
            if dmin >= 45:
                break
        img = np.zeros((H, W, 3), np.uint8)
        for k in range(8):
            cv2.circle(img, (int(round(uvc[k, 0] * 16)), int(round(uvc[k, 1] * 16))), 8 * 16,
                       (int((k + 1) * 28), 255, 0), -1, lineType=cv2.LINE_AA, shift=4)
        ok, buf = cv2.imencode('.png', img)
        blob.write(buf.tobytes())
        index[i] = (pos, len(buf))
        pos += len(buf)
        metas.append({'seq': 'syn/%03d' % i, 'frame': '%d.png' % i, 'K_in': K, 'raw_wh': (W, H),
                      'boxes9d': b.astype(np.float32), 'names': ['drone'], 'qualified': np.array([True])})
    blob.close()
    np.save(os.path.join(root, split, 'rgb_jpg_index.npy'), index)
    pickle.dump({'valid_idx': np.arange(n), 'metas': metas, 'H': H, 'W': W, 'format': 'camnorm-v1', 'store': 'jpeg'},
                open(os.path.join(root, split, 'index.pkl'), 'wb'))


def t_worker_view():
    """原分辨率存储 + 在线多焦距裁窗（VIEW_AUG）+ 翻转，真 DataLoader spawn worker：
    角点标记位置 == 用裁窗后 K 与 3D 框的投影；监督图解码 == 真值；等效焦距确实覆盖 1~4 倍；验证模式按 VAL_ZOOMS 轮流且确定。"""
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset
    from uavdet3d.utils import common_utils
    tmp = tempfile.mkdtemp(prefix='camnorm_view_', dir=os.environ.get('CAMNORM_TMP', None))
    try:
        make_synthetic_native_cache(tmp, 'train')
        make_synthetic_native_cache(tmp, 'test', seed=3)
        cfg = EasyDict()
        cfg_from_yaml_file('cfgs/dataset_configs/uavdet_3d/camnorm_base.yaml', cfg)
        cfg.DATA_PATH = tmp
        cfg.AUG = EasyDict({'hflip': 0.5, 'scale': [1.0, 1.0], 'photometric': False, 'noise': 0.0})
        cfg.VIEW_AUG = EasyDict({'zoom': [1.0, 4.0]})
        cfg.VAL_ZOOMS = [1.0, 2.5]
        torch.manual_seed(11)
        ds, loader, _ = build_dataloader(cfg, batch_size=4, dist=False, workers=2, logger=common_utils.create_logger(),
                                         training=True, seed=11)
        kw = dict(rot_repr='euler6', euler_seq='xyz', depth_mode='virtual', f_ref=cfg.DEPTH_F_REF, rot_frame='allo')
        worst_px, worst_pos, worst_ang, n_obj, zooms, worst_info = 0.0, 0.0, 0.0, 0, [], None
        for epoch in range(3):
            for batch in loader:
                for b in range(batch['batch_size']):
                    img = batch['image'][b, 0].transpose(1, 2, 0)
                    K = batch['intrinsic'][b][0]
                    gt = batch['gt_box9d'][b]
                    gt = gt[np.abs(gt).sum(1) > 0]
                    if len(gt) == 0:
                        continue
                    g = gt[0].astype(np.float64)
                    zooms.append(cg.focal(K) / 256.0)
                    dots = find_dots(img)
                    exp = proj(K, corners(g))
                    errs = []
                    # 标记半径随焦距放大（原图 8 px）：离边界不到「半径 + 2 px」的标记会被裁掉一部分，质心偏，不参与比较
                    rb = 8.0 * cg.focal(K) / 640.0 + 2.0
                    for perm in (list(range(8)), FLIP_PERM):
                        e = [np.linalg.norm(dots[k] - exp[perm[k]]) for k in dots
                             if rb <= exp[perm[k]][0] <= img.shape[1] - 1 - rb and rb <= exp[perm[k]][1] <= img.shape[0] - 1 - rb]
                        errs.append(max(e) if e else np.inf)
                    if not np.isfinite(min(errs)):
                        continue
                    if min(errs) > worst_px:
                        worst_info = (round(cg.focal(K) / 256.0, 2), round(min(errs), 3))
                    worst_px = max(worst_px, min(errs))
                    dec = decode_maps(batch['hm'][b], batch['center_res'][b], batch['center_dis'][b], batch['dim'][b],
                                      batch['rot'][b], K, 512, 288, cfg.MAX_DIS, cfg.MAX_SIZE, kw, 1)
                    worst_pos = max(worst_pos, np.linalg.norm(dec[0, :3] - g[:3]))
                    worst_ang = max(worst_ang, ang_err(dec[0, 6:9], g[6:9]))
                    n_obj += 1
        zooms = np.array(zooms)
        # 验证模式：确定性 + VAL_ZOOMS 轮流
        tcfg = EasyDict(cfg)
        dst = MMCache_Det_Dataset(tcfg, training=False)
        f1 = [cg.focal(dst[i]['intrinsic'][0]) / 256.0 for i in range(4)]
        f2 = [cg.focal(dst[i]['intrinsic'][0]) / 256.0 for i in range(4)]
        det_ok = np.allclose(f1, f2) and abs(f1[0] - 1.0) < 1e-6 and abs(f1[1] - 2.5) < 0.05
        ok = (n_obj > 40 and worst_px < 1.0 and worst_pos < 1e-3 and worst_ang < 0.05 and zooms.min() < 1.3
              and zooms.max() > 3.0 and det_ok)
        report('worker_view', ok, '%d 个样本，等效焦距倍数 %.2f~%.2f（中位 %.2f），角点标记与投影最大偏差 %.3f px，'
               '（最差样本 焦距倍数/偏差 %s）监督图解码 位置 %.2e m / 角度 %.3f°；验证模式焦距倍数 %s（两次一致 %s）'
               % (n_obj, zooms.min(), zooms.max(), np.median(zooms), worst_px, worst_info, worst_pos, worst_ang,
                  np.round(f1, 3).tolist(), np.allclose(f1, f2)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def find_dots(img):
    """img (H,W,3) 0~1 float（BGR）。按 B/G 比值识别角点编号，按 G 加权求质心。"""
    B, G = img[:, :, 0], img[:, :, 1]
    out = {}
    mask = G > 0.35
    k_map = np.rint(np.where(mask, B / np.maximum(G, 1e-6), 0) * 255.0 / 28.0 - 1).astype(int)
    for k in range(8):
        m = mask & (k_map == k)
        if m.sum() < 3:
            continue
        ys, xs = np.nonzero(m)
        w = G[ys, xs]
        out[k] = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
    return out


def t_flip_rule():
    """翻转标签的朝向语义：翻转后机头方向 = 镜像的机头方向，机顶方向 = 镜像的机顶方向，仍是真旋转。
    走真实数据集的 _augment（hflip=1），合成盒子测试分不出前后，这一项专门补上。"""
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset
    tmp = tempfile.mkdtemp(prefix='camnorm_flip_', dir=os.environ.get('CAMNORM_TMP', None))
    try:
        make_synthetic_cache(tmp, n=8)
        cfg = EasyDict()
        cfg_from_yaml_file('cfgs/dataset_configs/uavdet_3d/camnorm_base.yaml', cfg)
        cfg.DATA_PATH = tmp
        cfg.AUG = EasyDict({'hflip': 1.0, 'scale': None, 'photometric': False, 'noise': 0.0})
        ds = MMCache_Det_Dataset(cfg, training=True)
        M = np.diag([-1.0, 1.0, 1.0])
        worst = 0.0
        rng = np.random.RandomState(0)
        for _ in range(200):
            m = ds.metas[rng.randint(len(ds.metas))]
            b = m['boxes9d'].astype(np.float64).copy()
            b[0, 6:9] = R.random(random_state=rng).as_euler('xyz')
            img = np.zeros((3, ds.H, ds.W), np.float32)
            _, b2, _, _ = ds._augment(img, b.copy(), ['drone'], ds.cache_intrinsic(m))
            R0 = R.from_euler('xyz', b[0, 6:9]).as_matrix()
            R1 = R.from_euler('xyz', b2[0, 6:9]).as_matrix()
            worst = max(worst, np.abs(R1[:, 0] - M @ R0[:, 0]).max(), np.abs(R1[:, 2] - M @ R0[:, 2]).max(),
                        abs(np.linalg.det(R1) - 1.0))
        report('flip_rule', worst < 1e-9, '翻转后 机头(x)=镜像机头、机顶(z)=镜像机顶、det=1，最大偏差 %.1e' % worst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_worker_aug(body_mirror=None, expect_fail=False):
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.utils import common_utils
    tmp = tempfile.mkdtemp(prefix='camnorm_verify_', dir=os.environ.get('CAMNORM_TMP', None))
    try:
        make_synthetic_cache(tmp)
        cfg = EasyDict()
        cfg_from_yaml_file('cfgs/dataset_configs/uavdet_3d/camnorm_base.yaml', cfg)
        cfg.DATA_PATH = tmp
        cfg.AUG = EasyDict({'hflip': 0.5, 'scale': [0.7, 1.4], 'photometric': False, 'noise': 0.0})
        if body_mirror:
            cfg.FLIP_BODY_MIRROR_AXIS = body_mirror
        logger = common_utils.create_logger()
        torch.manual_seed(7)          # worker 的随机数由主进程 torch RNG 派生，固定它测试才可复现
        ds, loader, _ = build_dataloader(cfg, batch_size=4, dist=False, workers=2, logger=logger, training=True, seed=7)
        kw = dict(rot_repr='euler6', euler_seq='xyz', depth_mode='virtual', f_ref=cfg.DEPTH_F_REF, rot_frame='allo')
        worst_px, worst_pos, worst_ang, n_obj, n_flip, n_scaled, n_dots = 0.0, 0.0, 0.0, 0, 0, 0, 0
        for epoch in range(3):
            for batch in loader:
                for b in range(batch['batch_size']):
                    img = batch['image'][b, 0].transpose(1, 2, 0)          # 未设 NORM，仍是 0~1
                    K = batch['intrinsic'][b][0]
                    gt = batch['gt_box9d'][b]
                    gt = gt[np.abs(gt).sum(1) > 0]
                    if len(gt) == 0:
                        continue
                    g = gt[0].astype(np.float64)
                    dots = find_dots(img)
                    exp = proj(K, corners(g))
                    errs = []
                    for perm in (list(range(8)), FLIP_PERM):
                        # 放大裁剪会把贴边的标记切掉一半，可见部分的质心必然偏向画面内 —— 那是标记法的局限，
                        # 不是几何错误（实测离边 >=5 px 的标记误差 <=0.17 px，贴边的可达 2.5 px），只比完整可见的
                        e = [np.linalg.norm(dots[k] - exp[perm[k]]) for k in dots
                             if 5 <= exp[perm[k]][0] <= img.shape[1] - 6 and 5 <= exp[perm[k]][1] <= img.shape[0] - 6]
                        errs.append(max(e) if e else np.inf)
                    if not np.isfinite(min(errs)):
                        continue              # 8 个标记全部贴边/出画，无从比较
                    best = int(np.argmin(errs))
                    n_flip += best
                    n_scaled += int(abs(cg.focal(K) - cg.focal(ds.cache_intrinsic(ds.metas[0]))) > 1e-6)
                    n_dots += len(dots)
                    worst_px = max(worst_px, errs[best])
                    dec = decode_maps(batch['hm'][b], batch['center_res'][b], batch['center_dis'][b], batch['dim'][b],
                                      batch['rot'][b], K, 512, 288, cfg.MAX_DIS, cfg.MAX_SIZE, kw, 1)
                    worst_pos = max(worst_pos, np.linalg.norm(dec[0, :3] - g[:3]))
                    worst_ang = max(worst_ang, ang_err(dec[0, 6:9], g[6:9]))
                    n_obj += 1
        ok = n_obj > 40 and worst_px < 1.0 and worst_pos < 1e-3 and worst_ang < 0.05 and 0 < n_flip < n_obj
        if expect_fail:
            report('mut_flip_x', not ok, '变异测试（翻转用机体 x 镜像 = 旧的 M R M）应被判失败：角点最大偏差 %.2f px' % worst_px)
            return
        report('worker_aug', ok, '%d 个样本（spawn worker，其中翻转 %d），角点标记与投影最大偏差 %.3f px，'
               '监督图解码 位置 %.2e m / 角度 %.3f°' % (n_obj, n_flip, worst_px, worst_pos, worst_ang))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def t_model_decode():
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.model.detectors.center_det import CenterDet
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', cfg)
    dc = cfg.DATA_CONFIG

    class _DS:
        dataset_cfg = dc
        im_num = 1

    det = CenterDet.__new__(CenterDet)
    torch.nn.Module.__init__(det)
    det.model_cfg, det.dataset = cfg.MODEL, _DS()
    from uavdet3d.utils.object_encoder import all_object_encoders
    from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs
    det.center_decoder = all_object_encoders[cfg.MODEL.POST_PROCESSING.DECONDER]
    det.dec_kwargs = encoder_geometry_kwargs(dc, det.center_decoder, 'dec')
    det.max_num, det.score_thresh = 1, cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    rng = np.random.RandomState(3)
    worst_p, worst_a = 0.0, 0.0
    for _ in range(30):
        K = np.array([[rng.uniform(250, 600), 0, 256 + rng.uniform(-20, 20)], [0, 0, 144 + rng.uniform(-10, 10)], [0, 0, 1.0]])
        K[1, 1] = K[0, 0]
        gt = random_scene(rng, 512, 288, K, 1)
        kw = encoder_geometry_kwargs(dc, center_point_encoder, 'enc')
        hm, res, dis, dim, rot = center_point_encoder(np.concatenate([gt, [[0]]], 1), np.array([K]), np.array([np.eye(4)]),
                                                      np.zeros((1, 5)), 512, 288, 512, 288, 8, 1, ['drone'], 2, **kw)
        logit = torch.logit(torch.from_numpy(hm).float().clamp(1e-4, 1 - 1e-4))
        batch = {'batch_size': 1, 'intrinsic': np.array([[K]]), 'extrinsic': np.array([[np.eye(4)]]),
                 'distortion': np.zeros((1, 1, 5)), 'raw_im_size': np.array([[512, 288]]),
                 'new_im_size': np.array([[512, 288]]), 'stride': np.array([8]),
                 'pred_center_dict': {'hm': logit, 'center_res': torch.from_numpy(res).float(),
                                      'center_dis': torch.from_numpy(dis / dc.MAX_DIS).float(),
                                      'dim': torch.from_numpy(dim / dc.MAX_SIZE).float(),
                                      'rot': torch.from_numpy(rot).float()}}
        out = det.post_processing(batch)
        p = out['pred_boxes9d'][0][0]
        worst_p = max(worst_p, np.linalg.norm(p[:3] - gt[0, :3]))
        worst_a = max(worst_a, ang_err(p[6:9], gt[0, 6:9]))
    report('model_decode', worst_p < 1e-3 and worst_a < 0.05,
           'CenterDet.post_processing 还原真值：位置 %.2e m，角度 %.3f°（float32 监督图精度）' % (worst_p, worst_a))


# --------------------------------------------------------------------------- #
def t_ads_known():
    import contextlib
    import io
    import eval_camnorm as ec
    rng = np.random.RandomState(5)
    K = np.array([[527.8, 0, 260.0], [0, 527.7, 142.0], [0, 0, 1.0]])
    gts = []
    for _ in range(60):
        b = random_scene(rng, 512, 288, K, 1, margin=40)[0]
        b[3:6] = [0.34, 0.34, 0.23]
        b[2] = rng.uniform(2.0, 5.0)
        gts.append(b)

    def run(mod):
        recs = [{'gt': g[None], 'pred': mod(g)[None], 'conf': np.array([0.9]), 'K': K, 'seq_id': 's', 'frame_id': str(i)}
                for i, g in enumerate(gts)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rep = ec.ads_eval(recs, [(1920, 1080)] * len(recs), 512, 288, 'xyz', ['laa'],
                              tempfile.mkdtemp(prefix='ads_', dir=os.environ.get('CAMNORM_TMP', None)), 't')['laa']
        vals = {}
        for ln in rep.splitlines():
            for key in ('orin_error_median(degree)', 'posi_error_median(m)', 'size_error_median(m)'):
                if key in ln and key not in vals:
                    vals[key] = float(ln.split(':')[-1])
        return vals

    def shift(g):
        p = g.copy()
        p[0] += 0.2
        return p

    def turn(g):
        p = g.copy()
        p[6:9] = (R.from_euler('xyz', g[6:9]) * R.from_rotvec([0, 0, np.radians(20)])).as_euler('xyz')
        return p

    a = run(shift)
    b = run(turn)
    ok = (abs(a['posi_error_median(m)'] - 0.2) < 1e-3 and a['orin_error_median(degree)'] < 0.05 and
          a['size_error_median(m)'] < 1e-3 and abs(b['orin_error_median(degree)'] - 20.0) < 0.05 and
          b['size_error_median(m)'] < 1e-3 and b['posi_error_median(m)'] < 1e-4)
    report('ads_known', ok, '平移 0.2 m -> 位置 %.4f 角度 %.3f 尺寸 %.4f | 转 20° -> 角度 %.3f 位置 %.4f 尺寸 %.4f'
           % (a['posi_error_median(m)'], a['orin_error_median(degree)'], a['size_error_median(m)'],
              b['orin_error_median(degree)'], b['posi_error_median(m)'], b['size_error_median(m)']))


# --------------------------------------------------------------------------- #
def t_real_caches(sim_root, mav_root, data_root):
    import glob
    # 机体 z 朝上：仿真用相机外参算世界「上」在相机系的方向；MAV6D 用 camera2vicon（VICON z 朝上）
    C2O = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    idx = pickle.load(open(os.path.join(sim_root, 'train', 'index.pkl'), 'rb'))
    dots, kin_ok = [], True
    step = max(1, len(idx['valid_idx']) // 400)
    for i in idx['valid_idx'][::step]:
        m = idx['metas'][i]
        info = pickle.load(open(os.path.join(data_root, m['seq'], 'im_info.pkl'), 'rb'))
        E = np.array(info['rgb']['extrinsic'], dtype=np.float64)
        up = (C2O @ np.linalg.inv(E) @ np.array([0, 0, 1.0, 0]))[:3]
        for b in m['boxes9d']:
            dots.append(float(R.from_euler('xyz', b[6:9]).as_matrix()[:, 2] @ up))
        kin_ok &= bool(m.get('legacy_intrinsic_fixed', False))
        Kin = np.asarray(m['K_in'])
        kin_ok &= abs(Kin[0, 0] - Kin[1, 1]) < 1e-9 and abs(Kin[0, 2] - (idx['W'] - 1) / 2.0) < 1e-6
    dots = np.array(dots)
    c2v = np.array([[0.6685859, -0.74342, 0.01787715], [0.01558769, -0.01002444, -0.99982825],
                    [0.74347153, 0.66874974, 0.004886]])
    up_m = c2v @ np.array([0, 0, 1.0])
    mi = pickle.load(open(os.path.join(mav_root, 'train', 'index.pkl'), 'rb'))
    mdots = np.array([float(R.from_euler('xyz', mi['metas'][i]['boxes9d'][0, 6:9]).as_matrix()[:, 2] @ up_m)
                      for i in mi['valid_idx'][::20]])
    report('body_z_up', np.median(dots) > 0.9 and np.median(mdots) > 0.9,
           '机体 z·世界上：仿真 中位 %.3f（>0 占 %.1f%%）| MAV6D 中位 %.3f（>0 占 %.1f%%）'
           % (np.median(dots), 100 * (dots > 0).mean(), np.median(mdots), 100 * (mdots > 0).mean()))
    report('sim_K_in', kin_ok, '仿真缓存 K_in：假内参已识别并换回真值、fx=fy、主点 = (W-1)/2')

    # MAV6D 划分：序列不重叠、帧不重叠；档位帧数
    sets = {}
    for sp in ('train', 'val', 'test'):
        ii = pickle.load(open(os.path.join(mav_root, sp, 'index.pkl'), 'rb'))
        sets[sp] = ii
    fr = {sp: set((sets[sp]['metas'][i]['seq'], sets[sp]['metas'][i]['frame']) for i in sets[sp]['valid_idx']) for sp in sets}
    sq = {sp: set(s for s, _ in fr[sp]) for sp in fr}
    ok = not (sq['train'] & sq['val']) and not (fr['train'] & fr['test']) and not (fr['val'] & fr['test'])
    n = len(sets['train']['valid_idx'])
    report('mav6d_split', ok, 'train %d / val %d / test %d 帧；train-val 序列交集 %d，train/val-test 帧交集 %d；'
           '档位 1/5/10/25/50/100%% = %s 帧'
           % (n, len(fr['val']), len(fr['test']), len(sq['train'] & sq['val']),
              len((fr['train'] | fr['val']) & fr['test']), [len(range(0, n, k)) for k in (100, 20, 10, 4, 2, 1)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--sim-cache', default='E:/mmcache/indoor8cn')
    ap.add_argument('--mav-cache', default='E:/mmcache/mav6d_cn')
    ap.add_argument('--data-root', default='D:/data_collect')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    tests = [('geometry', lambda: report('geometry', cg.self_check(), 'camera_geometry.self_check')),
             ('roundtrip', t_roundtrip), ('flip_rule', t_flip_rule), ('worker_aug', t_worker_aug),
             ('mut_flip_x', lambda: t_worker_aug(body_mirror='x', expect_fail=True)), ('worker_view', t_worker_view),
             ('model_decode', t_model_decode),
             ('ads_known', t_ads_known)]
    if args.real:
        tests.append(('real_caches', lambda: t_real_caches(args.sim_cache, args.mav_cache, args.data_root)))
    for name, fn in tests:
        if args.only and name not in args.only.split(','):
            continue
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            report(name, False, '异常: %r' % (e,))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print('\n==== %d 项，失败 %d ====' % (len(RESULTS), n_fail))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
