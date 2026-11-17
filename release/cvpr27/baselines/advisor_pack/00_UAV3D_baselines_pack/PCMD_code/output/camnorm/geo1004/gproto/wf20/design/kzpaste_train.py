# -*- coding: utf-8 -*-
"""前提检查 C2（GPU 小训练）：朝向网络 KZ 用"跨域贴图（换景不换机）"续训，最小版。
臂：
  N   = 发布版 kz_s{seed} 原配方续训 E 轮（对照，排除"多训几轮"的作用）
  P   = 同上 + 训练样本以 PROB 概率换背景：仿真机身按【红外⇄LiDAR 互验抠像】留下，背景换成 KZ 尺度单帧真实背景（bank_kz_sf.npy）
  PI  = 同 P，但用红外蒙版（不经 LiDAR 体检），看互验那一步有没有用（可选）
换景发生在裁窗之后、光度增广之前：机身像素与几何完全不动 → 旋转/角点标签精确不变；用独立随机数流（default_rng(10000+seed)），
所以 N / P 两臂的裁窗、翻转、光度增广、样本顺序逐位相同，差别只有"背景是不是被换掉"。
网络只看彩色（3 通道）；红外 / LiDAR 标签只在数据加载里用来抠像，从不进网络。
用法（任意 cwd）：python kzpaste_train.py --arm P --seed 0 --epochs 2
输出 wf20/design/runs/<arm>_s<seed>/ckpt/checkpoint_epoch_<E>.pth + train.log + DONE.json"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse, json, time, datetime
from functools import partial
from pathlib import Path
import numpy as np, cv2
ROOT = '@@ROOT@@'
HERE = ROOT + '/output/camnorm/geo1004/gproto/wf20/design'
for p in (ROOT, ROOT + '/tools'):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT + '/tools')
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset, SIDECAR_FILE


def el(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def matte(ir, tag, rgb, mode):
    """ir: (H,W) uint8，tag: (H,W) 0/1，rgb: (H,W,3) uint8（同一裁窗）→ 软 alpha (H,W,1) float32，或 None（证据不足不换景）。
    红外蒙版：LiDAR 点外扩 3 px 区域里，|IR − 外环中位| > max(0.04, 3 MAD)（同 pastecheck 补充 1）。
    mode 'v'：LiDAR 查红外——剪影里没有 LiDAR 支撑（点间距自适应外扩 + 闭运算）且颜色更像外环背景的像素去掉（同 wf16 make_gate）。"""
    t = (tag > 0).astype(np.uint8)
    if t.sum() < 15:
        return None
    reg = cv2.dilate(t, el(3)) > 0
    ring = (cv2.dilate(t, el(10)) > 0) & ~reg
    if ring.sum() < 20:
        return None
    irf = ir.astype(np.float32) / 255.0
    med = np.median(irf[ring]); mad = np.median(np.abs(irf[ring] - med))
    M = reg & (np.abs(irf - med) > max(0.04, 3 * mad))
    area = int(M.sum())
    if area < 20:
        return None
    if mode == 'v':
        nh = int((t.astype(bool) & (cv2.dilate(M.astype(np.uint8), el(3)) > 0)).sum())
        r = int(np.clip(np.ceil(np.sqrt(area / max(nh, 1))), 1, 8))
        Rs = cv2.morphologyEx(cv2.dilate(t, el(r)), cv2.MORPH_CLOSE, el(2 * r)) > 0
        halo = M & ~Rs
        if halo.any():
            ringc = (cv2.dilate(M.astype(np.uint8), el(4)) > 0) & ~(cv2.dilate(M.astype(np.uint8), el(1)) > 0)
            core = (cv2.erode(M.astype(np.uint8), el(1)) > 0) & Rs
            if ringc.sum() >= 10 and core.sum() >= 10:
                imf = rgb.astype(np.float32)
                mr, mc = imf[ringc].mean(0), imf[core].mean(0)
                if np.linalg.norm(mr - mc) > 20:
                    hy, hx = np.nonzero(halo)
                    ph = imf[hy, hx]
                    bgl = np.linalg.norm(ph - mr, axis=1) < np.linalg.norm(ph - mc, axis=1)
                    M = M.copy(); M[hy[bgl], hx[bgl]] = False
    return cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.0)[..., None]


class KZPasteDataset(MMCache_Det_Dataset):
    """MMCache_Det_Dataset（jpeg 存储）+ 裁窗后换景。MODALITIES 必须是 ['rgb']；红外 / 标签边车只读来抠像。"""

    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None, xbg=None, seed=0):
        super().__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)
        assert self.modalities == ['rgb'] and self.store == 'jpeg'
        self.xbg = xbg
        self._xrng = np.random.default_rng(10000 + seed)
        self._bank = None
        self.n_paste = 0
        self.n_try = 0
        if xbg:
            for m in ('ir', 'tag'):
                assert os.path.exists(os.path.join(self.split_dir, SIDECAR_FILE[m] + '.bin'))

    def _open(self):
        mm = super()._open()
        if self.xbg and 'ir_jpg' not in mm:
            for m in ('ir', 'tag'):
                f = SIDECAR_FILE[m]
                mm[f] = np.memmap(os.path.join(self.split_dir, f + '.bin'), dtype=np.uint8, mode='r')
                mm[f + '_index'] = np.load(os.path.join(self.split_dir, f + '_index.npy'))
        return mm

    def __getitem__(self, item):
        i = int(self.valid_idx[item])
        mm = self._open()
        meta = self.metas[i]
        boxes = meta['boxes9d'].astype(np.float64).copy()
        names = list(meta['names'])
        if self.agnostic_name is not None:
            names = [self.agnostic_name] * len(names)
        K = self.cache_intrinsic(meta)
        o, ln = mm['rgb_jpg_index'][i]
        src = cv2.imdecode(np.asarray(mm['rgb_jpg'][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        do = False
        if self.xbg and self.training:
            u = float(self._xrng.random())              # 独立随机数流：每个样本都抽一次，不论贴不贴
            jb = int(self._xrng.integers(1 << 30))
            fl = bool(self._xrng.random() < 0.5)
            do = u < float(self.xbg['PROB'])
        if do:
            extra = []
            for m in ('ir', 'tag'):
                f = SIDECAR_FILE[m]
                o2, l2 = mm[f + '_index'][i]
                if l2 <= 0:
                    extra = None
                    break
                e = cv2.imdecode(np.asarray(mm[f][int(o2):int(o2) + int(l2)]), cv2.IMREAD_UNCHANGED)
                if e.ndim == 3:
                    e = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
                extra.append((m, e))
            if extra is None:
                view, K, boxes, names = self._view(src, K, boxes, names, item)
            else:
                view, K, boxes, names, ex = self._view(src, K, boxes, names, item, extra)
                exd = dict(ex)
                self.n_try += 1
                a = matte(exd['ir'], exd['tag'], view, self.xbg.get('MATTE', 'v'))
                if a is not None and len(boxes):
                    if self._bank is None:
                        self._bank = np.load(self.xbg['BANK'], mmap_mode='r')
                    bg = np.asarray(self._bank[jb % len(self._bank)])
                    if fl:
                        bg = bg[:, ::-1]
                    v0 = view
                    view = np.clip(a * view.astype(np.float32) + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                    self.n_paste += 1
                    dd = os.environ.get('KP_DUMP')
                    if dd and self.n_paste <= 12:
                        cv2.imwrite('%s/paste_%02d.jpg' % (dd, self.n_paste), np.hstack([v0, view, (np.repeat(a, 3, 2) * 255).astype(np.uint8)]))
        else:
            view, K, boxes, names = self._view(src, K, boxes, names, item)
        image = (view.astype(np.float32).transpose(2, 0, 1) / 255.0)
        rot_ok = None
        if self.training and self.aug:
            if self.aug.get('rot_mask_damaged', False):
                image, boxes, names, K, rot_ok = self._augment(image, boxes, names, K, with_rot_ok=True)
            else:
                image, boxes, names, K = self._augment(image, boxes, names, K)
        if self.norm_mean is not None:
            image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        image = image[None]
        data_dict = {
            'image': image, 'gt_box9d': boxes.reshape(-1, 9).astype(np.float32), 'gt_name': np.array(names),
            'intrinsic': np.array([K], dtype=np.float64), 'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
            'distortion': np.zeros((1, 5), dtype=np.float32),
            'raw_im_size': np.array([self.new_im_width, self.new_im_hight]), 'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.get('OB_SIZE', [[1.0, 1.0, 1.0]])), 'stride': self.stride,
            'scene_id': meta['seq'].split('/')[0], 'seq_id': meta['seq'], 'frame_id': meta['frame'],
        }
        if rot_ok is not None:
            data_dict['gt_rot_ok'] = np.asarray(rot_ok, bool)
        return self.data_pre_processor(data_dict)


class CacheDS(MMCache_Det_Dataset):
    """从 extract_kz_crops.py 抽好的固定裁窗训练（两臂同一批裁窗）；换景、光度/翻转/目标级增广照原配方在线做。"""

    def __init__(self, dataset_cfg, cache, training=True, logger=None, xbg=None, seed=0):
        super().__init__(dataset_cfg=dataset_cfg, training=training, root_path=None, logger=logger)
        import pickle
        self.cache = cache
        self.cmeta = pickle.load(open(cache + '/meta.pkl', 'rb'))
        self.valid_idx = np.arange(len(self.cmeta))
        self.xbg = xbg
        self._xrng = np.random.default_rng(10000 + seed)
        self._c = None
        self._bank = None
        self.n_paste = 0
        self.n_try = 0

    def __getitem__(self, item):
        if self._c is None:
            self._c = {k: np.load('%s/%s.npy' % (self.cache, k), mmap_mode='r') for k in ('rgb', 'a_v', 'a_ir')}
        n = int(self.valid_idx[item])
        cm = self.cmeta[n]
        view = np.asarray(self._c['rgb'][n])
        boxes = np.asarray(cm['boxes'], np.float64).reshape(-1, 9).copy()
        names = list(cm['names'])
        K = np.asarray(cm['K'], np.float64).copy()
        if self.xbg and self.training:
            u = float(self._xrng.random()); jb = int(self._xrng.integers(1 << 30)); fl = bool(self._xrng.random() < 0.5)
            if u < float(self.xbg['PROB']) and cm['has_matte'] and len(boxes):
                self.n_try += 1
                a = np.asarray(self._c['a_v' if self.xbg.get('MATTE', 'v') == 'v' else 'a_ir'][n], np.float32)[..., None] / 255.0
                if a.max() > 0:
                    if self._bank is None:
                        self._bank = np.load(self.xbg['BANK'], mmap_mode='r')
                    bg = np.asarray(self._bank[jb % len(self._bank)])
                    if fl:
                        bg = bg[:, ::-1]
                    view = np.clip(a * view.astype(np.float32) + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                    self.n_paste += 1
        image = (view.astype(np.float32).transpose(2, 0, 1) / 255.0)
        rot_ok = None
        if self.training and self.aug:
            if self.aug.get('rot_mask_damaged', False):
                image, boxes, names, K, rot_ok = self._augment(image, boxes, names, K, with_rot_ok=True)
            else:
                image, boxes, names, K = self._augment(image, boxes, names, K)
        if self.norm_mean is not None:
            image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        data_dict = {
            'image': image[None], 'gt_box9d': boxes.reshape(-1, 9).astype(np.float32), 'gt_name': np.array(names),
            'intrinsic': np.array([K], dtype=np.float64), 'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
            'distortion': np.zeros((1, 5), dtype=np.float32),
            'raw_im_size': np.array([self.new_im_width, self.new_im_hight]), 'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.get('OB_SIZE', [[1.0, 1.0, 1.0]])), 'stride': self.stride,
            'scene_id': str(cm['seq']).split('/')[0], 'seq_id': cm['seq'], 'frame_id': cm['frame'],
        }
        if rot_ok is not None:
            data_dict['gt_rot_ok'] = np.asarray(rot_ok, bool)
        return self.data_pre_processor(data_dict)


class SymViewProto:
    """RGB 特征里的四旋翼对称视角原型 + 专门损失（只管朝向）。
    视角 v = R^T·(−t/|t|)：从机体看相机的方向（机体系）。四旋翼绕机体竖轴 4 重对称 → 方位角按 90° 折叠（商空间）。
    原型 j 的中心视角 v_j：仰角 {−75,−45,−15,15,45,75}° × 折叠方位 {11.25,33.75,56.25,78.75}°，共 24 个。
    样本特征 f = 检测头输入特征图在前景格（fg_mask）上的平均（L2 归一化）——只取机身格，背景格不进原型。
    软目标 q_j ∝ exp(−d_C4(v, v_j)² / 2σ²)，d_C4 = min_k ∠(v, Rz(90k)·v_j)（对称感知、有序：离得近的原型也给分）。
    损失 = KL(q ‖ softmax(cos(f, μ)/τ))；原型 μ_j = 按 q 加权的特征 EMA（不回传梯度）。
    换景（贴图）样本与原图样本混在同一批里：同一视角的原型由不同背景下的机身特征平均而来 → 背景无关；损失把每个样本的机身特征拉向
    "同视角、跨背景"的原型 → 网络只能靠机身判视角，靠背景的捷径被罚。"""

    def __init__(self, w, dev, torch, sigma=15.0, tau=0.1, m=0.99, warm=100):
        from scipy.spatial.transform import Rotation as Rot
        self.t, self.Rot, self.dev = torch, Rot, dev
        el = np.radians([-75, -45, -15, 15, 45, 75]); az = np.radians([11.25, 33.75, 56.25, 78.75])
        V = np.array([[np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)] for e in el for z in az])
        self.Vk = np.stack([V @ np.array([[np.cos(k * np.pi / 2), -np.sin(k * np.pi / 2), 0], [np.sin(k * np.pi / 2), np.cos(k * np.pi / 2), 0], [0, 0, 1]]).T for k in range(4)])
        self.J, self.sigma, self.tau, self.m, self.warm = len(V), sigma, tau, m, warm
        self.mu = None
        self.seen = torch.zeros(self.J, dtype=torch.bool, device=dev)
        self.step = 0

    def __call__(self, batch):
        torch = self.t
        self.step += 1
        ft = batch['features_2d']
        ft = ft.reshape(-1, *ft.shape[-3:]).float()                      # (B,C,h,w)
        fg = batch['fg_mask'].reshape(ft.shape[0], -1, *ft.shape[-2:])[:, 0].float()
        gb = batch['gt_box9d'].detach().cpu().numpy().reshape(ft.shape[0], -1, 9)
        F, Q = [], []
        for b in range(ft.shape[0]):
            g = gb[b][np.abs(gb[b]).sum(1) > 0]
            if not len(g) or float(fg[b].sum()) < 1:
                continue
            t = g[0, :3]
            R = self.Rot.from_euler('xyz', g[0, 6:9]).as_matrix()
            v = R.T @ (-t / max(np.linalg.norm(t), 1e-6))
            d = np.degrees(np.arccos(np.clip(np.einsum('kjc,c->kj', self.Vk, v), -1, 1))).min(0)
            q = np.exp(-0.5 * (d / self.sigma) ** 2)
            Q.append(q / q.sum())
            F.append((ft[b] * fg[b][None]).sum((1, 2)) / fg[b].sum())
        if not F:
            return None
        f = torch.nn.functional.normalize(torch.stack(F), dim=1)
        q = torch.as_tensor(np.stack(Q), dtype=torch.float32, device=f.device)
        with torch.no_grad():
            if self.mu is None:
                self.mu = torch.zeros(self.J, f.shape[1], device=f.device)
            wsum = q.sum(0)
            upd = (q.T @ f.detach()) / wsum.clamp(min=1e-6)[:, None]
            ok = wsum > 1e-3
            new = torch.where(self.seen[:, None], self.m * self.mu + (1 - self.m) * upd, upd)
            self.mu = torch.where(ok[:, None], torch.nn.functional.normalize(new, dim=1), self.mu)
            self.seen |= ok
        if self.step <= self.warm:
            return None
        logit = (f @ self.mu.T) / self.tau
        logit = logit.masked_fill(~self.seen[None], -1e4)
        lp = torch.log_softmax(logit, 1)
        return (q * (torch.log(q.clamp(min=1e-8)) - lp)).sum(1).mean()


def to_device(batch, dev, torch):
    skip = ['batch_size', 'scene_id', 'seq_id', 'frame_id', 'brightness', 'intrinsic', 'extrinsic', 'distortion', 'raw_im_size',
            'new_im_size', 'obj_size', 'stride', 'im_num', 'obj_num', 'gt_name', 'relative_im_path', 'setting_id', 'gt_diff']
    for key, val in batch.items():
        if isinstance(val, np.ndarray) and key not in skip:
            batch[key] = torch.from_numpy(val).float().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['N', 'P', 'PI'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--prob', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0, help='冒烟用')
    ap.add_argument('--mem', type=float, default=0.25)
    ap.add_argument('--cache', default='', help='extract_kz_crops.py 的输出目录（两臂共用同一批裁窗）')
    ap.add_argument('--tag', default='')
    ap.add_argument('--proto', type=float, default=0.0, help='对称视角原型损失权重（0 = 关，逐位同原配方）')
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    from tools.train_utils.optimization import build_optimizer, build_scheduler
    from tools.train_utils.train_utils import train_model
    torch.cuda.set_per_process_memory_fraction(a.mem)
    run = Path(HERE) / 'runs' / ('%s_s%d%s%s' % (a.arm, a.seed, a.tag, '_smoke' if a.max_frames else ''))
    (run / 'ckpt').mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(str(run / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml', cfg)
    cfg.OPTIMIZATION.LR = a.lr
    cfg.OPTIMIZATION.NUM_EPOCHS = a.epochs
    xbg = None if a.arm == 'N' else {'PROB': a.prob, 'BANK': os.environ.get('KP_BANK', HERE + '/bank_kz_sf.npy'), 'MATTE': 'v' if a.arm == 'P' else 'ir'}
    common_utils.set_random_seed(a.seed)
    if a.cache:
        ds = CacheDS(cfg.DATA_CONFIG, a.cache, training=True, logger=logger, xbg=xbg, seed=a.seed)
    else:
        ds = KZPasteDataset(cfg.DATA_CONFIG, training=True, root_path=None, logger=logger, xbg=xbg, seed=a.seed)
    if a.max_frames:
        ds.valid_idx = ds.valid_idx[:a.max_frames]
    loader = DataLoader(ds, batch_size=int(cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU), pin_memory=True, num_workers=a.workers, shuffle=True,
                        collate_fn=ds.collate_batch, drop_last=False, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=a.seed))
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    init = ROOT + '/output/release_20260923/kz_s%d.pth' % a.seed
    n_ld, n_tot = model.load_params_from_file(filename=init, to_cpu=False, logger=logger)
    assert n_ld == n_tot, (n_ld, n_tot)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    sched, _ = build_scheduler(optimizer, total_iters_each_epoch=len(loader), total_epochs=a.epochs, last_epoch=-1, optim_cfg=cfg.OPTIMIZATION)
    model.train()
    logger.info('arm %s seed %d init %s | frames %d | iters %d x %d epochs | xbg %s | LR %.1e' % (a.arm, a.seed, init, len(ds), len(loader), a.epochs, xbg, a.lr))
    dev = torch.device('cuda')

    PR = SymViewProto(a.proto, dev, torch) if a.proto > 0 else None
    if PR is not None:
        logger.info('对称视角原型损失：权重 %.3f，%d 个原型（仰角 6 档 × 方位按 90° 折叠 4 档），温度 %.2f，软目标 σ %.0f°，EMA %.3f，前 %d 步只更新原型'
                    % (a.proto, PR.J, PR.tau, PR.sigma, PR.m, PR.warm))

    def model_func(m_, batch):
        to_device(batch, dev, torch)
        ret = m_(batch)
        loss = ret['loss']
        if PR is not None:
            lp = PR(batch)
            if lp is not None:
                loss = loss + a.proto * lp
                m_.dense_head_2d.loss_terms = dict(getattr(m_.dense_head_2d, 'loss_terms', None) or {}, proto=float(lp.detach()))
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss')
        m_.update_global_step()
        return loss

    t0 = time.time()
    train_model(model, optimizer, loader, model_func=model_func, lr_scheduler=sched, optim_cfg=cfg.OPTIMIZATION, start_epoch=0,
                total_epochs=a.epochs, start_iter=0, rank=0, tb_log=None, ckpt_save_dir=run / 'ckpt', train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=a.epochs + 1, logger=logger, log_interval=200, use_amp=False)
    J = {'arm': a.arm, 'seed': a.seed, 'cache': a.cache, 'epochs': a.epochs, 'lr': a.lr, 'prob': a.prob, 'xbg': xbg, 'init': init, 'frames': len(ds),
         'iters_per_epoch': len(loader), 'paste_tried': ds.n_try, 'paste_done': ds.n_paste, 'seconds': time.time() - t0,
         'ckpt': str(run / 'ckpt' / ('checkpoint_epoch_%d.pth' % a.epochs)), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(J, open(run / 'DONE.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    logger.info('DONE %s' % J)


if __name__ == '__main__':
    main()
