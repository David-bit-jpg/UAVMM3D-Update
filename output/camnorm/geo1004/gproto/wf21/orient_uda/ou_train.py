# -*- coding: utf-8 -*-
"""朝向网络的域适应训练（GPU）：源域换景样本 + 目标域教师伪标签 + 对称视角原型损失。从 KZ-IRLiD 成员 kzm_s{seed} 出发续训。
每一步 = 一个源域批（仿真裁窗 8 张）+ 一个目标域批（真实 train 裁窗 4 张，可关），一次反向：
  L = L_src + λ_t · L_tgt + λ_p · (L_proto_src + L_proto_tgt)
  L_src  ：KZ 原配方（热图 + 中心偏移 + 深度 + 尺寸 + 旋转 + 关键点×12），源域裁窗以 0.5 概率"换景不换机"
           （红外⇄LiDAR 互验抠像留下仿真机身，背景换成单帧真实背景库 bank_kz_sf_v3.npy；旋转标签精确不变）。
  L_tgt  ：只有 热图×1 + 中心偏移×2 + 关键点×12；关键点项按四旋翼对称取最小：
           min_k ‖p − g(R̂·Rz(90k))‖₁，g = 伪框（t̂, lwh, R̂·Rz(90k)）8 个角点在裁窗里相对中心的投影 / 64（目标是刚体投影）。
           伪框来自 pseudo_build.py：R̂ = 红外 / LiDAR 蒸馏过的 9 个成员的 SA-TM 融合，t̂ = 同一融合关键点的 PnP 平移。
           --tgt gate：只用红外⇄LiDAR 与教师⇄纯彩色都一致（≤10°）的帧；--tgt all：全部帧（消融）；--tgt none：不用目标域。
           每个种子只用目标帧池的随机 80%（种子 = 学生编号，全局随机、不按片段；保持成员多样性）。
  L_proto：RGB 特征里 24 个四旋翼对称视角原型（仰角 6 档 × 方位按 90° 折叠 4 档）。原型只由源域样本（真值视角）的机身特征 EMA 更新；
           每个样本的软目标 q_j ∝ exp(−d_C4(v, v_j)²/2σ²)；损失 KL(q ‖ softmax(cos(f, μ)/τ))。
           源域样本权重 1；目标域样本的视角来自伪框，权重 = 互验可信度 w = exp(−(d_IL² + d_TR²)/2·10²)。
           作用：把"同一视角"的真实机身特征拉向由仿真精确标签定义、跨背景平均的原型（按视角条件的跨域对齐）。
           --proto-metric ring（11-08 夜加，缺省 cos 不变）：度量换成域内标准化 + 对角马氏距离、损失换成环形加权类内/类间形式（ViewProtoRing，wf21/proto_var）。
输出 runs/<tag>_s<seed>/ckpt_last.pth（部署格式：只含 KZ 网络权重）+ train.log + DONE.json"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse, json, time, types
import numpy as np, cv2
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
ROOT = '@@ROOT@@'
HERE = os.path.dirname(os.path.abspath(__file__))
G = ROOT + '/output/camnorm/geo1004'
DESIGN = G + '/gproto/wf20/design'
for p in (DESIGN, ROOT, ROOT + '/tools'):
    if p not in sys.path:
        sys.path.insert(0, p)
SCRP = '@@ROOT@@/output/scratch'
import kzpaste_train as KP     # noqa: E402  (CacheDS / SymViewProto / to_device；import 时 chdir 到 tools)
from uavdet3d.datasets.dataset import DatasetTemplate                                  # noqa: E402
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset           # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import DataPreProcessor, make_kp2d_map   # noqa: E402
from scipy.spatial.transform import Rotation as Rot                                    # noqa: E402

RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
TGT_AUG = {'hflip': 0.5, 'scale': [1.0, 1.0], 'photometric': True, 'noise': 0.01, 'blur': 0.0, 'blur_prob': 0.5,
           'target_ring': 0.0, 'target_blur': 0.0}         # = 10-03 KZ-ST 第二段的真实裁窗增广（不做目标级抹除/拖糊）
TGT_LW = {'hm': 1.0, 'center_res': 2.0, 'center_dis': 0.0, 'dim': 0.0, 'rot': 0.0, 'kp2d': 12.0}


class RealPseudoDS(DatasetTemplate):
    """一个样本 = 一张真实 train 部署裁窗 + 伪框（4 个四旋翼对称等价朝向一起过几何增广）。"""
    _augment = MMCache_Det_Dataset._augment

    def __init__(self, dataset_cfg, crops, P, idx, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=True, root_path=None, logger=logger)
        self.data_pre_processor = DataPreProcessor(self.dataset_cfg, training=True)
        self.class_names = list(dataset_cfg.CLASS_NAMES)
        self.agnostic_name = list(dataset_cfg.HM_CLASS_NAMES)[0]
        self.modalities = ['rgb']
        self.teacher_rgb_path = None
        self.euler_seq = 'xyz'
        self.flip_body_mirror = str(dataset_cfg.get('FLIP_BODY_MIRROR_AXIS', 'y'))
        self.new_im_width, self.new_im_hight = int(dataset_cfg.IM_RESIZE[0]), int(dataset_cfg.IM_RESIZE[1])
        self.W, self.H = self.new_im_width, self.new_im_hight
        self.stride = int(dataset_cfg.STRIDE)
        self.im_num = 1
        self.aug = dict(TGT_AUG)
        self.norm_mean = np.array(dataset_cfg.NORM_MEAN, np.float32)
        self.norm_std = np.array(dataset_cfg.NORM_STD, np.float32)
        self.crops, self.P = crops, P
        self.valid_idx = np.asarray(idx, np.int64)
        self.rc = None            # wf23 --realcut：真实机身换景（教师抠图 + 红外⇄LiDAR 互验过滤），缺省关
        self.n_rc = 0

    def set_realcut(self, prob, masks, passed, bank, seed):
        self.rc = {'p': float(prob), 'A': masks, 'ok': np.asarray(passed, bool), 'bank': bank}
        self._rrng = np.random.default_rng(30000 + seed)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, item):
        n = int(self.valid_idx[item])
        ci = int(self.P['idx'][n])
        view = np.ascontiguousarray(self.crops[ci])
        if self.rc is not None:
            u = float(self._rrng.random()); jb = int(self._rrng.integers(1 << 30)); fl = bool(self._rrng.random() < 0.5)
            if u < self.rc['p'] and self.rc['ok'][ci]:
                a = np.asarray(self.rc['A'][ci], np.float32)[..., None] / 255.0
                bg = np.asarray(self.rc['bank'][jb % len(self.rc['bank'])])
                if fl:
                    bg = bg[:, ::-1]
                view = np.clip(a * view.astype(np.float32) + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                self.n_rc += 1
        image = view.astype(np.float32).transpose(2, 0, 1) / 255.0
        R = self.P['R'][n]
        t = self.P['t_pnp'][n]
        lwh = self.P['lwh'][n]
        boxes = np.stack([np.concatenate([t, lwh, Rot.from_matrix(R @ RZ[q]).as_euler('xyz')]) for q in range(4)]).astype(np.float64)
        names = [self.agnostic_name] * 4
        K = np.asarray(self.P['K_crop'][n], np.float64).copy()
        image, boxes, names, K = self._augment(image, boxes, names, K)
        assert len(boxes) == 4
        image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        dd = {'image': image[None], 'gt_box9d': boxes[:1].reshape(-1, 9).astype(np.float32), 'gt_name': np.array(names[:1]),
              'intrinsic': np.array([K], dtype=np.float64), 'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
              'distortion': np.zeros((1, 5), dtype=np.float32), 'raw_im_size': np.array([self.W, self.H]), 'new_im_size': np.array([self.W, self.H]),
              'obj_size': np.array(self.dataset_cfg.get('OB_SIZE', [[1.0, 1.0, 1.0]])), 'stride': self.stride,
              'scene_id': str(self.P['seq'][n]).split('/')[0], 'seq_id': str(self.P['seq'][n]), 'frame_id': str(self.P['frame'][n])}
        dd = self.data_pre_processor(dd)
        alts = [make_kp2d_map(boxes[q:q + 1], [K], np.zeros((1, 5)), self.W, self.H, self.stride, 1, euler_seq='xyz', use_distortion=False)
                for q in range(4)]
        k0 = np.asarray(dd['kp2d'], np.float32)
        assert k0.shape == alts[0].shape and np.allclose(k0, alts[0], atol=1e-4), (k0.shape, alts[0].shape, float(np.abs(k0 - alts[0]).max()))
        dd['kp2d'] = np.concatenate(alts, axis=-3).astype(np.float32)          # (1, 64, h, w)
        dd['pw'] = np.array([float(self.P['w'][n])], np.float32)
        return dd


def _mker(L, ang):
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2.0
    for t in np.linspace(-c, c, 4 * L):
        x, y = int(round(c + t * np.cos(ang))), int(round(c + t * np.sin(ang)))
        k[min(max(y, 0), L - 1), min(max(x, 0), L - 1)] = 1
    return k / k.sum()


class FgCacheDS(KP.CacheDS):
    """wf23 --fg-col / --fg-blur：前景中间域。源域换景样本（仿真机身 + 真实背景，标签精确）上再把机身的颜色明暗换成
    教师从真实帧里抠出的真实机身统计（随机挑一个真实机身，按 RGB 均值/标准差重上色），并以概率加运动模糊（机身层与蒙版一起拖影）。
    与 KP.CacheDS.__getitem__ 逐行相同，只在换景分支里多了前景操作；两个概率都为 0 时与父类逐位相同（前景随机数独立）。"""

    def set_fg(self, col, pc, pb, L0, L1, seed):
        self.fg = dict(col=col, pc=float(pc), pb=float(pb), L0=int(L0), L1=int(L1))
        self._frng = np.random.default_rng(40000 + seed)
        self.n_fgc = 0
        self.n_fgb = 0

    def __getitem__(self, item):
        if self._c is None:
            self._c = {k: np.load('%s/%s.npy' % (self.cache, k), mmap_mode='r') for k in ('rgb', 'a_v', 'a_ir')}
        n = int(self.valid_idx[item])
        cm = self.cmeta[n]
        view = np.asarray(self._c['rgb'][n])
        xp_flag = 0.0
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
                    x = view.astype(np.float32)
                    fg = getattr(self, 'fg', None)
                    if fg is not None:
                        r = self._frng
                        if fg['col'] is not None and r.random() < fg['pc']:
                            m = a[..., 0] > 0.5
                            if m.sum() > 20:
                                t = fg['col'][int(r.integers(len(fg['col'])))]
                                Y = x.mean(2, keepdims=True)                                     # 亮度按真实机身统计重设；色彩偏离压到 0.3（真实两型机都近似无彩色）
                                sc = float(np.clip(np.mean(t[3:6]) / (float(Y[m].std()) + 1e-3), 0.3, 3.0))
                                cdev = (x - Y) - (x[m] - Y[m]).mean(0)
                                x = t[:3] + sc * (Y - float(Y[m].mean())) + 0.3 * cdev
                                self.n_fgc += 1
                        if r.random() < fg['pb']:
                            k = _mker(int(r.integers(fg['L0'], fg['L1'] + 1)), float(r.uniform(0, np.pi)))
                            xa = cv2.filter2D(x * a, -1, k)
                            a = cv2.filter2D(a[..., 0], -1, k)[..., None]
                            view = np.clip(xa + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                            self.n_fgb += 1
                        else:
                            view = np.clip(a * x + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                    else:
                        view = np.clip(a * x + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                    self.n_paste += 1
                    xp_flag = 1.0
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
        out = self.data_pre_processor(data_dict)
        if getattr(self, 'emit_xp', False):
            out['xp'] = np.array([xp_flag], np.float32)          # wf23 方案四：标记这一张是不是中间域样本（仿真机身 + 真实背景）
        return out


NO_SYM = False   # --no-sym（消融）：目标域关键点损失不取 4 个对称等价目标的最小，只对伪标签自身的那一个


def c4_kp_term(p, g, sw=None):
    """p: (B*K,16,h,w)；g: (B,K,16A,h,w)，A 个四旋翼对称等价目标。中心格上 min_a ‖p − g_a‖₁ / 16 的均值。
    sw: (B,) 每个样本的权重（--tgt-w：互验可信度 w）；None = 等权（与之前逐位相同）。"""
    B, K = g.shape[0], g.shape[1]
    A = g.shape[2] // 16
    pp = p.reshape(B, K, 16, *p.shape[-2:])
    gs = g.reshape(B, K, A, 16, *g.shape[-2:])
    m = torch.abs(gs[:, :, 0]).sum(dim=2) > 0
    if m.sum() == 0:
        return (p.reshape(-1) * 0).sum()
    l = torch.stack([torch.abs(pp - gs[:, :, a]).sum(dim=2) for a in range(A)], 0).min(0).values
    if sw is None:
        return l[m].sum() / (m.sum() * 16)
    wc = sw.reshape(B, 1, 1, 1).expand_as(l)
    return (l * wc)[m].sum() / (wc[m].sum().clamp(min=1e-6) * 16)


_P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
_PI = [int(np.argmin(np.abs(_P8 - _P8[i] * np.array([1, -1, 1])).sum(1))) for i in range(8)]


def unmirror_kp(kp):
    """kp: (N,16,h,w) 镜像图上的角点图 → 换算回原图（与 wf23/mirror/recut_mirror.py 的部署换算逐项相同）"""
    k = torch.flip(kp, dims=[-1]).reshape(kp.shape[0], 8, 2, *kp.shape[-2:])
    k = torch.stack([-k[:, :, 0], k[:, :, 1]], 2)
    inv = [0] * 8
    for i in range(8):
        inv[_PI[i]] = i
    return k[:, inv].reshape(kp.shape[0], 16, *kp.shape[-2:])


def patched_get_loss(self):
    gt = self.forward_loss_dict['gt_center_dict']
    g = gt.get('kp2d', None)
    if g is None or g.shape[-3] == 16:
        return self._orig_get_loss()
    w = self.loss_weights.get('kp2d', 1.0)
    gt['kp2d'] = g[:, :, :16]
    self.loss_weights['kp2d'] = 0.0
    try:
        loss = self._orig_get_loss()
    finally:
        gt['kp2d'] = g
        self.loss_weights['kp2d'] = w
    term = c4_kp_term(self.forward_loss_dict['pred_center_dict']['kp2d'], g[:, :, :16] if NO_SYM else g, getattr(self, '_sample_w', None))
    self.loss_terms['kp2d'] = float(term.detach())
    return loss + w * term


class ViewProto:
    """四旋翼对称视角原型 + 原型分配损失（源域更新原型；目标域只受损失，按可信度加权）。"""

    def __init__(self, dev, sigma=15.0, tau=0.1, m=0.99, warm=100):
        el = np.radians([-75, -45, -15, 15, 45, 75]); az = np.radians([11.25, 33.75, 56.25, 78.75])
        V = np.array([[np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)] for e in el for z in az])
        self.Vk = np.stack([V @ RZ[k].T for k in range(4)])
        self.J, self.sigma, self.tau, self.m, self.warm = len(V), sigma, tau, m, warm
        self.mu = None
        self.seen = torch.zeros(self.J, dtype=torch.bool, device=dev)
        self.step = 0

    def feats_targets(self, batch):
        ft = batch['features_2d']
        ft = ft.reshape(-1, *ft.shape[-3:]).float()
        fg = batch['fg_mask'].reshape(ft.shape[0], -1, *ft.shape[-2:])[:, 0].float()
        gb = batch['gt_box9d'].detach().cpu().numpy().reshape(ft.shape[0], -1, 9)
        F_, Q, ib = [], [], []
        for b in range(ft.shape[0]):
            g = gb[b][np.abs(gb[b]).sum(1) > 0]
            if not len(g) or float(fg[b].sum()) < 1:
                continue
            t = g[0, :3]
            R = Rot.from_euler('xyz', g[0, 6:9]).as_matrix()
            v = R.T @ (-t / max(np.linalg.norm(t), 1e-6))
            d = np.degrees(np.arccos(np.clip(np.einsum('kjc,c->kj', self.Vk, v), -1, 1))).min(0)
            q = np.exp(-0.5 * (d / self.sigma) ** 2)
            Q.append(q / q.sum()); ib.append(b)
            F_.append((ft[b] * fg[b][None]).sum((1, 2)) / fg[b].sum())
        if not F_:
            return None, None, None
        f = torch.nn.functional.normalize(torch.stack(F_), dim=1)
        q = torch.as_tensor(np.stack(Q), dtype=torch.float32, device=f.device)
        return f, q, ib

    def update(self, f, q):
        with torch.no_grad():
            if self.mu is None:
                self.mu = torch.zeros(self.J, f.shape[1], device=f.device)
            wsum = q.sum(0)
            upd = (q.T @ f.detach()) / wsum.clamp(min=1e-6)[:, None]
            ok = wsum > 1e-3
            new = torch.where(self.seen[:, None], self.m * self.mu + (1 - self.m) * upd, upd)
            self.mu = torch.where(ok[:, None], torch.nn.functional.normalize(new, dim=1), self.mu)
            self.seen |= ok

    def loss(self, f, q, w=None, dom='src'):
        logit = (f @ self.mu.T) / self.tau
        logit = logit.masked_fill(~self.seen[None], -1e4)
        lp = torch.log_softmax(logit, 1)
        kl = (q * (torch.log(q.clamp(min=1e-8)) - lp)).sum(1)
        if w is None:
            return kl.mean()
        return (kl * w).sum() / w.sum().clamp(min=1e-6)


class ViewProtoRing(ViewProto):
    """--proto-metric ring（wf21/proto_var，11-08 夜）：同样 24 个对称视角原型、同样软目标 q（σ 15°）、同样只由源域真值更新，
    但度量换成"域内标准化 + 对角马氏距离"，损失换成环形加权的类内 / 类间形式：
      z = (f − m_d)/√(v_d + ε)，m_d / v_d 为源域、目标域各自的按维滑动均值方差（动量 mom_stat，不回传梯度）；
      μ_j = 源域 z 按 q 加权的滑动平均（不归一化）；s² = 源域池化类内对角方差的滑动平均，s̃² = (1−shrink)s² + shrink；
      d_k = mean_c (z_c − μ_kc)²/s̃_c²；
      L = −Σ_k q_k log[e^{−d_k/τ} / (e^{−d_k/τ} + Σ_{j≠k} ω_kj e^{−d_j/τ})]，ω_kj = 1 − exp(−Δ(c_k,c_j)²/2σ_ω²)，Δ = 原型中心的四旋翼对称角距离。
    τ 由仿真留出温度标定（proto_var/calib_tau.py），σ_ω、shrink、mom_stat 预先写死。"""

    def __init__(self, dev, tau=0.04, sig_w=30.0, shrink=0.3, mom_stat=0.99, **kw):
        super().__init__(dev, tau=tau, **kw)
        V = self.Vk[0]
        DC = np.degrees(np.arccos(np.clip(np.einsum('kjc,lc->klj', self.Vk, V), -1, 1))).min(0)     # (J,J)
        Wr = 1.0 - np.exp(-0.5 * (DC / sig_w) ** 2)
        np.fill_diagonal(Wr, 1.0)                     # 分母里 k 自己的项权重 1
        self.logW = torch.as_tensor(np.log(np.maximum(Wr, 1e-12)), dtype=torch.float32, device=dev)
        self.shrink, self.mom_stat = shrink, mom_stat
        self.stats = {'src': None, 'tgt': None}
        self.s2 = None

    def observe(self, f, dom):
        with torch.no_grad():
            fd = f.detach().float()
            if self.stats[dom] is None:
                m = fd.mean(0)
                v = ((fd - m) ** 2).mean(0) if fd.shape[0] > 1 else torch.ones_like(m)
            else:
                m0, v0 = self.stats[dom]
                m = self.mom_stat * m0 + (1 - self.mom_stat) * fd.mean(0)
                v = self.mom_stat * v0 + (1 - self.mom_stat) * ((fd - m) ** 2).mean(0)
            self.stats[dom] = (m, v)

    def standardize(self, f, dom):
        m, v = self.stats[dom]
        return (f - m) / torch.sqrt(v + 1e-6)

    def update(self, f, q):
        self.observe(f, 'src')
        with torch.no_grad():
            z = self.standardize(f.detach(), 'src')
            if self.mu is None:
                self.mu = torch.zeros(self.J, z.shape[1], device=z.device)
            wsum = q.sum(0)
            upd = (q.T @ z) / wsum.clamp(min=1e-6)[:, None]
            ok = wsum > 1e-3
            new = torch.where(self.seen[:, None], self.m * self.mu + (1 - self.m) * upd, upd)
            self.mu = torch.where(ok[:, None], new, self.mu)
            self.seen |= ok
            res = (q[:, :, None] * (z[:, None, :] - self.mu[None]) ** 2).sum((0, 1)) / q.sum().clamp(min=1e-6)
            self.s2 = res if self.s2 is None else self.m * self.s2 + (1 - self.m) * res

    def loss(self, f, q, w=None, dom='src'):
        z = self.standardize(f, dom)
        s2 = (1 - self.shrink) * self.s2 + self.shrink
        d = (((z[:, None, :] - self.mu[None]) ** 2) / s2[None, None]).mean(2)          # (B,J)
        lg = (-d / self.tau).masked_fill(~self.seen[None], -1e4)
        logW = self.logW.masked_fill(~self.seen[None], -1e4)                           # 未见原型不进分母
        logden = torch.logsumexp(lg[:, None, :] + logW[None], dim=2)                    # (B,K)：e_k + Σ_{j≠k} ω_kj e_j
        qs = q * self.seen[None].float()
        qs = qs / qs.sum(1, keepdim=True).clamp(min=1e-8)
        l = -(qs * (lg - logden)).sum(1)
        if w is None:
            return l.mean()
        return (l * w).sum() / w.sum().clamp(min=1e-6)


def main():
    global torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--tgt', default='gate', choices=['gate', 'all', 'none'])
    ap.add_argument('--proto', type=float, default=0.0)
    ap.add_argument('--proto-metric', default='cos', choices=['cos', 'ring'],
                    help='cos = 原余弦 + KL（缺省，行为不变）；ring = 域内标准化 + 对角马氏 + 环形权重（wf21/proto_var）')
    ap.add_argument('--ring-tau', type=float, default=0.04)
    ap.add_argument('--ring-sigw', type=float, default=30.0)
    ap.add_argument('--ring-shrink', type=float, default=0.3)
    ap.add_argument('--ring-mom', type=float, default=0.99)
    ap.add_argument('--runs-dir', default='', help='输出目录（缺省 = 本目录 runs/）')
    ap.add_argument('--pseudo', default='pseudo_train', help='pseudo/<名>.npz；pseudo_train_rgb = 只用纯彩色组的伪朝向（消融）')
    ap.add_argument('--tgt-w', action='store_true', help='目标域关键点损失按互验可信度 w 逐样本加权（事后加的臂）')
    ap.add_argument('--no-sym', action='store_true', help='消融：目标域关键点损失不做四旋翼对称取最小')
    ap.add_argument('--paste', type=float, default=0.5)
    ap.add_argument('--realcut', type=float, default=0.0, help='wf23：目标域真实裁窗以此概率"真实机身换景"（只用过滤通过的帧；伪框不变）')
    ap.add_argument('--realcut-dir', default=G + '/gproto/wf23/realcut')
    ap.add_argument('--fg-col', type=float, default=0.0, help='wf23 前景中间域：换景样本以此概率按教师统计的真实机身颜色重上色')
    ap.add_argument('--fg-blur', type=float, default=0.0, help='wf23 前景中间域：换景样本以此概率加运动模糊')
    ap.add_argument('--fg-L', default='4,16', help='运动模糊长度范围（像素）')
    ap.add_argument('--proto-xp', action='store_true', help='wf23 方案四：视角原型只用中间域样本（仿真机身 + 真实背景，标签精确）更新；损失照常')
    ap.add_argument('--mir', type=float, default=0.0, help='wf23：真实帧镜像等变一致损失——同一真实裁窗与其左右镜像各前向一次，镜像输出精确换算回原图（格子列 c→63−c、角点水平偏移取反、角点按机身 y 轴镜像重排）后与原图输出在伪标签中心格上 L1 一致')
    ap.add_argument('--steps', type=int, default=2250)
    ap.add_argument('--bs-src', type=int, default=8)
    ap.add_argument('--bs-tgt', type=int, default=4)
    ap.add_argument('--lam-t', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--bag', type=float, default=0.8)
    ap.add_argument('--init', default='kzm')
    ap.add_argument('--mem', type=float, default=0.3)
    ap.add_argument('--max-steps-smoke', type=int, default=0)
    a = ap.parse_args()
    global NO_SYM
    NO_SYM = bool(a.no_sym)
    import torch as _t
    torch = _t
    from torch.utils.data import DataLoader
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    from tools.train_utils.optimization import build_optimizer, build_scheduler
    torch.cuda.set_per_process_memory_fraction(a.mem)
    run = os.path.join(a.runs_dir or os.path.join(HERE, 'runs'), '%s_s%d%s' % (a.tag, a.seed, '_smoke' if a.max_steps_smoke else ''))
    os.makedirs(run, exist_ok=True)
    logger = common_utils.create_logger(os.path.join(run, 'train_%s.log' % time.strftime('%Y%m%d-%H%M%S')))
    logger.info('args %s' % vars(a))
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml', cfg)
    cfg.OPTIMIZATION.LR = a.lr
    common_utils.set_random_seed(a.seed)
    xbg = {'PROB': a.paste, 'BANK': SCRP + '/ou/bank_kz_sf_v3.npy', 'MATTE': 'v'} if a.paste > 0 else None
    src = (FgCacheDS if (a.fg_col > 0 or a.fg_blur > 0 or a.proto_xp) else KP.CacheDS)(cfg.DATA_CONFIG, SCRP + '/kz/crops_6000', training=True, logger=logger, xbg=xbg, seed=a.seed)
    if a.proto_xp:
        assert xbg is not None and a.proto > 0
        src.emit_xp = True
        logger.info('方案四：视角原型只用中间域样本更新')
    if a.fg_col > 0 or a.fg_blur > 0:
        assert xbg is not None, '前景中间域只加在换景样本上'
        L0, L1 = (int(v) for v in a.fg_L.split(','))
        src.set_fg(np.load(G + '/gproto/wf23/fgstats_kz.npz')['col_real'], a.fg_col, a.fg_blur, L0, L1, a.seed)
        logger.info('前景中间域：重上色 %.2f（教师统计的真实机身 %d 个）、运动模糊 %.2f（%d–%d 像素）' % (a.fg_col, len(np.load(G + '/gproto/wf23/fgstats_kz.npz')['col_real']), a.fg_blur, L0, L1))
    g_src = torch.Generator(); g_src.manual_seed(a.seed)
    L_src = DataLoader(src, batch_size=a.bs_src, shuffle=True, num_workers=0, collate_fn=src.collate_batch, drop_last=True, generator=g_src)
    L_tgt = None
    if a.tgt != 'none':
        P = dict(np.load(HERE + '/pseudo/%s.npz' % a.pseudo, allow_pickle=False))
        crops = np.load(SCRP + '/ou/train_crops_4289.npy', mmap_mode='r')
        pool = np.where(P['gate'])[0] if a.tgt == 'gate' else np.arange(len(P['idx']))
        rng = np.random.default_rng(1000 + a.seed)
        pool = np.sort(rng.choice(pool, int(round(a.bag * len(pool))), replace=False))
        tgt = RealPseudoDS(cfg.DATA_CONFIG, crops, P, pool, logger=logger)
        if a.realcut > 0:
            qz = np.load(a.realcut_dir + '/q_kz_train.npz')
            tgt.set_realcut(a.realcut, np.load(a.realcut_dir + '/masks_kz_train.npy', mmap_mode='r'), qz['passed'],
                            np.load(SCRP + '/ou/bank_kz_sf_v3.npy', mmap_mode='r'), a.seed)
            logger.info('真实机身换景：概率 %.2f，帧池里通过过滤 %d / %d' % (a.realcut, int(qz['passed'][P['idx'][pool]].sum()), len(pool)))
        g_t = torch.Generator(); g_t.manual_seed(2000 + a.seed)
        L_tgt = DataLoader(tgt, batch_size=a.bs_tgt, shuffle=True, num_workers=0, collate_fn=tgt.collate_batch, drop_last=True, generator=g_t)
        logger.info('目标域：%s，帧池 %d（gate %d / 全部 %d），随机 %.0f%% → %d 帧' % (a.tgt, len(pool), int(P['gate'].sum()), len(P['idx']), 100 * a.bag, len(pool)))
    model = build_network(model_cfg=cfg.MODEL, dataset=src)
    init = {'kzm': G + '/lidar/irlid/release/kzm_s%d.pth', 'kz': ROOT + '/output/release_20260923/kz_s%d.pth'}[a.init] % a.seed
    n_ld, n_tot = model.load_params_from_file(filename=init, to_cpu=False, logger=logger)
    assert n_ld == n_tot, (n_ld, n_tot)
    model.cuda()
    head = model.dense_head_2d
    head._orig_get_loss = head.get_loss
    head.get_loss = types.MethodType(patched_get_loss, head)
    src_lw = dict(head.loss_weights)
    steps = a.max_steps_smoke or a.steps
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    sched, _ = build_scheduler(optimizer, total_iters_each_epoch=steps, total_epochs=1, last_epoch=-1, optim_cfg=cfg.OPTIMIZATION)
    dev = torch.device('cuda')
    if a.proto <= 0:
        PR = None
    elif a.proto_metric == 'ring':
        PR = ViewProtoRing(dev, tau=a.ring_tau, sig_w=a.ring_sigw, shrink=a.ring_shrink, mom_stat=a.ring_mom)
    else:
        PR = ViewProto(dev)
    logger.info('init %s | steps %d | 源域批 %d 换景 %s | 目标域批 %d λ_t %.2f | 原型 λ_p %.3f | LR %.1e | 源域损失权重 %s | 目标域损失权重 %s'
                % (init, steps, a.bs_src, xbg, a.bs_tgt if L_tgt else 0, a.lam_t, a.proto, a.lr, src_lw, TGT_LW))
    it_s, it_t = iter(L_src), (iter(L_tgt) if L_tgt else None)

    def nxt(it, L):
        try:
            return next(it), it
        except StopIteration:
            it = iter(L)
            return next(it), it
    t0 = time.time()
    acc = {}
    model.train()
    for step in range(steps):
        sched.step(step)
        b_s, it_s = nxt(it_s, L_src)
        xpm = np.asarray(b_s.pop('xp'), np.float32).reshape(-1) if 'xp' in b_s else None
        KP.to_device(b_s, dev, torch)
        head.loss_weights.clear(); head.loss_weights.update(src_lw)
        ret = model(b_s)
        loss = ret['loss']
        terms = {'src_' + k: v for k, v in (head.loss_terms or {}).items()}
        if PR is not None:
            f, q, ib_s = PR.feats_targets(b_s)
            if f is not None:
                if xpm is not None:
                    sel = torch.from_numpy(xpm[np.asarray(ib_s)] > 0.5).to(f.device)
                    if bool(sel.any()):
                        PR.update(f[sel], q[sel])
                else:
                    PR.update(f, q)
                if PR.step >= PR.warm:
                    lp = PR.loss(f, q)
                    loss = loss + a.proto * lp; terms['proto_src'] = float(lp.detach())
            PR.step += 1
        if it_t is not None:
            b_t, it_t = nxt(it_t, L_tgt)
            pw = torch.from_numpy(np.asarray(b_t.pop('pw'), np.float32).reshape(-1)).to(dev)
            KP.to_device(b_t, dev, torch)
            head.loss_weights.clear(); head.loss_weights.update(TGT_LW)
            head._sample_w = pw if a.tgt_w else None
            ret_t = model(b_t)
            head._sample_w = None
            loss = loss + a.lam_t * ret_t['loss']
            if a.mir > 0:
                p1 = head.forward_loss_dict['pred_center_dict']['kp2d']
                gk = b_t['kp2d']
                Bm, Km = gk.shape[0], gk.shape[1]
                mm = gk[:, :, :16].abs().sum(dim=2) > 0
                tb = dict(b_t)
                tb['image'] = torch.flip(b_t['image'], dims=[-1])
                for mod in model.module_list:
                    tb = mod(tb)
                p2 = unmirror_kp(head.forward_loss_dict['pred_center_dict']['kp2d'])
                if bool(mm.any()):
                    lmir = (p1 - p2).abs().reshape(Bm, Km, 16, *p1.shape[-2:]).sum(dim=2)[mm].mean() / 16
                    loss = loss + a.lam_t * a.mir * lmir
                    terms['tgt_mir'] = float(lmir.detach())
                    if os.environ.get('OU_MIRCHK'):          # 换算自检：只翻转不换角点/不取反 的分歧应明显更大
                        pn = torch.flip(head.forward_loss_dict['pred_center_dict']['kp2d'], dims=[-1])
                        terms['chk_naive'] = float(((p1 - pn).abs().reshape(Bm, Km, 16, *p1.shape[-2:]).sum(dim=2)[mm].mean() / 16).detach())
            terms.update({'tgt_' + k: v for k, v in (head.loss_terms or {}).items()})
            if PR is not None and (PR.step > PR.warm or a.proto_metric == 'ring'):
                f, q, ib = PR.feats_targets(b_t)
                if f is not None:
                    if a.proto_metric == 'ring':
                        PR.observe(f, 'tgt')          # 目标域标准化统计从第 1 步起累计
                    if PR.step > PR.warm:
                        lt = PR.loss(f, q, pw[ib], dom='tgt')
                        loss = loss + a.proto * lt; terms['proto_tgt'] = float(lt.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss at step %d' % step)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
        optimizer.step()
        model.update_global_step()
        terms['total'] = float(loss.detach())
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + v
        acc['_n'] = acc.get('_n', 0) + 1
        if (step + 1) % 150 == 0 or step + 1 == steps:
            n = acc.pop('_n')
            lr = optimizer.param_groups[0]['lr']
            logger.info('step %d/%d  %.0f s  lr %.2e  %s' % (step + 1, steps, time.time() - t0, lr,
                                                           ' '.join('%s %.4f' % (k, v / n) for k, v in sorted(acc.items()))))
            acc = {}
    model.update_global_step()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck = os.path.join(run, 'ckpt_last.pth')
    torch.save({'model_state': sd, 'epoch': 1, 'src': 'wf21/orient_uda/ou_train.py %s' % vars(a)}, ck)
    J = dict(vars(a), init=init, ckpt=ck, seconds=time.time() - t0, paste_tried=src.n_try, paste_done=src.n_paste, realcut_done=(tgt.n_rc if (L_tgt is not None) else 0), fg_col_done=getattr(src, 'n_fgc', 0), fg_blur_done=getattr(src, 'n_fgb', 0),
             n_tgt=(len(L_tgt.dataset) if L_tgt else 0), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(J, open(os.path.join(run, 'DONE.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    logger.info('DONE %s' % J)


if __name__ == '__main__':
    main()
