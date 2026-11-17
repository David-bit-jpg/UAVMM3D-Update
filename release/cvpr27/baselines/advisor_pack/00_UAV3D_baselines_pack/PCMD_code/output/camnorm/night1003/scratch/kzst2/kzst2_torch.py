# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2 -- runtime-registered dataset + head for the LOO student training (the repo is NOT modified).

KZST2_Crop_Dataset  (registered into uavdet3d.datasets.__all__)
    one sample = one selected gated TRAIN frame for student k:
      image   = the deployed zoom-95 crop (data/crops_train_u8.npy, uint8 BGR, exactly what deployed KZ saw)
      box     = pseudo box (t = position-network centre, lwh = deployed PnP box, R = LOO teacher R_T(k)) in the crop
                camera K_crop (isotropic, cv2 pixel-centre convention)
      AUG     = MMCache_Det_Dataset._augment (the repo's own code) with the frozen AUG dict (hflip 0.5 = image flip +
                hflip_K + R -> M R S; photometric; noise 0.01; scale [1,1]; blur 0; target_ring 0; target_blur 0)
      NORM    = MAV6D NORM_MEAN / NORM_STD (as the deployed hook feeds KZ)
      targets = the repo's DataPreProcessor (center_point_encoder_mav6d: hm / center_res / center_dis / dim / rot; and
                make_kp2d_map for kp2d) on that box; then kp2d (16 ch, offset / 64 at the centre cell) is turned into the
                32-channel pair [diag(rho) g, P diag(rho) g] for quotient frames, [diag(rho) g, diag(rho) g] otherwise.
CenterHeadKZST      (registered into uavdet3d.model.dense_head_2d.__all__)
    CenterHead with one change: when the kp2d target has 2 x 16 channels, the kp2d term is
        L = sum over target cells of min(|p - g_a|_1, |p - g_b|_1) / (16 * n_cells)
    which equals the stock L1 term when g_a == g_b (agree / non-quotient frames). 16-channel targets go through the
    unchanged parent code. No new parameters -> checkpoints load into the stock CenterHead (mav6d_r34_kp.yaml).
"""
import os
import sys

sys.dont_write_bytecode = True
import numpy as np                                                   # noqa: E402

import kzst2_common as S                                             # noqa: E402

for _p in (S.ROOT, S.TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                         # noqa: E402
from scipy.spatial.transform import Rotation as Rot                  # noqa: E402

import uavdet3d.datasets as UDS                                      # noqa: E402
import uavdet3d.model.dense_head_2d as UDH                           # noqa: E402
from uavdet3d.datasets.dataset import DatasetTemplate                # noqa: E402
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import DataPreProcessor, KP_SCALE as REPO_KP_SCALE   # noqa: E402
from uavdet3d.model.dense_head_2d.center_head import CenterHead      # noqa: E402

assert float(REPO_KP_SCALE) == S.KP_SCALE


# ======================================================================================= loss
def quotient_kp_term(p, g32):
    """p: (B*K, 16, h, w) prediction; g32: (B, K, 32, h, w) pair target. Returns (term, n_cells, n_pair_cells)."""
    B, K = g32.shape[0], g32.shape[1]
    pp = p.reshape(B, K, 16, *p.shape[-2:])
    ga, gb = g32[:, :, :16], g32[:, :, 16:]
    m = torch.abs(ga).sum(dim=2) > 0                                   # (B, K, h, w) target cells
    if m.sum() == 0:
        return (p.reshape(-1) * 0).sum(), 0, 0
    la = torch.abs(pp - ga).sum(dim=2)
    lb = torch.abs(pp - gb).sum(dim=2)
    best = torch.minimum(la, lb)
    pair = (torch.abs(ga - gb).sum(dim=2) > 0) & m
    return best[m].sum() / (m.sum() * 16), int(m.sum()), int(pair.sum())


class CenterHeadKZST(CenterHead):
    def get_loss(self):
        gt = self.forward_loss_dict['gt_center_dict']
        g = gt.get('kp2d', None)
        oc = int(self.net_config['kp2d']['out_channels'])
        if g is None or g.shape[-3] == oc:
            return super().get_loss()
        assert g.shape[-3] == 2 * oc == 32, g.shape
        w = self.loss_weights.get('kp2d', 1.0)
        # parent computes every other head exactly as before; its kp2d term is evaluated on the first half with weight 0
        gt['kp2d'] = g[:, :, :oc]
        self.loss_weights['kp2d'] = 0.0
        try:
            loss = super().get_loss()
        finally:
            gt['kp2d'] = g
            self.loss_weights['kp2d'] = w
        l1_first = self.loss_terms.get('kp2d', float('nan'))
        term, n_cells, n_pair = quotient_kp_term(self.forward_loss_dict['pred_center_dict']['kp2d'], g)
        self.loss_terms['kp2d'] = float(term.detach())
        self.loss_terms['kp2d_l1_first'] = float(l1_first)
        self.loss_terms['kp2d_pair_cells'] = float(n_pair)
        return loss + w * term


UDH.__all__['CenterHeadKZST'] = CenterHeadKZST


# ======================================================================================= dataset
class KZST2_Crop_Dataset(DatasetTemplate):
    _augment = MMCache_Det_Dataset._augment               # the repo's online augmentation, verbatim

    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)
        self.data_pre_processor = DataPreProcessor(self.dataset_cfg, training=self.training)
        kc = dataset_cfg.KZST2
        self.student = int(kc.STUDENT)
        self.class_names = list(dataset_cfg.CLASS_NAMES)
        self.agnostic_name = list(dataset_cfg.HM_CLASS_NAMES)[0]
        self.modalities = ['rgb']
        self.teacher_rgb_path = None
        self.euler_seq = str(dataset_cfg.get('EULER_SEQ', 'xyz'))
        self.flip_body_mirror = str(dataset_cfg.get('FLIP_BODY_MIRROR_AXIS', 'y'))
        self.new_im_width, self.new_im_hight = int(dataset_cfg.IM_RESIZE[0]), int(dataset_cfg.IM_RESIZE[1])
        assert (self.new_im_width, self.new_im_hight) == S.IN_WH
        self.W, self.H = self.new_im_width, self.new_im_hight
        self.stride = int(dataset_cfg.STRIDE)
        self.im_num = int(dataset_cfg.IM_NUM)
        self.aug = dict(dataset_cfg.get('AUG', {}) or {})
        self.norm_mean = np.array(dataset_cfg.NORM_MEAN, np.float32)
        self.norm_std = np.array(dataset_cfg.NORM_STD, np.float32)
        Z = dict(np.load(str(kc.PSEUDO), allow_pickle=False))
        self.P = {k: Z[k] for k in ('seq', 'frame', 'cls', 'K_crop', 't', 'lwh', 'rho', 'R_T', 'quotient', 'valid', 'target16')}
        self.crops = np.load(str(kc.CROPS), mmap_mode='r')
        assert len(self.crops) == len(self.P['seq'])
        idx = np.where(self.P['valid'][:, self.student])[0]
        mx = int(kc.get('MAX_FRAMES', 0) or 0)
        if mx:
            idx = idx[np.linspace(0, len(idx) - 1, min(mx, len(idx))).astype(int)]
        self.frame_idx = idx
        rep = int(dataset_cfg.get('TRAIN_REPEAT', 1)) if self.training else 1
        self.valid_idx = np.tile(idx, rep) if rep > 1 else idx
        self.r16 = {c: np.tile(np.asarray(self.P['rho'][self.P['cls'] == c][0], np.float32), 8)
                    for c in sorted(set(self.P['cls'].tolist()))}
        if self.logger is not None:
            self.logger.info('KZST2[student %d]: %d frames (quotient %d), crops %s' % (
                self.student, len(idx), int(self.P['quotient'][idx, self.student].sum()), kc.CROPS))

    def __len__(self):
        return len(self.valid_idx)

    def pseudo_box(self, i):
        R = self.P['R_T'][i, self.student]
        eul = Rot.from_matrix(R).as_euler(self.euler_seq)
        return np.concatenate([self.P['t'][i], self.P['lwh'][i], eul]).reshape(1, 9).astype(np.float64)

    def __getitem__(self, item):
        i = int(self.valid_idx[item])
        image = np.ascontiguousarray(self.crops[i]).astype(np.float32).transpose(2, 0, 1) / 255.0
        boxes = self.pseudo_box(i)
        names = [self.agnostic_name]
        K = np.asarray(self.P['K_crop'][i], np.float64).copy()
        if self.training and self.aug:
            image, boxes, names, K = self._augment(image, boxes, names, K)
        image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        data_dict = {
            'image': image[None],
            'gt_box9d': boxes.reshape(-1, 9).astype(np.float32),
            'gt_name': np.array(names),
            'intrinsic': np.array([K], dtype=np.float64),
            'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
            'distortion': np.zeros((1, 5), dtype=np.float32),
            'raw_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.get('OB_SIZE', [[1.0, 1.0, 1.0]])),
            'stride': self.stride,
            'scene_id': str(self.P['seq'][i]).split('/')[0],
            'seq_id': str(self.P['seq'][i]),
            'frame_id': str(self.P['frame'][i]),
        }
        data_dict = self.data_pre_processor(data_dict)
        kp = data_dict['kp2d']                                       # (1, 16, h, w), offset / 64 at the centre cell
        data_dict['kp2d'] = S.make_pair(kp, self.r16[str(self.P['cls'][i])], bool(self.P['quotient'][i, self.student]))
        return data_dict


UDS.__all__['KZST2_Crop_Dataset'] = KZST2_Crop_Dataset


# ======================================================================================= config
def build_cfg(student, pseudo, crops, max_frames=0):
    """KZ sim training config (sim_full_r34_kp_w12_zoom.yaml) + the frozen stage-2 overrides. Returns EasyDict."""
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(S.TOOLS)
    try:
        cfg = EasyDict()
        cfg_from_yaml_file(S.SIM_KP_CFG, cfg)
    finally:
        os.chdir(cwd)
    dc = cfg.DATA_CONFIG
    dc.DATASET = 'KZST2_Crop_Dataset'
    dc.DATA_PATH = S.DATA
    dc.NORM_MEAN = list(S.TRAIN['norm_mean'])
    dc.NORM_STD = list(S.TRAIN['norm_std'])
    dc.AUG = EasyDict(dict(S.AUG))                                   # replaced as a whole: nothing inherited
    dc.TRAIN_REPEAT = S.TRAIN['train_repeat']
    dc.VIEW_AUG = None                                               # not used: crops are the deployed windows
    dc.MIN_RGB_VIS = 0.0
    dc.KZST2 = EasyDict({'STUDENT': int(student), 'PSEUDO': str(pseudo), 'CROPS': str(crops), 'MAX_FRAMES': int(max_frames)})
    m = cfg.MODEL
    m.DENSE_HEAD_2D.NAME = 'CenterHeadKZST'
    m.DENSE_HEAD_2D.LOSS_WEIGHTS = EasyDict(dict(S.LOSS_WEIGHTS))
    o = cfg.OPTIMIZATION
    o.OPTIMIZER = S.TRAIN['optimizer']
    o.LR = S.TRAIN['lr']
    o.NUM_EPOCHS = S.TRAIN['epochs']
    o.BATCH_SIZE_PER_GPU = S.TRAIN['batch']
    return cfg


def assert_cfg(cfg):
    """Pre-registered config asserts (merged AUG == prereg AUG key by key, FREEZE_BN false, recipe constants)."""
    aug = dict(cfg.DATA_CONFIG.AUG)
    assert set(aug) == set(S.AUG) and all(aug[k] == S.AUG[k] for k in S.AUG), ('merged AUG != prereg AUG', aug)
    assert float(aug['target_ring']) == 0.0 and float(aug['target_blur']) == 0.0 and float(aug['blur']) == 0.0
    assert cfg.MODEL.BACKBONE_2D.FREEZE_BN is False
    assert float(cfg.MODEL.BACKBONE_2D.get('MIXSTYLE', {}).get('p', 0.0)) == 0.0
    lw = dict(cfg.MODEL.DENSE_HEAD_2D.LOSS_WEIGHTS)
    assert lw == S.LOSS_WEIGHTS, lw
    o = cfg.OPTIMIZATION
    assert (o.OPTIMIZER, float(o.LR), int(o.NUM_EPOCHS), int(o.BATCH_SIZE_PER_GPU)) == ('adam_onecycle', 2e-4, 3, 8)
    assert (float(o.WEIGHT_DECAY), list(o.MOMS), float(o.PCT_START), float(o.DIV_FACTOR), float(o.GRAD_NORM_CLIP)) == \
        (S.TRAIN['weight_decay'], S.TRAIN['moms'], S.TRAIN['pct_start'], S.TRAIN['div_factor'], S.TRAIN['grad_norm_clip'])
    assert list(cfg.DATA_CONFIG.NORM_MEAN) == S.TRAIN['norm_mean'] and list(cfg.DATA_CONFIG.NORM_STD) == S.TRAIN['norm_std']
    assert int(cfg.DATA_CONFIG.get('KP_NEIGHBOR', 0)) == 0 and bool(cfg.DATA_CONFIG.MAKE_KP2D)
    return True


def count_bn(model):
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
