# -*- coding: utf-8 -*-
"""[review1113 lidraw] arm (b) of the supplement control "LiDAR body map vs raw LiDAR points" (11-14).
CenterDetLidPts = CenterDetLiD (geo1004/lidar/kzlid/center_det_lid.py, the KZ-LiD orientation teacher) with ONLY the auxiliary head and its target
replaced; network input (RGB, first 3 channels), backbone, keypoint / corner / detection losses, data, schedule are untouched.

  body map (arm a, CenterDetLiD.lid_head):   per cell of the 64x36 crop grid: occupancy of drone-tagged LiDAR returns (BCE) + mean depth of the
                                             returns in the cell relative to the body centre / max edge (L1 on occupied cells).
  raw points (arm b, this file, pts_head):   per object a FIXED SET of N = 64 points, each (u, v, dz):
                                               u, v = position in the crop grid, in units of 8 grid cells (= 64 input px, the unit of the kp2d
                                                      corner offsets), predicted as soft-argmax anchor of a learned saliency map + per-point offset
                                               dz   = depth relative to the body centre / max edge (same normalisation and clamp +-1.5 as the body map)
                                             supervised by the symmetric Chamfer distance (Euclidean, not squared - L1-like, as the body-map depth term)
                                             to the drone-tagged LiDAR returns in the crop at INPUT-PIXEL resolution (no 8x8 pooling, no occupancy
                                             raster): every kept tagged pixel is a target point (u_px/64, v_px/64, (d - z_c)/L), randomly subsampled
                                             to at most MAX_TARGET = 512 per crop.
Tagging / filtering = CenterDetLiD.lid_targets exactly (tag > 0.5 and depth > 0.05 m; the GT drone with most returns inside the depth band
|d - z_box| < 0.75 L + 0.3 m; >= 5 returns; z_c = median depth of the kept returns; L = that drone's max edge). Note: the multimodal training cache
stores the LiDAR returns 5x5-dilated at raw resolution (tools/build_mm_sidecar_obj.py --dilate 5); undilated returns are not in the training
batch, so the "raw" point set here = the per-pixel (un-pooled, un-rasterised) samples of the same returns the body map pools.
Loss weight = LIDAR_AUX.WEIGHT (1.0, the body-map weight). Crops without a valid target contribute 0 (as the body map's ok-mask).
At deployment the head is unused; checkpoints carry extra pts_head.* tensors (stripped by strip_aux.py before reading)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdet3d.model.detectors.center_det_kd import CenterDetKD


class CenterDetLidPts(CenterDetKD):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        L = model_cfg.get('LIDAR_AUX', None) or {}
        self.w_lid = float(L.get('WEIGHT', 1.0))
        self.ch_d = int(L.get('DEPTH_CH', 4))
        self.ch_t = int(L.get('TAG_CH', 5))
        self.n_pts = int(L.get('N_POINTS', 64))
        self.max_tgt = int(L.get('MAX_TARGET', 512))
        self.unit = float(L.get('UV_UNIT_PX', 64.0))
        self.depth_max = float(dataset.dataset_cfg.get('DEPTH_MAX', 24.0))
        C = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        # same first conv as the body-map head (C -> 64, 3x3); then saliency logits (soft-argmax anchor) and an MLP for the point set
        self.pts_head = nn.ModuleDict(dict(
            conv=nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True)),
            sal=nn.Conv2d(64, 1, 1),
            mlp=nn.Sequential(nn.Linear(64, 256), nn.ReLU(inplace=True), nn.Linear(256, self.n_pts * 3))))

    @torch.no_grad()
    def pts_targets(self, batch_dict):
        """-> list (len B*K) of (M_n, 3) target point sets (u, v in UV_UNIT_PX units of input pixels, dz) or None.
        Selection / filtering copied from CenterDetLiD.lid_targets (body-map arm), only the pooling step is replaced."""
        img = batch_dict['image']
        B, K, _, H, W = img.shape
        d = img[:, :, self.ch_d].reshape(B * K, 1, H, W).float() * self.depth_max
        hit = ((img[:, :, self.ch_t] > 0.5).reshape(B * K, 1, H, W) & (d > 0.05)).float()
        gt = batch_dict.get('gt_box9d', None)
        out = [None] * (B * K)
        for n in range(B * K):
            if gt is None:
                break
            g = gt[n // K]
            g = g[g.abs().sum(-1) > 0]
            hn = hit[n, 0] > 0
            if len(g) == 0 or int(hn.sum()) < 5:
                continue
            dn = d[n, 0]
            v = dn[hn]
            best = (0, -1, None)
            for k in range(len(g)):
                L = g[k, 3:6].max().clamp(min=0.05)
                c = int(((v - g[k, 2]).abs() < 0.75 * L + 0.3).sum())
                if c > best[0]:
                    best = (c, k, L)
            if best[0] < 5:
                continue
            k, L = best[1], best[2]
            sel = hn & ((dn - g[k, 2]).abs() < 0.75 * L + 0.3)
            zc = dn[sel].median()
            yy, xx = torch.nonzero(sel, as_tuple=True)
            dz = ((dn[yy, xx] - zc) / L).clamp(-1.5, 1.5)
            P = torch.stack([(xx.float() + 0.5) / self.unit, (yy.float() + 0.5) / self.unit, dz], 1)
            if len(P) > self.max_tgt:
                P = P[torch.randperm(len(P), device=P.device)[:self.max_tgt]]
            out[n] = P
        return out

    def pts_pred(self, f):
        """f (N, C, h, w) features_2d -> (N, n_pts, 3) points (u, v in UV_UNIT_PX units of input px, dz)."""
        N, C, h, w = f.shape
        z = self.pts_head['conv'](f.float())
        a = torch.softmax(self.pts_head['sal'](z).reshape(N, h * w), 1)                       # spatial attention
        sx_px = self._in_W / w; sy_px = self._in_H / h                                          # input px per grid cell (8)
        gy, gx = torch.meshgrid(torch.arange(h, device=f.device, dtype=torch.float32), torch.arange(w, device=f.device, dtype=torch.float32), indexing='ij')
        cu = (a * ((gx.reshape(-1) + 0.5) * sx_px)[None]).sum(1) / self.unit                    # anchor (soft-argmax), UV units
        cv = (a * ((gy.reshape(-1) + 0.5) * sy_px)[None]).sum(1) / self.unit
        g = torch.bmm(a[:, None], z.reshape(N, 64, h * w).transpose(1, 2))[:, 0]                # attention-pooled feature (N, 64)
        o = self.pts_head['mlp'](g).reshape(N, self.n_pts, 3)
        return torch.stack([cu[:, None] + o[..., 0], cv[:, None] + o[..., 1], o[..., 2]], -1)

    @staticmethod
    def chamfer(p, t):
        """symmetric Chamfer, Euclidean (not squared): mean_p min_t |p-t| + mean_t min_p |p-t|"""
        D = torch.cdist(p[None], t[None])[0]
        return D.min(1)[0].mean() + D.min(0)[0].mean()

    def forward(self, batch_dict):
        out = super().forward(batch_dict)
        if not self.training:
            return out
        f = self._student_out['features_2d']                  # (B,K,C,h,w)
        B, K, C, h, w = f.shape
        self._in_H, self._in_W = batch_dict['image'].shape[-2:]
        P = self.pts_pred(f.reshape(B * K, C, h, w))
        T = self.pts_targets(batch_dict)
        ls = [self.chamfer(P[n], t) for n, t in enumerate(T) if t is not None and len(t) >= 5]
        l = torch.stack(ls).mean() if ls else P.sum() * 0.0
        self.dense_head_2d.loss_terms['pts'] = float(l.detach())
        out['loss'] = out['loss'] + self.w_lid * l
        return out
