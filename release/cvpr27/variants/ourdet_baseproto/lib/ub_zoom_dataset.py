# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto, 11-17] upper-bound row with view augmentation: MMCache_Det_Dataset on the full-resolution JPEG store of
the RealDrone3D train frames (lib/build_ub_jpeg.py), i.e. the SAME VIEW_AUG code path as the sim cache, plus ONE extension:

  VIEW_AUG.zoom_out_max (Z >= 1, default 1 = repository behaviour bit-for-bit): the window chosen by t_range may be up to Z x the frame
  (wider than the frame). Why: VIEW_AUG only crops (zooms IN). The sim cache has far drones natively, so cropping covers the t range; the
  real train frames are near (native t = log(Zv/diag) median 2.47, p95 3.55), so crop-only can never make them look farther/smaller.
  Window > frame = a virtual camera with a shorter focal length: the frame is resized by s = W_in / w and placed at offset (-x0, -y0)*s
  on a canvas filled with the per-domain mean colour (NORM_MEAN*255 -> exactly 0 after input normalisation). K is updated with the
  same exact formula as the repository (_view: K2 = scale(translate(K, -x0, -y0), W_in/w, H_in/h), cv2 pixel-centre convention);
  the warp is cv2.resize(INTER_AREA) + an exact affine residual, so pixels and K agree exactly.
  Z = 8 = e^(4.6 - 2.47) rounded: lets the median TRAIN frame reach the recipe's t upper bound 4.6 (chosen from train labels only).

Everything else (random draws, t uniform in t_range, max_upscale, window placement keeping the chosen target >= 8 input px inside,
dropping boxes whose centre leaves the view, AUG, NORM) is the repository code. The window position rule is generalised so that for
w <= frame it is identical: x0 in [max(min(0, sW-w), cu-w+m), min(max(0, sW-w), cu-m)]."""
import numpy as np
import cv2
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset
from uavdet3d.utils import camera_geometry as cg

NAME = 'UBZoom_Det_Dataset'


class UBZoomDataset(MMCache_Det_Dataset):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)
        va = dataset_cfg.get('VIEW_AUG', {}) or {}
        self.zoom_out_max = float(va.get('zoom_out_max', 1.0))
        nm = dataset_cfg.get('NORM_MEAN', None)
        self.pad_rgb = tuple(float(v) * 255.0 for v in nm) if nm else (0.0, 0.0, 0.0)
        assert self.store == 'jpeg' and len(self.modalities) == 1, 'UBZoom: rgb-only jpeg store'
        if logger is not None:
            logger.info('[ubzoom] VIEW_AUG %s, zoom_out_max %.1f, pad colour %s' % (dict(va), self.zoom_out_max, self.pad_rgb))

    def _view(self, img, K, boxes, names, item, extra=None):
        assert extra is None
        sH, sW = img.shape[:2]
        oW, oH = self.new_im_width, self.new_im_hight
        unit = int(np.gcd(oW, oH))
        uw, uh = oW // unit, oH // unit
        uv = None
        cand = []
        if len(boxes):
            pr = (K @ boxes[:, :3].T).T
            uv = pr[:, :2] / pr[:, 2:3]
            cand = [j for j in range(len(boxes)) if boxes[j, 2] > 0 and 0 <= uv[j, 0] < sW and 0 <= uv[j, 1] < sH]
        j = None
        if cand:
            j = cand[np.random.randint(len(cand))] if self.training else cand[0]
        k_full = min(sW // uw, sH // uh)
        if self.training and self.t_range is not None and j is not None:
            f_src = float(np.sqrt(K[0, 0] * K[1, 1]))
            L = float(np.linalg.norm(boxes[j, 3:6]))
            t = float(np.random.uniform(*self.t_range))
            w_want = L * np.exp(t) * f_src * oW / (max(float(boxes[j, 2]), 1e-6) * self.f_ref)
            k_lo = max(1, int(round(unit / max(self.max_upscale, 1.0))))
            k_hi = int(np.floor(k_full * self.zoom_out_max))           # zoom_out_max = 1 -> k_full (repository)
            k = max(k_lo, min(int(round(w_want / uw)), k_hi))
        else:
            if self.training:
                z0, z1 = self.view_zoom
                z = float(np.exp(np.random.uniform(np.log(z0), np.log(z1)))) if z1 > z0 else z0
            else:
                z = self.val_zooms[item % len(self.val_zooms)]
            k_min = unit if self.no_upscale else 1
            k = max(k_min, min(int(round(sW / z / uw)), k_full))
        w, h = k * uw, k * uh
        m = 8.0 * w / oW
        bx0, bx1 = min(0.0, float(sW - w)), max(0.0, float(sW - w))   # admissible window origins (frame-inside, or frame-covering)
        by0, by1 = min(0.0, float(sH - h)), max(0.0, float(sH - h))
        if j is not None:
            cu, cv_ = uv[j]
            lo_x, hi_x = max(bx0, cu - w + m), min(bx1, cu - m)
            lo_y, hi_y = max(by0, cv_ - h + m), min(by1, cv_ - m)
            x0 = np.random.uniform(lo_x, hi_x) if (self.training and lo_x <= hi_x) else np.clip(cu - w / 2.0, bx0, bx1)
            y0 = np.random.uniform(lo_y, hi_y) if (self.training and lo_y <= hi_y) else np.clip(cv_ - h / 2.0, by0, by1)
        else:
            x0 = np.random.uniform(bx0, bx1) if self.training else (sW - w) / 2.0
            y0 = np.random.uniform(by0, by1) if self.training else (sH - h) / 2.0
        x0, y0 = int(round(x0)), int(round(y0))
        if w <= sW and h <= sH:                                       # repository branch, unchanged
            win = img[y0:y0 + h, x0:x0 + w]
            if (w, h) != (oW, oH):
                win = cv2.resize(win, (oW, oH), interpolation=cv2.INTER_AREA if w > oW else cv2.INTER_LINEAR)
        else:                                                         # zoom-out: frame smaller than the view, mean-colour canvas
            s = oW / float(w)
            nw, nh = max(1, int(round(sW * s))), max(1, int(round(sH * s)))
            sm = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = nw / float(sW), nh / float(sH)
            ax, ay = s / sx, (oH / float(h)) / sy
            # exact map: u_out = s*(u - x0 + 0.5) - 0.5 with u = (u_r + 0.5)/sx - 0.5
            M = np.array([[ax, 0.0, 0.5 * ax - s * x0 - 0.5], [0.0, ay, 0.5 * ay - (oH / float(h)) * y0 - 0.5]], np.float64)
            win = cv2.warpAffine(sm, M, (oW, oH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.pad_rgb)
        K2 = cg.scale_K(cg.translate_K(K, -x0, -y0), oW / float(w), oH / float(h))
        if len(boxes):
            pr = (K2 @ boxes[:, :3].T).T
            uv2 = pr[:, :2] / pr[:, 2:3]
            keep = (boxes[:, 2] > 0) & (uv2[:, 0] >= 0) & (uv2[:, 0] < oW) & (uv2[:, 1] >= 0) & (uv2[:, 1] < oH)
            boxes, names = boxes[keep], [n for n, kk in zip(names, keep) if kk]
        return np.ascontiguousarray(win), K2, boxes, names


def build(dataset_cfg, root_path=None, training=True, logger=None):
    # the instance carries DATASET = 'MMCache_Det_Dataset' (DatasetTemplate / model code dispatch on that name: pre-processor etc.)
    import copy
    dataset_cfg = copy.deepcopy(dataset_cfg); dataset_cfg.DATASET = 'MMCache_Det_Dataset'
    return UBZoomDataset(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)
