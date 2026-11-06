# -*- coding: utf-8 -*-
"""放大裁窗的旋转来源钩子（2026-09-22 夜，KZ 臂）。

位置网络 A 照常在整幅 512x288 输入上出检出 / 中心 / 深度 / 尺寸。对每帧，取 A 的峰值格和它预测的 2D 宽 w2d，
按 z = target_w2d / w2d 从【去畸变的原分辨率图】裁一个以峰值为中心、宽 = 原宽 / z、16:9 的窗口，缩放成 512x288
喂给关键点网络 B（B 训练时目标就在这个表观尺寸附近，见 camnorm_sim_full_kp_t_zoom.yaml）。
B 的 kp2d 在窗口中心附近的峰值格读出，偏移除以 z 换算回整幅输入像素，写进 A 的 pred_center_dict['kp2d'] 的峰值格；
A 的解码器随后用 PnP 只换旋转（KP2D_POSE_PARTS = rot），位置逐帧不变。

必须装在所有会改热图的钩子（集成、翻转、时序背景）**之后**，这样读到的峰值格就是解码器用的那个。
"""
import os

import cv2
import numpy as np
import torch

from uavdet3d.utils import camera_geometry as cg

KP_SCALE = 64.0


class RawSource:
    """真实数据：MAV6D 原图（1920x1080 JPEG）按标定去畸变到 K_raw；按 (seq, frame) 取。缓存 meta 里有 K_raw / raw_wh。"""

    def __init__(self, ds, root):
        from build_mav6d_cache import D_CALIB, K_CALIB
        self.root = root
        self.K_calib, self.D_calib = K_CALIB, D_CALIB
        self.meta = {(m['seq'], m['frame']): m for m in ds.metas if m}
        self.maps = {}

    def get(self, seq, frame):
        m = self.meta[(str(seq), str(frame))]
        if 'raw_path' in m:
            # 无畸变数据集（AntiUAV3D，build_antiuav3d_cache / _zoom）：meta 自带原图路径；变焦缓存再按 win 裁出窗口，
            # K_raw / raw_wh 已是窗口原生分辨率上的值。MAV6D meta 没有 raw_path，走下面原逻辑，行为不变。
            img = cv2.imread(m['raw_path'], cv2.IMREAD_COLOR)
            assert img is not None, m['raw_path']
            if 'win' in m:
                x0, y0, w, h = (int(v) for v in m['win'])
                img = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
            assert (img.shape[1], img.shape[0]) == tuple(int(v) for v in m['raw_wh']), (img.shape, m['raw_wh'])
            return img, np.asarray(m['K_raw'], np.float64)
        cls, scene, sq = m['seq'].split('/')
        p = os.path.join(self.root, cls, 'JPEGImages', scene, sq, m['frame'])
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        assert img is not None, p
        K_raw = np.asarray(m['K_raw'], np.float64)
        key = tuple(np.round(K_raw.reshape(-1), 6))
        if key not in self.maps:
            wh = tuple(int(v) for v in m['raw_wh'])
            self.maps[key] = cv2.initUndistortRectifyMap(self.K_calib, self.D_calib, None, K_raw, wh, cv2.CV_32FC1)
        m1, m2 = self.maps[key]
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), K_raw


def crop_window(u_in, v_in, z, raw_w, raw_h, in_w, in_h):
    """整幅输入像素 (u,v) 为中心、缩放 z 的 16:9 窗口（原图像素），夹到图内。返回 x0, y0, w, h（浮点）。"""
    w = raw_w / z
    h = w * in_h / in_w
    cu, cv_ = u_in * raw_w / in_w, v_in * raw_h / in_h
    x0 = float(np.clip(cu - w / 2, 0, max(0.0, raw_w - w)))
    y0 = float(np.clip(cv_ - h / 2, 0, max(0.0, raw_h - h)))
    return x0, y0, w, h


def install_zoom_kp2d_hook(model, kp_models, source, target_w2d=95.0, z_range=(1.0, 4.0), search=2,
                           norm_mean=None, norm_std=None, log=None, oracle_gt=False):
    """见模块说明。source: RawSource；kp_models: 关键点网络列表（多个时 kp2d 图逐格平均）。
    oracle_gt=True（归因用，不是部署方式）：裁窗改按【真值】中心与真值 2D 宽来定，B 的 kp2d 仍写回位置网络的峰值格 ——
    用来区分「裁窗质量（位置网络的中心/尺寸误差）」和「关键点网络本身的迁移」各占多少。"""
    from scipy.spatial.transform import Rotation as _R
    _P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    head = model.dense_head_2d
    orig = head.forward
    pp = model.model_cfg.POST_PROCESSING
    pp.POSE_FROM_KP2D = True
    pp.KP2D_POSE_PARTS = 'rot'
    stride = int(model.dataset.dataset_cfg.STRIDE)
    in_w, in_h = int(model.dataset.new_im_width), int(model.dataset.new_im_hight)
    nm = np.asarray(norm_mean, np.float32).reshape(3, 1, 1) if norm_mean is not None else None
    ns = np.asarray(norm_std, np.float32).reshape(3, 1, 1) if norm_std is not None else None
    stats = {'n': 0, 'z': [], 'rows': []}     # rows: (seq, frame, z, w2d_pred) 便于按机型看裁窗倍数

    def run_b(img_in):
        x = img_in.astype(np.float32).transpose(2, 0, 1) / 255.0
        if nm is not None:
            x = (x - nm) / ns
        t = torch.from_numpy(x[None, None]).cuda()
        maps, hms = [], []
        with torch.no_grad():
            for km in kp_models:
                tb = {'image': t, 'batch_size': 1}
                for mod in km.module_list:
                    tb = mod(tb)
                maps.append(tb['pred_center_dict']['kp2d'].float()[0])
                hms.append(torch.sigmoid(tb['pred_center_dict']['hm'].float()[0, 0]))
        return torch.stack(maps).mean(0), torch.stack(hms).mean(0)

    def fwd(batch_dict):
        out = orig(batch_dict)
        pc = batch_dict['pred_center_dict']
        hm = torch.sigmoid(pc['hm'].float())                      # (B*K, 1, h, w)
        B = hm.shape[0]
        kp_out = torch.zeros((B, 16, hm.shape[-2], hm.shape[-1]), dtype=torch.float32, device=hm.device)
        seqs = [str(s) for s in batch_dict['seq_id']]
        frames = [str(f) for f in batch_dict['frame_id']]
        for b in range(B):
            flat = hm[b, 0].reshape(-1)
            idx = int(torch.argmax(flat))
            r, c = idx // hm.shape[-1], idx % hm.shape[-1]
            u = (c + float(pc['center_res'][b, 0, r, c])) * stride
            v = (r + float(pc['center_res'][b, 1, r, c])) * stride
            w2d = float(torch.exp(pc['size2d'][b, 0, r, c].float().clamp(-5, 9)))
            if oracle_gt:
                g = batch_dict['gt_box9d'][b]
                g = np.asarray(g.detach().cpu().numpy() if hasattr(g, 'detach') else g, np.float64).reshape(-1, 9)
                g = g[np.abs(g).sum(1) > 0]
                if len(g):
                    g = g[0]
                    K_in = np.asarray(batch_dict['intrinsic'][b].cpu() if hasattr(batch_dict['intrinsic'][b], 'cpu') else batch_dict['intrinsic'][b], np.float64).reshape(-1, 3, 3)[0]
                    q = K_in @ g[:3]
                    u, v = float(q[0] / q[2]), float(q[1] / q[2])
                    C = (_P8 * g[3:6]) @ _R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
                    uv = C @ K_in.T
                    uv = uv[:, :2] / uv[:, 2:]
                    w2d = float(np.ptp(uv[:, 0]))
            z = float(np.clip(target_w2d / max(w2d, 1e-3), z_range[0], z_range[1]))
            img, K_raw = source.get(seqs[b], frames[b])
            raw_h, raw_w = img.shape[:2]
            x0, y0, w, h = crop_window(u, v, z, raw_w, raw_h, in_w, in_h)
            s = in_w / w
            M = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
            crop = cv2.warpAffine(img, M, (in_w, in_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            kp_map, hm_b = run_b(crop)
            # 目标在窗口里的位置 = 峰值像素相对窗口原点（窗口被夹到图内时不一定在正中）
            uc = (u * raw_w / in_w - x0) * s
            vc = (v * raw_h / in_h - y0) * s
            cc, rc = int(uc / stride), int(vc / stride)
            r0, r1 = max(0, rc - search), min(hm_b.shape[0], rc + search + 1)
            c0, c1 = max(0, cc - search), min(hm_b.shape[1], cc + search + 1)
            sub = hm_b[r0:r1, c0:c1]
            j = int(torch.argmax(sub.reshape(-1)))
            rb, cb = r0 + j // sub.shape[1], c0 + j % sub.shape[1]
            kp_out[b, :, r, c] = kp_map[:, rb, cb] / z                 # 窗口输入像素 -> 整幅输入像素
            stats['n'] += 1
            stats['z'].append(z)
            stats['rows'].append((seqs[b], frames[b], z, w2d))
        pc['kp2d'] = kp_out.to(pc['hm'].dtype)
        return out
    head.forward = fwd
    return stats
