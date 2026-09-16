# -*- coding: utf-8 -*-
"""把仿真采集参数对齐到真实数据：先量 MAV6D 的采集特征，再反推录制器该用的 FOV / 分辨率 / 相机距离。

核心不变量是【目标在原始图像里占多少像素】 native_px = L * f_native / Z，
因为 camnorm 之后网络看到的细节量由它决定（裁窗只能搬运像素，不能创造像素）。

    python tools/align_sim_to_real.py
"""
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
MAV_F_NATIVE = 1979.4 * 0.932      # 去畸变后的针孔焦距（1920 宽）
MAV_W, MAV_H = 1920, 1080
SIM_W, SIM_H, SIM_FOV = 1280, 720, 90.0


def span_px(b, f):
    """目标 8 角点在【原始分辨率】上的投影长边（正交近似足够，目标很小）。"""
    c = (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T
    return max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) * f / b[2]


def view_angles(b):
    """相机方向在机体系里的方位角 / 俯仰角（机头 +x，左 +y，上 +z）。"""
    M = R.from_euler('xyz', b[6:9]).as_matrix()
    v = M.T @ (-b[:3])
    v = v / np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(np.clip(v[2], -1, 1)))


def gather(root, split, f_native, scale_K_to_native):
    idx = pickle.load(open(os.path.join(root, split, 'index.pkl'), 'rb'))
    Z, S, AZ, EL, N, UV = [], [], [], [], [], []
    for i in idx['valid_idx']:
        m = idx['metas'][int(i)]
        K = np.asarray(m['K_in']).reshape(3, 3)
        f_cache = float(np.sqrt(K[0, 0] * K[1, 1]))
        f = f_native if scale_K_to_native else f_cache
        bs = np.asarray(m['boxes9d']).reshape(-1, 9)
        bs = bs[(np.abs(bs).sum(1) > 0) & (bs[:, 2] > 0)]
        N.append(len(bs))
        for b in bs:
            Z.append(b[2]); S.append(span_px(b, f))
            az, el = view_angles(b); AZ.append(az); EL.append(el)
            uv = K @ b[:3]; UV.append((uv[0] / uv[2] / idx['W'], uv[1] / uv[2] / idx['H']))
    return (np.array(Z), np.array(S), np.array(AZ), np.array(EL), np.array(N), np.array(UV))


def show(tag, d, fov, W):
    Z, S, AZ, EL, N, UV = d
    q = lambda a, p: np.percentile(a, p)
    print('\n%s' % tag)
    print('  角分辨率 %.1f px/度（%d px / %.1f°）' % (W / fov, W, fov))
    print('  目标深度 Z       中位 %5.2f m   p10 %5.2f  p90 %5.2f' % (np.median(Z), q(Z, 10), q(Z, 90)))
    print('  原始图上像素跨度  中位 %5.0f px  p10 %5.0f  p90 %5.0f   <<< 决定细节量' % (np.median(S), q(S, 10), q(S, 90)))
    print('  角大小 L/Z       中位 %5.2f°    p10 %5.2f  p90 %5.2f' % (
        np.median(np.degrees(S / (W / (2 * np.tan(np.radians(fov / 2))))) ),
        np.degrees(q(S, 10) / (W / (2 * np.tan(np.radians(fov / 2))))),
        np.degrees(q(S, 90) / (W / (2 * np.tan(np.radians(fov / 2)))))))
    print('  每帧目标数        中位 %.1f（1 架 %.0f%% / 2 架 %.0f%% / >=3 架 %.0f%%）' % (
        np.median(N), 100 * (N == 1).mean(), 100 * (N == 2).mean(), 100 * (N >= 3).mean()))
    print('  视线方位角 |az|   中位 %5.1f°   正面(|az|<45°) %.0f%%  侧面 %.0f%%  背面(|az|>135°) %.0f%%' % (
        np.median(np.abs(AZ)), 100 * (np.abs(AZ) < 45).mean(),
        100 * ((np.abs(AZ) >= 45) & (np.abs(AZ) <= 135)).mean(), 100 * (np.abs(AZ) > 135).mean()))
    print('  视线俯仰角 el     中位 %+5.1f°   p10 %+5.1f  p90 %+5.1f （>0 = 相机在机体上方看下来）' % (
        np.median(EL), q(EL, 10), q(EL, 90)))
    print('  目标在画面里 u/W  中位 %.2f [%.2f,%.2f]   v/H 中位 %.2f [%.2f,%.2f]' % (
        np.median(UV[:, 0]), q(UV[:, 0], 10), q(UV[:, 0], 90),
        np.median(UV[:, 1]), q(UV[:, 1], 10), q(UV[:, 1], 90)))
    return dict(Z=np.median(Z), S=np.median(S))


def main():
    real = gather('E:/mmcache/mav6d_cn', 'test', MAV_F_NATIVE, True)
    sim = gather('E:/mmcache/pp_realsize', 'train', None, False)
    r = show('真实 MAV6D（目标）', real, 2 * np.degrees(np.arctan(MAV_W / 2 / MAV_F_NATIVE)), MAV_W)
    s = show('仿真 PowerPlant（现状）', sim, SIM_FOV, SIM_W)

    print('\n' + '=' * 96)
    print('对齐：要让仿真目标的原始像素跨度追上真实，需要 f_native / Z 提高 %.2f 倍' % (
        (r['S'] / r['Z'] * 0 + 1) * (r['S'] / s['S'])))
    need = r['S'] / s['S']
    f_sim_now = SIM_W / (2 * np.tan(np.radians(SIM_FOV / 2)))
    print('  现状 f_native=%.0f px、Z 中位 %.2f m -> 跨度 %.0f px' % (f_sim_now, s['Z'], s['S']))
    print('  目标                                 -> 跨度 %.0f px' % r['S'])
    print('\n  可选组合（需要总增益 %.2f 倍）：' % need)
    for W, fov, Zt in ((1920, 55.0, 3.4), (1920, 60.0, 3.4), (1920, 70.0, 3.0), (2560, 90.0, 3.4), (1920, 90.0, 2.6)):
        f = W / (2 * np.tan(np.radians(fov / 2)))
        gain = (f / f_sim_now) * (s['Z'] / Zt)
        width_m = 2 * Zt * np.tan(np.radians(fov / 2))
        print('    %4dx%-4d FOV %4.1f°  相机距 %.1f m -> f=%6.0f px，增益 %.2f 倍，跨度 %3.0f px，'
              '该距离画面宽 %.1f m，像素数 x%.1f' % (
                  W, int(W * 9 / 16), fov, Zt, f, gain, s['S'] * gain, width_m, (W * W * 9 / 16) / (SIM_W * SIM_H)))


if __name__ == '__main__':
    main()
