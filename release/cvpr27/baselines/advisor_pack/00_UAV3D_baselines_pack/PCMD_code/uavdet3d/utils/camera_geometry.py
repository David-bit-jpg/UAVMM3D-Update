# -*- coding: utf-8 -*-
"""相机自适应几何 —— 同一个网络吃任意内参（焦距 / 视场 / 主点 / 分辨率 / 畸变）的数据。

不写死任何一种相机，所有量都只由【每帧自己的内参 K】决定：

1. 虚拟深度（depth_mode='virtual'）
       Zv = Z * f_ref / f，   f = sqrt(fx * fy) 取【网络输入分辨率】上的焦距
   单目深度的主线索是表观大小 ∝ f / Z。同一架机、同一 Zv，在任何焦距的相机上看起来一样大，
   所以网络只需要学「看起来多大 -> Zv」，解码时再按这一帧的 f 乘回米制深度。
   随机缩放增广（图像放大 s 倍、f 乘 s）也因此自动自洽：Zv 跟着除以 s。
   反例（修之前）：仿真水平视场 90°、MAV6D 51.7°，同距离同机身在 MAV6D 输入上大 2 倍，
   网络只能凭大小估深度，于是深度知识根本迁不过去。

2. 视线相对旋转（rot_frame='allo'，allocentric）
       R_allo = R_ray^T * R_cam，   R_ray = 把光轴 +z 转到目标中心视线方向的最小旋转
   目标的外观由它相对【视线】的姿态决定。同一个 R_cam 的目标，放在宽视场画面边缘和画面中心
   看起来是不一样的；卷积网络不知道自己在画面的哪个位置、也不知道这台相机的主点和焦距，
   学不了 R_cam，但能学 R_allo。解码时用预测中心像素的视线把 R_allo 转回 R_cam。

3. 畸变在像素层面去掉：有畸变的数据先 undistort 到针孔相机再进网络（undistort_maps），
   编码 / 解码 / 增广一律按针孔处理，不同镜头模型不会变成额外的外观差异。

像素坐标约定：OpenCV —— 像素中心在整数坐标。cv2.resize / remap 用的是同一约定：
    dst = (src + 0.5) * s - 0.5
所有对 K 的变换（缩放、裁剪、翻转）都按这个约定写成精确式，自检见 self_check()。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# 内参变换
# --------------------------------------------------------------------------- #
def scale_K(K, sx, sy):
    """图像按 (sx, sy) 缩放（cv2.resize 约定）后的内参。"""
    K = np.array(K, dtype=np.float64).reshape(3, 3).copy()
    out = K.copy()
    out[0, 0] = K[0, 0] * sx
    out[0, 1] = K[0, 1] * sx
    out[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    out[1, 1] = K[1, 1] * sy
    out[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return out


def translate_K(K, dx, dy):
    """像素整体平移 (dx, dy)（裁剪时 dx=-x0，贴到画布时 dx=+x0）。"""
    out = np.array(K, dtype=np.float64).reshape(3, 3).copy()
    out[0, 2] += dx
    out[1, 2] += dy
    return out


def hflip_K(K, width):
    """水平翻转 u -> W-1-u 后的内参。等价于相机系 x -> -x；斜切项跟着变号。"""
    out = np.array(K, dtype=np.float64).reshape(3, 3).copy()
    out[0, 2] = (width - 1.0) - out[0, 2]
    out[0, 1] = -out[0, 1]
    return out


def hflip_rotation(R_cam, body_mirror_axis='y'):
    """水平翻转后目标的旋转标签：R' = M R S。

    翻转图像 = 相机系镜像 M = diag(-1,1,1)，镜像后的目标 {M R p + M t}。要把它写成【同一个物体】的真旋转位姿，
    必须借助物体自身的镜像对称 S（S O = O，det S = -1）：M R O = M R S O，于是 R' = M R S（det = +1）。
    S 取物体真正的对称面：无人机（机体 x 前 / y 左 / z 上）左右对称、前后不对称 -> S = diag(1,-1,1)。
    用 S = diag(-1,1,1)（即 M R M）时 R' 与正确值差绕机体 z 180°：机头机尾对调。
    """
    idx = {'x': 0, 'y': 1, 'z': 2}[body_mirror_axis]
    S = np.eye(3)
    S[idx, idx] = -1.0
    M = np.diag([-1.0, 1.0, 1.0])
    Rm = np.asarray(R_cam, dtype=np.float64)
    return np.einsum('ij,...jk,kl->...il', M, Rm, S)


def focal(K):
    """几何平均焦距 sqrt(fx*fy)（像素）。"""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return float(np.sqrt(abs(K[0, 0] * K[1, 1])))


def input_focal(K, raw_wh, new_wh):
    """K 给在 raw 分辨率、网络输入是 new 分辨率时，输入上的焦距。"""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    sx = float(new_wh[0]) / float(raw_wh[0])
    sy = float(new_wh[1]) / float(raw_wh[1])
    return float(np.sqrt(abs(K[0, 0] * sx * K[1, 1] * sy)))


def to_virtual_depth(Z, f, f_ref):
    return np.asarray(Z, dtype=np.float64) * (float(f_ref) / float(f))


def from_virtual_depth(Zv, f, f_ref):
    return np.asarray(Zv, dtype=np.float64) * (float(f) / float(f_ref))


# --------------------------------------------------------------------------- #
# 视线相对旋转
# --------------------------------------------------------------------------- #
def ray_rotation(dirs):
    """(N,3) 视线方向（不必归一化）-> (N,3,3) 把 +z 转到该方向的最小旋转。"""
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-12)
    # 轴 = z x d = (-dy, dx, 0)，|轴| = sin(theta)，cos(theta) = dz
    axis = np.stack([-d[:, 1], d[:, 0], np.zeros(len(d))], axis=1)
    s = np.linalg.norm(axis, axis=1)
    c = np.clip(d[:, 2], -1.0, 1.0)
    theta = np.arctan2(s, c)
    unit = np.where(s[:, None] > 1e-12, axis / np.maximum(s[:, None], 1e-12), np.array([[1.0, 0.0, 0.0]]))
    return R.from_rotvec(unit * theta[:, None]).as_matrix()


def ego_to_allo(R_cam, dirs):
    """相机系旋转 -> 视线相对旋转。R_cam (N,3,3)，dirs (N,3) = 目标中心方向。"""
    Rr = ray_rotation(dirs)
    return np.einsum('nji,njk->nik', Rr, np.asarray(R_cam, dtype=np.float64).reshape(-1, 3, 3))


def allo_to_ego(R_allo, dirs):
    Rr = ray_rotation(dirs)
    return np.einsum('nij,njk->nik', Rr, np.asarray(R_allo, dtype=np.float64).reshape(-1, 3, 3))


# --------------------------------------------------------------------------- #
# 仿真数据的内参
# --------------------------------------------------------------------------- #
def legacy_sim_intrinsic_fix(K, width, height, tol=1e-2):
    """旧 CARLA / UavSim 录制器写进 im_info 的 RGB 内参是故意扰动过的假值：
        fx, fy = fx*1.01, cx = W/2 + 2, cy = H/2 - 1.5
    真渲染是方像素、主点在画面正中（UE 按 NDC 渲染，像素中心约定下主点 = W/2 - 0.5）。
    识别出这个特征就返回 (真内参, True)；不是这种扰动（例如真实相机标定值）就原样返回 (K, False)。
    """
    K = np.array(K, dtype=np.float64).reshape(3, 3)
    fx = K[0, 0]
    legacy = (abs(K[0, 2] - (width / 2.0 + 2.0)) < tol and
              abs(K[1, 2] - (height / 2.0 - 1.5)) < tol and
              abs(K[1, 1] - fx * 1.01) < tol * max(1.0, fx / 100.0) and
              abs(K[0, 1]) < 1e-9)
    if not legacy:
        return K.copy(), False
    Kt = np.array([[fx, 0.0, width / 2.0 - 0.5],
                   [0.0, fx, height / 2.0 - 0.5],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return Kt, True


# --------------------------------------------------------------------------- #
# 去畸变
# --------------------------------------------------------------------------- #
_UNDIST_CACHE = {}


def undistorted_pinhole_K(K, D, size, samples=64):
    """去畸变后的针孔内参：主点不变、去掉斜切，焦距统一乘 s，s 取「原图整条边界都落在输出画面内」的最大值。

    桶形畸变去掉后边缘向外拉伸，焦距不变的话画面边缘会被裁掉（MAV6D 实测约 1.5% 的帧中心出画）；
    缩小焦距保住全部原始像素，代价只是目标略小 —— 虚拟深度按新焦距自动换算，不影响标签。
    枕形畸变（边界向内收）时 s 可能 > 1，封顶 1，不放大。
    """
    import cv2
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    W, H = int(size[0]), int(size[1])
    xs = np.linspace(0, W - 1, samples)
    ys = np.linspace(0, H - 1, samples)
    border = np.concatenate([np.stack([xs, np.zeros_like(xs)], 1), np.stack([xs, np.full_like(xs, H - 1)], 1),
                             np.stack([np.zeros_like(ys), ys], 1), np.stack([np.full_like(ys, W - 1), ys], 1)])
    n = cv2.undistortPoints(border.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    s = 1.0
    for x, y in n:
        if x > 1e-9:
            s = min(s, (W - 1 - cx) / (fx * x))
        elif x < -1e-9:
            s = min(s, cx / (fx * -x))
        if y > 1e-9:
            s = min(s, (H - 1 - cy) / (fy * y))
        elif y < -1e-9:
            s = min(s, cy / (fy * -y))
    return np.array([[fx * s, 0.0, cx], [0.0, fy * s, cy], [0.0, 0.0, 1.0]]), float(s)


def undistort_maps(K, D, size, K_new=None):
    """(map1, map2, K_new)：cv2.remap(img, map1, map2, INTER_LINEAR) 得到针孔图像，内参为 K_new。
    K_new 缺省用 undistorted_pinhole_K（保住全部原始像素）。按 (K, D, size, K_new) 缓存，每个 worker 只算一次。"""
    import cv2
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if K_new is None:
        K_new, _ = undistorted_pinhole_K(K, D, size)
    K_new = np.asarray(K_new, dtype=np.float64).reshape(3, 3)
    key = (tuple(np.round(K.reshape(-1), 6)), tuple(np.round(D, 9)), tuple(size), tuple(np.round(K_new.reshape(-1), 6)))
    if key not in _UNDIST_CACHE:
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (int(size[0]), int(size[1])), cv2.CV_32FC1)
        _UNDIST_CACHE[key] = (m1, m2, K_new)
    return _UNDIST_CACHE[key]


# --------------------------------------------------------------------------- #
def self_check(seed=0):
    import cv2
    rng = np.random.RandomState(seed)
    ok = True

    # 1) scale_K 与 cv2.resize 一致：在大图上画一个亚像素高斯斑点，缩放后用质心定位
    W, H = 1920, 1080
    K = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0, 0, 1]])
    worst = 0.0
    for _ in range(20):
        u, v = rng.uniform(200, W - 200), rng.uniform(200, H - 200)
        yy, xx = np.mgrid[0:H, 0:W]
        img = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 12.0 ** 2)).astype(np.float32)
        for (nw, nh) in ((512, 288), (640, 360), (777, 431)):
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = nw / W, nh / H
            ys, xs = np.mgrid[0:nh, 0:nw]
            cu, cv_ = (small * xs).sum() / small.sum(), (small * ys).sum() / small.sum()
            eu, ev = (u + 0.5) * sx - 0.5, (v + 0.5) * sy - 0.5
            worst = max(worst, abs(cu - eu), abs(cv_ - ev))
    print('scale_K vs cv2.resize(INTER_AREA) 质心偏差最大 %.4f px（应 < 0.05）' % worst)
    ok &= worst < 0.05

    # 2) hflip_K：翻转图像上的投影 == 用翻转内参投影镜像点
    pts = rng.uniform([-2, -1, 2], [2, 1, 8], (200, 3))
    Kf = hflip_K(K, W)
    uv = (K @ pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    mir = pts * np.array([-1, 1, 1])
    uvf = (Kf @ mir.T).T
    uvf = uvf[:, :2] / uvf[:, 2:3]
    e = np.abs((W - 1 - uv[:, 0]) - uvf[:, 0]).max() + np.abs(uv[:, 1] - uvf[:, 1]).max()
    print('hflip_K 投影误差 %.2e px' % e)
    ok &= e < 1e-9

    # 3) 视线相对旋转往返 + 性质
    Rc = R.random(500, random_state=seed).as_matrix()
    dirs = rng.uniform([-1, -1, 0.2], [1, 1, 1], (500, 3))
    back = allo_to_ego(ego_to_allo(Rc, dirs), dirs)
    e = np.abs(back - Rc).max()
    z = ray_rotation(dirs) @ np.array([0, 0, 1.0])
    ez = np.abs(z - dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).max()
    on_axis = np.abs(ray_rotation(np.array([[0, 0, 1.0]]))[0] - np.eye(3)).max()
    print('allo<->ego 往返 %.2e | R_ray*z = 视线 %.2e | 光轴上 R_ray = I %.2e' % (e, ez, on_axis))
    ok &= e < 1e-9 and ez < 1e-9 and on_axis < 1e-12

    # 4) 翻转与视线相对旋转可交换：hflip(R_allo) == allo(hflip(R_cam), M d)（翻转标签 R' = M R S）
    a1 = hflip_rotation(ego_to_allo(Rc, dirs))
    a2 = ego_to_allo(hflip_rotation(Rc), dirs * np.array([-1, 1, 1]))
    e = np.abs(a1 - a2).max()
    print('翻转 与 allocentric 可交换 %.2e' % e)
    ok &= e < 1e-9

    # 5) 虚拟深度：缩放 s 后 Zv 除以 s，解码回米制不变
    f = focal(K)
    Z = rng.uniform(1, 8, 100)
    for s in (0.8, 1.0, 1.25):
        Ks = scale_K(K, s, s)
        zv = to_virtual_depth(Z, focal(Ks), 512)
        e1 = np.abs(zv - to_virtual_depth(Z, f, 512) / s).max()
        e2 = np.abs(from_virtual_depth(zv, focal(Ks), 512) - Z).max()
        ok &= e1 < 1e-9 and e2 < 1e-9
    print('虚拟深度 缩放一致性 / 往返: 通过' if ok else '虚拟深度: 未通过')

    # 6) 旧仿真假内参识别
    Kt, hit = legacy_sim_intrinsic_fix(np.array([[640, 0, 642], [0, 646.4, 358.5], [0, 0, 1]]), 1280, 720)
    _, hit2 = legacy_sim_intrinsic_fix(K, 1920, 1080)
    print('假内参识别: 仿真 %s -> fx=fy=%.1f c=(%.1f,%.1f) | MAV6D 标定值 %s' % (hit, Kt[1, 1], Kt[0, 2], Kt[1, 2], hit2))
    ok &= hit and not hit2 and Kt[1, 1] == 640 and Kt[0, 2] == 639.5

    print('camera_geometry 自检: %s' % ('通过' if ok else '未通过'))
    return bool(ok)


if __name__ == '__main__':
    self_check()
