# -*- coding: utf-8 -*-
"""旋转的两种回归表示，以及它们与旋转矩阵之间的转换。

背景（实测数据，不是理论顾虑）：
    原来 rot 头的 6 个通道是 (cos, sin) x 三个欧拉角。cos/sin 编码确实解决了
    ±180° 处的绕回问题 —— MAV6D 训练集里 a3 有 31% 的帧 |a3| > 170°，这部分是被
    正确处理的。但它解决不了【万向锁】：as_euler('xyz') 在中间角 a2 接近 ±90° 时
    (a1, a3) 不再唯一，同一个姿态可以对应无穷多组角。MAV6D 训练集实测
    |a2| > 80° 占 9.5%、> 85° 占 3.4%，这部分数据给网络的是自相矛盾的监督。

    'r6d' 用旋转矩阵的前两列作为 6 维表示（Zhou et al., CVPR 2019,
    "On the Continuity of Rotation Representations in Neural Networks"），
    解码时用 Gram-Schmidt 正交化还原。这个映射在 SO(3) 上处处连续，没有退化点，
    而且通道数恰好也是 6，网络结构一个字都不用改。

两种表示的名字：
    'euler6' —— 原行为，(cos,sin) x 3
    'r6d'    —— 6D 连续表示
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

REPRS = ('euler6', 'r6d')


def euler_to_vec(angles, seq, repr_name):
    """欧拉角 -> 6 维回归目标。angles: (3,) 或 (N,3)，返回 (6,) 或 (N,6)。"""
    a = np.asarray(angles, dtype=np.float64)
    single = (a.ndim == 1)
    a = a.reshape(-1, 3)

    if repr_name == 'euler6':
        out = np.stack([np.cos(a[:, 0]), np.sin(a[:, 0]),
                        np.cos(a[:, 1]), np.sin(a[:, 1]),
                        np.cos(a[:, 2]), np.sin(a[:, 2])], axis=1)
    elif repr_name == 'r6d':
        m = R.from_euler(seq, a).as_matrix()          # (N, 3, 3)
        # 取前两【列】。列是物体坐标轴在相机系下的方向，比取行更有几何意义。
        out = np.concatenate([m[:, :, 0], m[:, :, 1]], axis=1)
    else:
        raise ValueError('未知的旋转表示: %s，可选 %s' % (repr_name, REPRS))

    return out[0] if single else out


def vec_to_euler(vec, seq, repr_name):
    """euler_to_vec 的逆。vec: (6,) 或 (N,6)，返回 (3,) 或 (N,3)。"""
    v = np.asarray(vec, dtype=np.float64)
    single = (v.ndim == 1)
    v = v.reshape(-1, 6)

    if repr_name == 'euler6':
        a1 = np.arctan2(v[:, 1], v[:, 0])
        a2 = np.arctan2(v[:, 3], v[:, 2])
        a3 = np.arctan2(v[:, 5], v[:, 4])
        out = np.stack([a1, a2, a3], axis=1)
    elif repr_name == 'r6d':
        out = R.from_matrix(vec_to_matrix(v, repr_name)).as_euler(seq)
    else:
        raise ValueError('未知的旋转表示: %s，可选 %s' % (repr_name, REPRS))

    return out[0] if single else out


def vec_to_matrix(vec, repr_name, seq=None):
    """6 维表示 -> 旋转矩阵 (N,3,3)。r6d 走 Gram-Schmidt，网络输出不必是正交的。"""
    v = np.asarray(vec, dtype=np.float64).reshape(-1, 6)

    if repr_name == 'r6d':
        a1, a2 = v[:, 0:3], v[:, 3:6]
        n1 = np.linalg.norm(a1, axis=1, keepdims=True)
        # 网络刚初始化时可能输出接近零的向量，兜个底避免除零
        n1 = np.where(n1 < 1e-8, 1.0, n1)
        b1 = a1 / n1
        proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = a2 - proj
        n2 = np.linalg.norm(b2, axis=1, keepdims=True)
        degenerate = (n2 < 1e-8).reshape(-1)
        n2 = np.where(n2 < 1e-8, 1.0, n2)
        b2 = b2 / n2
        if degenerate.any():
            # a2 与 a1 共线时第二列没定义，随便找一个与 b1 正交的方向顶上
            alt = np.tile(np.array([1.0, 0.0, 0.0]), (degenerate.sum(), 1))
            bad = b1[degenerate]
            alt = np.where(np.abs(bad[:, 0:1]) > 0.9,
                           np.tile(np.array([0.0, 1.0, 0.0]), (degenerate.sum(), 1)), alt)
            alt = alt - np.sum(bad * alt, axis=1, keepdims=True) * bad
            b2[degenerate] = alt / np.linalg.norm(alt, axis=1, keepdims=True)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=2)          # 按【列】拼

    if repr_name == 'euler6':
        assert seq is not None, 'euler6 转旋转矩阵需要欧拉角顺序'
        return R.from_euler(seq, vec_to_euler(v, seq, 'euler6')).as_matrix()

    raise ValueError('未知的旋转表示: %s，可选 %s' % (repr_name, REPRS))


def self_check(n=2000, seq='xyz', seed=0):
    """两种表示各跑一遍欧拉角往返，并单独统计万向锁附近的误差。"""
    rng = np.random.RandomState(seed)
    mats = R.random(n, random_state=rng).as_matrix()
    eul = R.from_matrix(mats).as_euler(seq)

    print('自检 (%d 个随机姿态, seq=%s)' % (n, seq))
    for name in REPRS:
        v = euler_to_vec(eul, seq, name)
        m2 = vec_to_matrix(v, name, seq)
        # 用测地距离比，避免欧拉角本身的多值性干扰
        err = np.degrees(R.from_matrix(
            np.einsum('nij,nkj->nik', m2, mats)).magnitude())
        near_lock = np.abs(np.degrees(eul[:, 1])) > 80
        print('  %-7s 往返测地误差 中位 %.2e deg  最大 %.2e deg  |  近万向锁(%d 个) 最大 %.2e deg'
              % (name, np.median(err), err.max(), near_lock.sum(),
                 err[near_lock].max() if near_lock.any() else 0.0))

    # 连续性才是换表示的理由：姿态微小变化时，回归目标也应该微小变化。
    # 分开统计「近万向锁」和「其余」两组 —— 差别只出现在前者，
    # 而 MAV6D 训练集里前者占 9.5%。
    print('\n连续性（对同一姿态施加 0.5 度扰动后，6 维回归目标的变化量）')
    axis = rng.randn(n, 3)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    small = R.from_rotvec(axis * np.radians(0.5))
    eul2 = R.from_matrix(np.einsum('nij,njk->nik', small.as_matrix(), mats)).as_euler(seq)
    near_lock = np.abs(np.degrees(eul[:, 1])) > 80
    for label, sel in (('全部', np.ones(n, bool)),
                       ('近万向锁 |a2|>80', near_lock),
                       ('其余', ~near_lock)):
        if not sel.any():
            continue
        print('  --- %s (%d 个) ---' % (label, sel.sum()))
        for name in REPRS:
            d = np.linalg.norm(euler_to_vec(eul2[sel], seq, name) -
                               euler_to_vec(eul[sel], seq, name), axis=1)
            print('      %-7s 中位 %.4f  95%% %.4f  最大 %.4f' %
                  (name, np.median(d), np.percentile(d, 95), d.max()))


if __name__ == '__main__':
    self_check()
