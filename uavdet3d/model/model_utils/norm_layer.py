# -*- coding: utf-8 -*-
"""可切换的归一化层。

为什么要这个（2026-09-16）：BatchNorm 的 running_mean / running_var 是在【训练域】上统计的，
换到真实域后内部激活整体偏。实测（tools/annot_audit/diag_adabn.py）：只用无标签真实图重新统计 BN
（不训练、不碰标签），P1 在 MAV6D 上位置中位 4.593 -> 1.679 m、2D 中心误差 108 -> 10.8 px。
说明这一项是纯模型侧的域依赖。GroupNorm / InstanceNorm 没有 running statistics，
每个样本自己归一化，结构上就不存在这个问题（InstanceNorm 还天然对「风格/细节强度」不敏感）。

注意：换了归一化类型的权重与 BN 权重【不可互相加载】（BN 多 running_mean/var 两个 buffer）。
"""
import torch.nn as nn


def build_norm(num_features, kind='bn', groups=32):
    k = str(kind).lower()
    if k in ('bn', 'batch', 'batchnorm'):
        return nn.BatchNorm2d(num_features)
    if k in ('gn', 'group', 'groupnorm'):
        g = int(groups)
        while g > 1 and num_features % g != 0:
            g //= 2
        return nn.GroupNorm(g, num_features)
    if k in ('in', 'instance', 'instancenorm'):
        return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)
    raise ValueError('NORM_TYPE 只能是 bn / gn / in，收到 %r' % (kind,))
