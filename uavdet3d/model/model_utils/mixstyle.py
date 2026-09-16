# -*- coding: utf-8 -*-
"""MixStyle：训练时随机混合样本之间的特征统计量，逼网络不要依赖「风格」。

为什么用它（2026-09-16）：我们量到的失败就是【特征统计量跨域漂移】——
只把 BN 的 running statistics 换成无标签真实图重新统计（AdaBN，不训练不碰标签），
P1 在 MAV6D 上位置中位就从 4.593 掉到 1.679 m、2D 中心误差 108 -> 10.8 px
（tools/annot_audit/diag_adabn.py）。AdaBN 是事后补救、且需要目标域数据；
MixStyle 是在训练时就把这条捷径堵死，只用源域数据。

做法（Zhou et al., ICLR 2021）：对每个样本算逐通道的实例统计量 (mu, sigma)，
先按自己的统计量归一化，再用「自己的」和「批内另一个样本的」统计量的随机凸组合还原：
    x_out = sigma_mix * (x - mu) / sigma + mu_mix,   lam ~ Beta(alpha, alpha)
只在训练时生效，推理时是恒等映射，不加参数、不改输出形状。

「风格」在我们这里具体指什么：亮度/对比度/色偏，以及【细节强度】——
仿真在 MAV6D 的工作点上被放大 1.32 倍，目标块细节量只剩真实的 1/16（§12.2），
而深度头正是把细节量当成了距离线索。这一轴恰好属于实例统计量能表达的范围。

用法（MODEL.BACKBONE_2D）：
    MIXSTYLE: {p: 0.5, alpha: 0.1, layers: [1, 2]}
p<=0 或不配 = 完全关闭，逐位等同于没有这个模块。layers 指在骨干的第几个 stage 之后插入
（论文的建议是只插在浅层：深层特征已经是语义，混了会伤判别力）。
"""
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def extra_repr(self):
        return 'p=%g, alpha=%g' % (self.p, self.alpha)

    def forward(self, x):
        if not self.training or self.p <= 0 or x.size(0) < 2:
            return x
        if torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu.detach()) / sig.detach()
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1.0 - lam)
        sig_mix = sig * lam + sig[perm] * (1.0 - lam)
        return x_norm * sig_mix.detach() + mu_mix.detach()


def build_mixstyle(model_cfg):
    """-> (MixStyle 模块或 None, 要插入的 stage 序号集合)。不配或 p<=0 时返回 (None, set())。"""
    cfg = model_cfg.get('MIXSTYLE', None) or {}
    p = float(cfg.get('p', 0.0) or 0.0)
    if p <= 0:
        return None, set()
    layers = set(int(v) for v in cfg.get('layers', [1, 2]))
    return MixStyle(p=p, alpha=float(cfg.get('alpha', 0.1))), layers
