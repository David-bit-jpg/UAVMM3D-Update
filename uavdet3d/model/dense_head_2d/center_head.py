import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

from uavdet3d.model.model_utils.norm_layer import build_norm


class CenterHead(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHead, self).__init__()

        self.model_cfg = model_cfg

        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG

        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT

        self.in_c = self.model_cfg.INPUT_CHANNELS

        # 归一化类型：'bn'(缺省) / 'gn' / 'in'，与骨干用同一个开关
        self.norm_type = self.model_cfg.get('NORM_TYPE', 'bn')

        # 回归头输出层是否带 bias（见下面建头处的说明）。缺省 false = 历史行为
        self.head_bias = bool(self.model_cfg.get('HEAD_BIAS', False))

        # ---- 热力图损失 ----
        # 'bce'  : 原实现，逐像素普通 BCE / (正样本数+1)。背景占 99.9%，简单负样本主导梯度。
        # 'focal': CenterNet / CornerNet 的 penalty-reduced focal loss（alpha=2, beta=4），
        #          高斯峰值=1 的格子是正样本，其余按 (1-gt)^4 降权，按正样本数归一。
        self.hm_loss_type = self.model_cfg.get('HM_LOSS', 'bce')
        assert self.hm_loss_type in ('bce', 'focal'), self.hm_loss_type
        # hm 输出层偏置先验：None = 原实现（无偏置）；给数值 = 带偏置并初始化成该值
        # （CenterNet 用 -2.19，即 sigmoid 后 0.1，训练初期不会被海量负样本的损失淹没）
        self.hm_bias_prior = self.model_cfg.get('HM_BIAS_PRIOR', None)

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']
            # HEAD_BIAS（2026-09-16）：原来只有 hm 的输出层有 bias，其余四个头没有 —— 于是
            # 输出 = W·f 是 ReLU 后特征的一次齐次函数，特征幅值跨域一漂移，预测就【整体缩放】。
            # 实测 S1 在 MAV6D 上头内激活是仿真的 3.35 倍、深度输出 1.76 倍，而零样本误差
            # 正是「预测 = 2.21×真值 − 0.27」（截距为零的纯乘性误差），就是无偏置头的签名。
            # 内部对照：唯一有 bias 的 hm 恰恰是唯一迁移得好的头。缺省 false = 与之前逐位一致。
            hm_bias = (cur_head_name == 'hm' and self.hm_bias_prior is not None)
            with_bias = hm_bias or (self.head_bias and cur_head_name != 'hm')

            fc = nn.Sequential(
                nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                build_norm(conv_mid, self.norm_type),
                nn.ReLU(),
                nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=with_bias)
            )
            if hm_bias:
                nn.init.constant_(fc[-1].bias, float(self.hm_bias_prior))
            elif with_bias:
                # 真值标准化后均值为 0，所以 0 就是正确的先验；center_res 真值在 [0,1) 故取 0.5
                nn.init.constant_(fc[-1].bias, 0.5 if cur_head_name == 'center_res' else 0.0)
            self.__setattr__(cur_head_name, fc)

        self.forward_loss_dict=dict()

        # ---- 损失权重 ----
        # 原实现把五项【等权相加】。实测（tools/loss_term_audit.py，C 臂最终权重 25 个 batch）：
        #   hm 0.2743 (88.8%) / center_res 0.0187 (6.1%) / center_dis 0.0033 (1.1%)
        #   dim 0.0003 (0.1%) / rot 0.0121 (3.9%)
        # hm 是个近乎常数的地板（标准差 0.0005），rot 只占 3.9%。
        # 不给权重就没法把梯度往真正学不好的头上调。缺省全 1.0，行为与原来一致。
        lw = self.model_cfg.get('LOSS_WEIGHTS', None) or {}
        self.loss_weights = {k: float(lw.get(k, 1.0)) for k in self.head_keys}

        # ---- 前景掩码 ----
        # 'center_dis'（缺省）: 用 gt['center_dis'] > 0 当前景指示。它是深度的归一化值，
        #   在目标像素上恒为正、其余位置恒为 0，是唯一干净的指示通道。
        # 'nonzero'          : 旧行为，对每个通道各自取 |gt| > 0。真值恰好为 0 的位置会被
        #   丢掉（cos(±90°)、中心正好落在整像素时的 res），换成 6D 旋转表示后更严重
        #   —— 6D 向量里 0 是完全合法的分量。
        self.fg_mask_mode = self.model_cfg.get('LOSS_FG_MASK', 'center_dis')
        if self.fg_mask_mode == 'center_dis' and 'center_dis' not in self.head_keys:
            raise ValueError("LOSS_FG_MASK='center_dis' 需要 HEAD_ORDER 里含 center_dis")

        # 每项的当前值，供训练日志打印，省得再单独跑一遍 loss_term_audit
        self.loss_terms = {}

    @staticmethod
    def focal_loss(logits, gt, alpha=2.0, beta=4.0):
        """CenterNet penalty-reduced focal loss。gt 是高斯热图，目标中心格恰好为 1。"""
        g = gt.float().reshape(-1)
        p = torch.sigmoid(logits.float().reshape(-1)).clamp(1e-4, 1.0 - 1e-4)
        pos = (g >= 1.0 - 1e-6).float()
        neg = 1.0 - pos
        pos_loss = torch.log(p) * torch.pow(1.0 - p, alpha) * pos
        neg_loss = torch.log(1.0 - p) * torch.pow(p, alpha) * torch.pow(1.0 - g, beta) * neg
        return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)

    def get_loss(self):
        pred = self.forward_loss_dict['pred_center_dict']
        gt = self.forward_loss_dict['gt_center_dict']

        # (B, K, 1, h, w)；下面按各头的通道数 expand。
        # 优先用编码器给的显式 fg_mask —— 真值标准化后 center_dis 会有负数，
        # 再拿 `> 0` 当前景指示就会把近距离目标当成背景丢掉。
        if self.forward_loss_dict.get('fg_mask', None) is not None:
            fg = self.forward_loss_dict['fg_mask'] > 0
        else:
            fg = (gt['center_dis'] > 0) if self.fg_mask_mode == 'center_dis' else None

        loss = 0
        self.loss_terms = {}
        for cur_name in self.head_keys:
            w = self.loss_weights.get(cur_name, 1.0)
            p, g = pred[cur_name], gt[cur_name]

            if cur_name == 'hm' and self.hm_loss_type == 'focal':
                term = self.focal_loss(p, g)
            elif cur_name == 'hm':
                l = F.binary_cross_entropy(torch.sigmoid(p.reshape(-1)), g.reshape(-1),
                                           reduction='none')
                term = l.sum() / ((g > 0).sum() + 1)
            elif cur_name == 'kp2d':
                # kp2d 可以写在中心格的邻域里（DATA_CONFIG.KP_NEIGHBOR），所以用它自己的非零掩码，
                # 不跟着只写中心格的 center_dis/dim/rot 用同一个前景掩码。
                m = torch.abs(g).sum(dim=-3, keepdim=True).expand_as(g) > 0
            else:
                m = (torch.abs(g) > 0) if fg is None else fg.expand_as(g)
                m = m.reshape(-1)
                if m.sum() == 0:
                    # 整个 batch 没有有效目标（例如全被 filter_box_outside 滤掉），
                    # 返回一个连着计算图的 0，避免 backward 报错
                    term = (p.reshape(-1) * 0).sum()
                else:
                    term = torch.abs(g.reshape(-1)[m] - p.reshape(-1)[m]).mean()

            self.loss_terms[cur_name] = float(term.detach())
            loss = loss + w * term

        return loss

    def forward(self, batch_dict):

        pred_dict = {}

        gt_dict = {}

        x = batch_dict['features_2d']

        B, K, C, W, H = x.shape

        x = x.reshape(B*K, C, W, H)

        for cur_name in self.head_keys:

            pred_dict[cur_name] = self.__getattr__(cur_name)(x)

            if self.training:
                gt_dict[cur_name] = batch_dict[cur_name]

        batch_dict['pred_center_dict'] = pred_dict

        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict
            self.forward_loss_dict['fg_mask'] = batch_dict.get('fg_mask', None)

        return batch_dict
