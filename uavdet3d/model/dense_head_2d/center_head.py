import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2


class CenterHead(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHead, self).__init__()

        self.model_cfg = model_cfg

        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG

        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT

        self.in_c = self.model_cfg.INPUT_CHANNELS

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            fc = nn.Sequential(
                nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(conv_mid),
                nn.ReLU(),
                nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            )
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

    def get_loss(self):
        pred = self.forward_loss_dict['pred_center_dict']
        gt = self.forward_loss_dict['gt_center_dict']

        # (B, K, 1, h, w)；下面按各头的通道数 expand
        fg = (gt['center_dis'] > 0) if self.fg_mask_mode == 'center_dis' else None

        loss = 0
        self.loss_terms = {}
        for cur_name in self.head_keys:
            w = self.loss_weights.get(cur_name, 1.0)
            p, g = pred[cur_name], gt[cur_name]

            if cur_name == 'hm':
                l = F.binary_cross_entropy(torch.sigmoid(p.reshape(-1)), g.reshape(-1),
                                           reduction='none')
                term = l.sum() / ((g > 0).sum() + 1)
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

        return batch_dict
