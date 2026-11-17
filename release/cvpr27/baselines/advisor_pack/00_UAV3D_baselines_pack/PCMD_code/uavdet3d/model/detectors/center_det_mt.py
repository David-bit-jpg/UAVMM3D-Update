# -*- coding: utf-8 -*-
"""多任务学生：RGB 学生除了位姿的五个头，再从【同一张特征图】幻觉出其它模态（辅助任务），
只在源域训练时用，权重存下来仍是干净的 RGB CenterDet（aux_heads.* 迁移时按名字跳过）。

动机（docs/results 第三、四版）：
  RGB 学生零样本在 MAV6D 上像素位置能找对、深度差 5 倍、朝向几乎无信息；多模态教师的优势在召回与深度。
  与其只在特征层面向教师看齐（蒸馏），不如直接逼 RGB 特征去解释 LiDAR 深度 / IR / 无人机 LiDAR 命中 ——
  Hoffman et al. 2016 的「模态幻觉」在 CenterNet 上的对应物；所有目标都在 batch 的 image 通道里，
  数据集一行不用改（MODALITIES 全开，学生在模型里只切前 3 通道，其余通道当监督）。

辅助头（stride 与 features_2d 相同，缺省 8）：
  ir     : 池化后的 IR 灰度（[0,1]），L1
  depth  : 池化后的 LiDAR 归一化深度（只在有回波的像素上平均；整格无回波不计损失），L1
  tag    : 该格子里是否有打在无人机上的 LiDAR 点（max 池化 > 0.5），带 pos_weight 的 BCE —— 稠密的「无人机在哪」
配置（MODEL.AUX，权重 0 = 关闭该头）：
  AUX: {W_IR: 1.0, W_DEPTH: 1.0, W_TAG: 1.0, CONV_DIM: 128, TAG_POS_WEIGHT: 10.0}
可与 DISTILL 同时开（继承 CenterDetKD）：蒸馏 + 多任务。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_det_kd import CenterDetKD

MODAL_ORDER = ('rgb', 'ir', 'depth', 'tag')


class CenterDetMT(CenterDetKD):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        a = model_cfg.get('AUX', None) or {}
        self.aux_w = {k: float(a.get(k, 0.0)) for k in ('W_IR', 'W_DEPTH', 'W_TAG')}
        self.tag_pos_weight = float(a.get('TAG_POS_WEIGHT', 10.0))
        mods = [m for m in MODAL_ORDER if m in list(dataset.dataset_cfg.get('MODALITIES', ['rgb']))]
        self.aux_ch, c = {}, 0
        for m in mods:
            n = 3 if m == 'rgb' else 1
            self.aux_ch[m] = (c, c + n)
            c += n
        in_c = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        mid = int(a.get('CONV_DIM', 128))
        self.aux_heads = nn.ModuleDict()
        for wk, key in (('W_IR', 'ir'), ('W_DEPTH', 'depth'), ('W_TAG', 'tag')):
            if self.aux_w[wk] > 0:
                assert key in self.aux_ch, 'AUX.%s 需要 DATA_CONFIG.MODALITIES 里含 %s' % (wk, key)
                self.aux_heads[key] = nn.Sequential(
                    nn.Conv2d(in_c, mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(mid), nn.ReLU(),
                    nn.Conv2d(mid, 1, kernel_size=3, stride=1, padding=1))

    def aux_targets(self, img, hw):
        """img (N, Ctot, H, W) 数据集吐的全部通道；返回各辅助头的目标与有效掩码（N,1,h,w）。"""
        out = {}
        h, w = hw
        if 'ir' in self.aux_heads:
            c0, c1 = self.aux_ch['ir']
            out['ir'] = (F.adaptive_avg_pool2d(img[:, c0:c1], (h, w)), None)
        if 'depth' in self.aux_heads:
            c0, c1 = self.aux_ch['depth']
            d = img[:, c0:c1]
            valid = (d > 0).float()
            cnt = F.adaptive_avg_pool2d(valid, (h, w))
            t = F.adaptive_avg_pool2d(d * valid, (h, w)) / cnt.clamp_min(1e-6)
            out['depth'] = (t, cnt > 0)
        if 'tag' in self.aux_heads:
            c0, c1 = self.aux_ch['tag']
            out['tag'] = ((F.adaptive_max_pool2d(img[:, c0:c1], (h, w)) > 0.5).float(), None)
        return out

    def forward(self, batch_dict):
        ret = super().forward(batch_dict)
        if not self.training or not self.aux_heads:
            return ret
        feat = self._student_out['features_2d']                 # (B,K,C,h,w)
        B, K, C, h, w = feat.shape
        x = feat.reshape(B * K, C, h, w)
        img = batch_dict['image']
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        tg = self.aux_targets(img, (h, w))
        terms = self.dense_head_2d.loss_terms
        loss = ret['loss']
        for key, head in self.aux_heads.items():
            pred = head(x)
            t, m = tg[key]
            if key == 'tag':
                pw = torch.tensor(self.tag_pos_weight, device=pred.device)
                l = F.binary_cross_entropy_with_logits(pred, t, pos_weight=pw)
            elif m is not None:
                l = torch.abs(pred - t)[m].mean() if bool(m.any()) else (pred * 0).sum()
            else:
                l = torch.abs(pred - t).mean()
            terms['aux_' + key] = float(l.detach())
            loss = loss + self.aux_w['W_' + key.upper()] * l
        self.dense_head_2d.loss_terms = terms
        return {'loss': loss}
