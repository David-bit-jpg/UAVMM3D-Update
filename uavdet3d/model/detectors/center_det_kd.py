# -*- coding: utf-8 -*-
"""跨模态蒸馏：多模态教师（rgb+ir+depth+tag）-> 单目 RGB 学生。

设定与 MM'26《Beyond RGB》的 Stage 1 同构：源域有成对的多模态观测，目标域和部署只有
RGB。区别是我们的 IR/LiDAR 与 RGB 是【同一时刻同一场景】的成对数据，所以教师可以
真的吃多模态输入，而不只是"用 IR 训过的参数"。

数据集吃全部通道（rgb 排第一），这里：
    教师  <- image 全部通道，冻结，永远 eval
    学生  <- image[:, :, :3]（rgb）
蒸馏项（权重全在 DISTILL 里，设 0 即关闭；缺省全 0 时退化成普通 CenterDet）：
    W_FEAT        features_2d 逐像素 L2（论文的 multi-level feature distillation，这里只有一层）
    W_CENTER_COS  GT 中心像素处特征的 1-cos（论文的 RoI 余弦迁移在 CenterNet 里的对应物）
    W_HM          热图：学生 sigmoid 对教师 sigmoid 的 BCE（软标签，全图）
    W_REG         center_res / center_dis / dim / rot 四个回归头在前景像素上对教师输出的 L1
    W_CROSS       CrossKD：学生的头接教师的特征，输出仍用 GT 监督，约束「特征->头」接口

教师参数不进 state_dict（放在普通 list 里），所以存下来的 ckpt 就是一个干净的 RGB
CenterDet，可以直接被 --pretrained_model 载入去迁移。
"""
import copy

import torch
import torch.nn.functional as F

from .center_det import CenterDet


class CenterDetKD(CenterDet):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        d = model_cfg.get('DISTILL', None) or {}
        self.kd_w = {k: float(d.get(k, 0.0)) for k in ('W_FEAT', 'W_CENTER_COS', 'W_HM', 'W_REG', 'W_CROSS')}
        self.student_channels = int(d.get('STUDENT_CHANNELS', 3))
        # 特征项形式：'mse'（缺省，逐元素 L2）或 'cos'（逐像素 1-余弦，尺度不变——保留式微调用：
        # 学生 BN 用 batch 统计、教师用滑动统计，二者特征幅值差很多，但方向一致，L2 会被幅值差主导）
        self.feat_mode = str(d.get('FEAT_MODE', 'mse'))
        self.teacher_channels = int(d.get('TEACHER_CHANNELS', 6))
        self._teacher = []          # 用 list 藏起来：不注册为子模块，不进 state_dict / parameters()
        if any(v > 0 for v in self.kd_w.values()):
            self._build_teacher(d)

    # ------------------------------------------------------------------ #
    def _build_teacher(self, d):
        from easydict import EasyDict
        from uavdet3d.config import cfg_from_yaml_file
        from uavdet3d.model import build_network
        tcfg = EasyDict()
        cfg_from_yaml_file(d['TEACHER_CFG'], tcfg)
        teacher = build_network(model_cfg=tcfg.MODEL, dataset=self.dataset)
        teacher.load_params_from_file(filename=d['TEACHER_CKPT'], to_cpu=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.train = lambda mode=True, _m=teacher: _m       # 永远 eval，BN 统计量不动
        self._teacher.append(teacher)

    @property
    def teacher(self):
        return self._teacher[0] if self._teacher else None

    def cuda(self, device=None):
        if self.teacher is not None:
            self.teacher.cuda(device)
        return super().cuda(device)

    def to(self, *args, **kwargs):
        if self.teacher is not None:
            self.teacher.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ------------------------------------------------------------------ #
    def _student_batch(self, batch_dict):
        sb = dict(batch_dict)
        sb['image'] = batch_dict['image'][:, :, :self.student_channels]
        return sb

    def _teacher_batch(self, batch_dict):
        """翻译缓存时 image 布局是 [rgb_翻译(3) | ir | depth | tag | rgb_原始(3)]（数据集 TEACHER_RGB_DIR），
        教师要看原始 RGB + 其它模态，重排成 [rgb_原始 | ir | depth | tag]；普通缓存原样透传。"""
        img = batch_dict['image']
        tb = dict(batch_dict)
        if img.shape[2] > self.teacher_channels:
            tb['image'] = torch.cat([img[:, :, -3:], img[:, :, 3:-3]], dim=2)
        return tb

    def forward(self, batch_dict):
        sb = self._student_batch(batch_dict)
        for cur_module in self.module_list:
            sb = cur_module(sb)
        self._student_out = sb                               # 子类（多任务头）要用学生的 features_2d
        if not self.training:
            return self.post_processing(sb)

        loss = self.dense_head_2d.get_loss()
        terms = self.dense_head_2d.loss_terms
        if self.teacher is None:
            return {'loss': loss}

        with torch.no_grad():
            tb = self._teacher_batch(batch_dict)
            for cur_module in self.teacher.module_list:
                tb = cur_module(tb)
            t_feat = tb['features_2d']                       # (B,K,C,h,w)
            t_pred = tb['pred_center_dict']                  # {head: (B*K,c,h,w)}
        s_feat = sb['features_2d']
        s_pred = sb['pred_center_dict']
        gt = self.dense_head_2d.forward_loss_dict['gt_center_dict']
        fg = (gt['center_dis'] > 0)                          # (B,K,1,h,w) 前景 = GT 中心像素

        w = self.kd_w
        if w['W_FEAT'] > 0:
            if self.feat_mode == 'cos':
                l = (1.0 - F.cosine_similarity(s_feat, t_feat, dim=2)).mean()
            else:
                l = F.mse_loss(s_feat, t_feat)
            terms['kd_feat'] = float(l)
            loss = loss + w['W_FEAT'] * l
        if w['W_CENTER_COS'] > 0 and fg.any():
            B, K, C, h, wd = s_feat.shape
            m = fg.reshape(B * K, 1, h, wd).expand(-1, C, -1, -1)
            sf = s_feat.reshape(B * K, C, h, wd)[m].reshape(C, -1).t()
            tf = t_feat.reshape(B * K, C, h, wd)[m].reshape(C, -1).t()
            l = (1.0 - F.cosine_similarity(sf, tf, dim=1)).mean()
            terms['kd_ccos'] = float(l)
            loss = loss + w['W_CENTER_COS'] * l
        if w['W_HM'] > 0:
            l = F.binary_cross_entropy_with_logits(s_pred['hm'], torch.sigmoid(t_pred['hm']))
            terms['kd_hm'] = float(l)
            loss = loss + w['W_HM'] * l
        if w['W_REG'] > 0 and fg.any():
            l = 0
            for k in ('center_res', 'center_dis', 'dim', 'rot'):
                if k not in s_pred:
                    continue
                g = gt[k]
                m = fg.expand_as(g).reshape(-1)
                l = l + torch.abs(s_pred[k].reshape(-1)[m] - t_pred[k].reshape(-1)[m]).mean()
            terms['kd_reg'] = float(l)
            loss = loss + w['W_REG'] * l
        if w['W_CROSS'] > 0:
            # 学生的头接教师特征，仍用 GT 监督（CrossKD 的变体，论文式 (6)）
            xb = dict(sb)
            xb['features_2d'] = t_feat.detach()
            saved = dict(self.dense_head_2d.forward_loss_dict)
            self.dense_head_2d(xb)
            l = self.dense_head_2d.get_loss()
            self.dense_head_2d.forward_loss_dict = saved      # 还原，别污染学生自己的记录
            terms['kd_cross'] = float(l)
            loss = loss + w['W_CROSS'] * l
        self.dense_head_2d.loss_terms = terms
        return {'loss': loss}
