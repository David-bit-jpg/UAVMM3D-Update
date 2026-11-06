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
        # 前景掩码（2026-10-03 修正）：'center_dis'（缺省 = 旧行为）用 gt['center_dis'] > 0，但 TARGET_NORM standard 下
        # center_dis 目标 = (Zv − DEPTH_MEAN)/DEPTH_STD，只有 Zv > DEPTH_MEAN（5.95 m）的目标为正 —— 实测仿真训练前景格只覆盖 56%，
        # MAV6D test 目标 Zv 全部 < 5.95（中位 3.54），即 W_CENTER_COS / W_REG 在真实相关深度段覆盖率为 0。
        # 'fg_mask' = 用编码器给的显式前景掩码（与 center_head.get_loss 一致）。旧配置不设这个键，结果逐位不变。
        self.fg_mode = str(d.get('FG_MODE', 'center_dis'))
        assert self.fg_mode in ('center_dis', 'fg_mask'), self.fg_mode
        # 中心特征余弦的取法（2026-10-03 第二个修正）：'legacy'（缺省 = 旧行为）用 [m].reshape(C,-1).t()，布尔索引按
        # 帧→通道→像素 展平，reshape 却按 通道→所有格 解释，批内 ≥2 帧有前景时同一行混进不同帧不同通道（只有 1 个前景格时恰好对）。
        # 'per_target' = permute 到通道最后再按前景格取，每行是一个目标中心格的 C 维特征向量。
        self.ccos_index = str(d.get('CCOS_INDEX', 'legacy'))
        assert self.ccos_index in ('legacy', 'per_target'), self.ccos_index
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
        if self.fg_mode == 'fg_mask':
            fm = self.dense_head_2d.forward_loss_dict.get('fg_mask', None)
            assert fm is not None, 'FG_MODE=fg_mask 需要 batch 里有 fg_mask'
            fg = fm > 0                                      # (B,K,1,h,w) 前景 = 所有 GT 中心格
        else:
            fg = (gt['center_dis'] > 0)                      # 旧行为：standard 归一化下只选中 Zv > DEPTH_MEAN 的目标

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
            if self.ccos_index == 'per_target':
                sel = fg[:, :, 0]                                         # (B,K,h,w)
                sf = s_feat.permute(0, 1, 3, 4, 2)[sel]                   # (N,C)：每行 = 一个前景格的特征向量
                tf = t_feat.permute(0, 1, 3, 4, 2)[sel]
            else:
                m = fg.reshape(B * K, 1, h, wd).expand(-1, C, -1, -1)
                sf = s_feat.reshape(B * K, C, h, wd)[m].reshape(C, -1).t()   # 旧行为（≥2 帧有前景时混行）
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
            # 除热图外的所有回归头（含 camnorm 管线的 size2d / kp2d），教师、学生都得有这个头
            for k in self.dense_head_2d.head_keys:
                if k == 'hm' or k not in s_pred or k not in t_pred or k not in gt:
                    continue
                # 掩码按【预测】的通道数展开，不按 gt：ROT_SYMMETRY 时 gt['rot'] 存了多组等价目标（通道数是预测的整数倍）
                p_s, p_t = s_pred[k], t_pred[k]                       # (B*K, c, h, w)
                BK, c, h, wd = p_s.shape
                m = fg.reshape(BK, 1, h, wd).expand(-1, c, -1, -1).reshape(-1)
                l = l + torch.abs(p_s.reshape(-1)[m] - p_t.reshape(-1)[m]).mean()
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
