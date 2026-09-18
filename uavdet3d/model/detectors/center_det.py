import inspect
import torch
import numpy as np
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder import all_object_encoders
from uavdet3d.datasets.pre_processor.pre_processor import denormalize_regression
import torch.nn.functional as F


class CenterDet(DetectorTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.center_decoder = all_object_encoders[self.model_cfg.POST_PROCESSING.DECONDER]

        # ---- 锚点蒸馏（2026-09-18）：把「深度所依赖的那个函数」钉在早期权重上 ----
        # 为什么：跨域深度在第 4~5 轮最好、之后退化（0.50 -> 0.77 m），而朝向要靠长训练（折算 61 -> 41）。
        # 冻结骨干只训旋转头无效（15 轮角度纹丝不动），说明朝向的提升来自骨干特征本身。
        # 所以让骨干继续学，但约束 size2d 头在同一张图上的输出必须贴住早期教师 —— 钉的是函数行为，不是权重。
        self.anchor_cfg = self.model_cfg.get('ANCHOR_DISTILL', None)
        self.anchor_heads = [str(x) for x in (self.anchor_cfg.get('HEADS', ['size2d']) if self.anchor_cfg else [])]
        self._teacher = []          # 放进 list，避免被注册成子模块（否则会跟着存进 checkpoint）
        self._anchor_pred = None
        if self.anchor_cfg and str(self.anchor_cfg.get('CKPT', '')):
            import copy as _copy
            from uavdet3d.model import build_network
            tcfg = _copy.deepcopy(self.model_cfg)
            tcfg.pop('ANCHOR_DISTILL', None)
            t = build_network(tcfg, dataset)
            n_ld, n_tot = t.load_params_from_file(str(self.anchor_cfg.CKPT), to_cpu=False)
            assert n_ld == n_tot, ('锚点教师权重没全载入', n_ld, n_tot)
            for prm in t.parameters():
                prm.requires_grad_(False)
            self._teacher.append(t.cuda().eval())
            print('锚点蒸馏: 教师 %s，头 %s，权重 %.2f' % (self.anchor_cfg.CKPT, self.anchor_heads,
                                                     float(self.anchor_cfg.get('WEIGHT', 4.0))), flush=True)

        self.max_num = self.model_cfg.POST_PROCESSING.MAX_OBJ
        self.score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH

        # 解码用的旋转表示必须与编码一致，所以【只从 DATA_CONFIG 读一处】，
        # 不在 MODEL 下再放一个同名开关，避免两边写岔。
        # 深度模式 / 视线相对旋转 / 欧拉顺序同理，与预处理共用 encoder_geometry_kwargs
        from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs
        self.dec_kwargs = encoder_geometry_kwargs(self.dataset.dataset_cfg, self.center_decoder,
                                                  self.model_cfg.POST_PROCESSING.DECONDER)

    def forward(self, batch_dict):

        if self.training and self._teacher:
            with torch.no_grad():
                tb = {k: v for k, v in batch_dict.items()}
                for cur_module in self._teacher[0].module_list:
                    tb = cur_module(tb)
                self._anchor_pred = {k: tb['pred_center_dict'][k].detach() for k in self.anchor_heads
                                     if k in tb['pred_center_dict']}

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }

            return ret_dict
        else:
            return self.post_processing(batch_dict)

    def get_training_loss(self):

        loss = self.dense_head_2d.get_loss()

        if self._teacher and self._anchor_pred:
            w = float(self.anchor_cfg.get('WEIGHT', 4.0))
            pred = self.dense_head_2d.forward_loss_dict['pred_center_dict']
            fg = self.dense_head_2d.forward_loss_dict.get('fg_mask', None)
            for k, tv in self._anchor_pred.items():
                p = pred[k]
                if fg is not None:
                    m = (fg.expand(-1, -1, p.shape[-3], -1, -1).reshape(-1) > 0)
                    if m.sum() == 0:
                        continue
                    term = torch.abs(p.reshape(-1)[m] - tv.reshape(-1)[m]).mean()
                else:
                    term = torch.abs(p - tv).mean()
                self.dense_head_2d.loss_terms['anchor_' + k] = float(term.detach())
                loss = loss + w * term

        return loss

    def post_processing(self, batch_dict):

        batch_size = batch_dict['batch_size']

        im_num = self.dataset.im_num

        all_pred_boxes9d = []

        all_confidence = []

        hm = batch_dict['pred_center_dict']['hm']
        center_res = batch_dict['pred_center_dict']['center_res']
        center_dis = batch_dict['pred_center_dict']['center_dis']
        dim = batch_dict['pred_center_dict']['dim']
        rot = batch_dict['pred_center_dict']['rot']
        # 有 size2d 头且配置要求时，深度改由「预测的 3D 尺寸 + 姿态」和「预测的 2D 跨度」几何解出，
        # 不用自由回归的 center_dis（见 object_encoder_mav6d.center_point_decoder 里的说明）
        size2d = batch_dict['pred_center_dict'].get('size2d', None) \
            if self.model_cfg.POST_PROCESSING.get('DEPTH_FROM_SIZE2D', False) else None
        # 关键点 + PnP：位姿完全由 8 个角点解出（2026-09-18）。能跨域迁移的是定位类的 2D 量，
        # 尺度类的量（size2d 头：域内 0.99、真实域宽 0.85 高 0.53）不迁移，所以把位姿交给几何求解。
        kp2d = batch_dict['pred_center_dict'].get('kp2d', None) \
            if self.model_cfg.POST_PROCESSING.get('POSE_FROM_KP2D', False) else None

        def reshape_t(tensor, batch_size):
            BK, C, W, H = tensor.shape
            return tensor.reshape(batch_size, -1, C, W, H)

        size2d = reshape_t(size2d, batch_size) if size2d is not None else None
        kp2d = reshape_t(kp2d, batch_size) if kp2d is not None else None
        hm = reshape_t(hm, batch_size)
        center_res = reshape_t(center_res, batch_size)
        center_dis = reshape_t(center_dis, batch_size)
        dim = reshape_t(dim, batch_size)
        rot = reshape_t(rot, batch_size)

        for batch_id in range(batch_size):
            intrinsic = batch_dict['intrinsic'][batch_id]  # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id]  # 4, 4
            distortion = batch_dict['distortion'][batch_id]  # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id]  # 2,
            new_im_size = batch_dict['new_im_size'][batch_id]  # 2,
            # obj_size = batch_dict['obj_size'][batch_id]  # 3,
            # 原来这里有三行 print(intrinsic/extrinsic/distortion)，
            # 每个样本刷一次，评测几千帧时把真正的输出全冲掉了，删除。
            stride = batch_dict['stride'][batch_id]

            this_hm = hm[batch_id]
            this_hm = torch.sigmoid(this_hm)
            this_center_res = center_res[batch_id]
            # this_center_dis = torch.exp(center_dis[batch_id])*self.dataset.dataset_cfg.MAX_DIS
            # this_dim = torch.exp(dim[batch_id]) #*self.dataset.dataset_cfg.MAX_SIZE

            # 与 pre_processor 的 TARGET_NORM / DEPTH_TARGET 严格互逆（共用 denormalize_regression，只从 DATA_CONFIG 读）
            this_center_dis, this_dim = denormalize_regression(self.dataset.dataset_cfg, self.model_cfg.POST_PROCESSING,
                                                               center_dis[batch_id], dim[batch_id])
            this_rot = rot[batch_id]
            this_size2d = size2d[batch_id] if size2d is not None else None
            this_kp2d = kp2d[batch_id] if kp2d is not None else None

            dec_kwargs = dict(self.dec_kwargs)
            if this_size2d is not None:
                dec_kwargs['size2d'] = this_size2d
                dec_kwargs['size2d_mode'] = str(self.model_cfg.POST_PROCESSING.get('SIZE2D_MODE', 'both'))
            if this_kp2d is not None:
                from uavdet3d.datasets.pre_processor.pre_processor import KP_SCALE
                dec_kwargs['kp2d'] = this_kp2d
                dec_kwargs['kp_scale'] = KP_SCALE

            pred_boxes9d, confidence = self.center_decoder(this_hm,
                                                           this_center_res,
                                                           this_center_dis,
                                                           this_dim,
                                                           this_rot,
                                                           intrinsic,
                                                           extrinsic,
                                                           distortion,
                                                           new_im_size[0],
                                                           new_im_size[1],
                                                           raw_im_size[0],
                                                           raw_im_size[1],
                                                           stride,
                                                           im_num,
                                                           self.max_num,
                                                           **dec_kwargs)
            # pred_boxes9d[:,0:2]+=0.2

            all_pred_boxes9d.append(pred_boxes9d[confidence > self.score_thresh])
            all_confidence.append(confidence[confidence > self.score_thresh])

        batch_dict['pred_boxes9d'] = all_pred_boxes9d
        batch_dict['confidence'] = all_confidence

        return batch_dict
