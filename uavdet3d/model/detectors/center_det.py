import inspect
import torch
import numpy as np
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder import all_object_encoders
import torch.nn.functional as F


class CenterDet(DetectorTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.center_decoder = all_object_encoders[self.model_cfg.POST_PROCESSING.DECONDER]

        self.max_num = self.model_cfg.POST_PROCESSING.MAX_OBJ
        self.score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH

        # 解码用的旋转表示必须与编码一致，所以【只从 DATA_CONFIG 读一处】，
        # 不在 MODEL 下再放一个同名开关，避免两边写岔。
        # 深度模式 / 视线相对旋转 / 欧拉顺序同理，与预处理共用 encoder_geometry_kwargs
        from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs
        self.dec_kwargs = encoder_geometry_kwargs(self.dataset.dataset_cfg, self.center_decoder,
                                                  self.model_cfg.POST_PROCESSING.DECONDER)

    def forward(self, batch_dict):

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

        def reshape_t(tensor, batch_size):
            BK, C, W, H = tensor.shape
            return tensor.reshape(batch_size, -1, C, W, H)

        size2d = reshape_t(size2d, batch_size) if size2d is not None else None
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

            # 与 pre_processor 的 TARGET_NORM 严格互逆（只从 DATA_CONFIG 读一处，避免两边写岔）
            dcfg = self.dataset.dataset_cfg
            if str(dcfg.get('TARGET_NORM', 'maxdis')) == 'standard':
                import numpy as _np
                _smu = torch.as_tensor(_np.asarray(dcfg.SIZE_MEAN, _np.float32).reshape(3, 1, 1),
                                       device=dim.device)
                _ssd = torch.as_tensor(_np.asarray(dcfg.SIZE_STD, _np.float32).reshape(3, 1, 1),
                                       device=dim.device).clamp(min=1e-6)
                this_center_dis = center_dis[batch_id] * float(dcfg.DEPTH_STD) + float(dcfg.DEPTH_MEAN)
                this_dim = dim[batch_id] * _ssd + _smu
            else:
                this_center_dis = center_dis[batch_id] * dcfg.MAX_DIS
                this_dim = dim[batch_id] * dcfg.MAX_SIZE
            this_rot = rot[batch_id]
            this_size2d = size2d[batch_id] if size2d is not None else None

            dec_kwargs = dict(self.dec_kwargs)
            if this_size2d is not None:
                dec_kwargs['size2d'] = this_size2d

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
