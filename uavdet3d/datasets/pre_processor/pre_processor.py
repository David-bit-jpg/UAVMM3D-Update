import inspect
import numpy as np
from uavdet3d.utils.object_encoder import all_object_encoders
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import torch
import cv2
from functools import partial
import copy


PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], dtype=np.float64)


def make_size2d_map(gt_box9d, intrinsic, distortion, new_im_width, new_im_hight, stride, im_num,
                    euler_seq='xyz', use_distortion=True):
    """目标 8 角点投影 2D 包围盒的 (log 宽, log 高)，写在中心格上；落格方式与 center_point_encoder 一致。

    取 log 是为了让损失是【相对误差】：Z = f·L_3d/s_2d，s_2d 的相对误差就是深度的相对误差，
    而目标跨度从十几像素到几百像素跨一个量级，用绝对 L1 会被大目标独占梯度。
    """
    from scipy.spatial.transform import Rotation as R
    from uavdet3d.utils.object_encoder_mav6d import project_points

    hm_w, hm_h = int(new_im_width) // int(stride), int(new_im_hight) // int(stride)
    boxes = np.asarray(gt_box9d, dtype=np.float64).reshape(-1, 9)
    out = []
    for im_id in range(int(im_num)):
        K = np.asarray(intrinsic[im_id], dtype=np.float64).reshape(3, 3)
        D = np.asarray(distortion[im_id], dtype=np.float64).reshape(-1)
        m = np.zeros((2, hm_h, hm_w), dtype=np.float32)
        for b in boxes:
            if not np.isfinite(b).all() or b[2] <= 1e-6:
                continue
            uv_c, _ = project_points(b[None, :3], K, D, use_distortion)
            wi, hi = int(uv_c[0, 0] / stride), int(uv_c[0, 1] / stride)
            if not (0 <= hi < hm_h and 0 <= wi < hm_w):
                continue
            corners = (PROTO8 * b[3:6]) @ R.from_euler(euler_seq, b[6:9]).as_matrix().T + b[:3]
            if (corners[:, 2] <= 1e-6).any():
                continue
            uv, _ = project_points(corners, K, D, use_distortion)
            w2, h2 = float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))
            if w2 < 1e-3 or h2 < 1e-3:
                continue
            m[0, hi, wi] = np.log(w2)
            m[1, hi, wi] = np.log(h2)
        out.append(m)
    return np.array(out)


def encoder_geometry_kwargs(dataset_cfg, fn, fn_name):
    """编码器 / 解码器共用的几何参数，只从 DATA_CONFIG 读一处，保证两边一致。

    ROT_REPR    旋转回归表示 euler6 / r6d
    EULER_SEQ   存储角的欧拉顺序（显式传，不读模块全局 —— spawn worker 里全局是 'zyx'）
    DEPTH_MODE  metric = 学米制 Z；virtual = 学 Z*f_ref/f（相机自适应，见 camera_geometry.py）
    DEPTH_F_REF 虚拟深度的参考焦距（网络输入像素）
    ROT_FRAME   ego = 学相机系旋转；allo = 学视线相对旋转（相机自适应）
    缺省值与历史行为一致；编码器不支持某项而配置要求非缺省值时直接报错，不静默忽略。
    """
    params = inspect.signature(fn).parameters
    wanted = {
        'rot_repr': (dataset_cfg.get('ROT_REPR', 'euler6'), 'euler6'),
        'euler_seq': (dataset_cfg.get('EULER_SEQ', None), None),
        'depth_mode': (dataset_cfg.get('DEPTH_MODE', 'metric'), 'metric'),
        'f_ref': (dataset_cfg.get('DEPTH_F_REF', None), None),
        'rot_frame': (dataset_cfg.get('ROT_FRAME', 'ego'), 'ego'),
    }
    kwargs = {}
    for k, (v, default) in wanted.items():
        if k in params:
            kwargs[k] = v
        elif v != default:
            raise ValueError('编码器/解码器 %s 不支持 %s=%s' % (fn_name, k, v))
    return kwargs


class DataPreProcessor():
    def __init__(self, dataset_cfg, training):

        self.dataset_cfg, self.training = dataset_cfg, training

        processor_configs = self.dataset_cfg.DATA_PRE_PROCESSOR

        self.data_processor_queue = []

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)

        obj_num = self.dataset_cfg.OBJ_NUM

        im_num = self.dataset_cfg.IM_NUM

        center_rad = self.dataset_cfg.CENTER_RAD

        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        intrinsic = data_dict['intrinsic']  # 1,3,3
        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']

        if 'gt_box9d' not in data_dict:
            return data_dict

        gt_box9d = data_dict['gt_box9d']  # N,9

        raw_im_width, raw_im_hight = data_dict['raw_im_size'][0], data_dict['raw_im_size'][1]

        new_im_width, new_im_hight = data_dict['new_im_size'][0], data_dict['new_im_size'][1]

        stride = data_dict['stride']

        # N, 4, 2

        gt_heat_map = []

        res_x = []
        res_y = []

        all_pts2d = []

        for im_id in range(im_num):
            all_ob_heat_map = []

            all_ob_res_x = []
            all_ob_res_y = []

            pts3d, pts2d = key_point_encoder(gt_box9d,
                                             encode_corner=encode_corner,
                                             intrinsic_mat=intrinsic[im_id],
                                             extrinsic_mat=extrinsic[im_id],
                                             distortion_matrix=distortion[im_id],
                                             offset=offset)

            key_pts_num = pts2d.shape[1]

            for ob_id in range(obj_num):
                ob_pts = pts2d[ob_id]
                this_heat_map = torch.zeros(key_pts_num, new_im_hight // stride, new_im_width // stride)
                this_residual_x = np.zeros(shape=(key_pts_num, new_im_hight // stride, new_im_width // stride))
                this_residual_y = np.zeros(shape=(key_pts_num, new_im_hight // stride, new_im_width // stride))

                for k_i, center in enumerate(ob_pts):
                    center[0] *= (new_im_width / raw_im_width / stride)
                    center[1] *= (new_im_hight / raw_im_hight / stride)

                    this_heat_map[k_i] = draw_gaussian_to_heatmap(this_heat_map[k_i], center, center_rad)
                    this_residual_x[k_i], this_residual_y[k_i] = draw_res_to_heatmap(this_residual_x[k_i],
                                                                                     this_residual_y[k_i], center)

                all_ob_heat_map.append(this_heat_map)

                all_ob_res_x.append(this_residual_x)
                all_ob_res_y.append(this_residual_y)

            all_ob_heat_map = torch.stack(all_ob_heat_map)
            all_ob_res_x = np.stack(all_ob_res_x)
            all_ob_res_y = np.stack(all_ob_res_y)

            gt_heat_map.append(all_ob_heat_map.cpu().numpy())
            res_x.append(all_ob_res_x)
            res_y.append(all_ob_res_y)
            all_pts2d.append(pts2d)

        gt_heatmap = np.array(gt_heat_map)  # 1,1,4,W,H
        gt_res_x = np.array(res_x)  # 1,1,4,W,H
        gt_res_y = np.array(res_y)  # 1,1,4,W,H
        all_pts2d = np.array(all_pts2d)  # 1,1,4,2
        # print(gt_res_x.max())
        # #
        # cv2.imwrite('im.png', data_dict['image'][0].transpose(1,2,0)*255)
        # cv2.imwrite('center.png', gt_res_x[0,0,0:3,:,:].transpose(1,2,0)*255)
        # input()

        data_dict['gt_heatmap'] = gt_heatmap
        data_dict['gt_res_x'] = gt_res_x
        data_dict['gt_res_y'] = gt_res_y
        data_dict['gt_pts2d'] = all_pts2d.reshape(im_num * obj_num, -1, 2)

        return data_dict

    def map_name_to_index(self, gt_name, class_name_config):

        return np.array([[class_name_config.index(x) for x in gt_name]])

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        # HM_CLASS_NAMES：热力图用的类别表，默认就是 CLASS_NAMES。设成 ['drone'] = 训练不分机型
        # （MAV6D 的 CLASS_NAMES 同时是数据目录名，不能直接改）
        class_name_config = self.dataset_cfg.get('HM_CLASS_NAMES', self.dataset_cfg.CLASS_NAMES)
        im_num = self.dataset_cfg.IM_NUM

        offset = config.OFFSET
        center_point_encoder = all_object_encoders[config.ENCODER]

        # 旋转表示：'euler6' = (cos,sin)x3（缺省，与历史行为一致）；
        # 'r6d' = 旋转矩阵前两列的 6D 连续表示。两者都占 6 个通道，网络结构不变。
        # 换 r6d 的理由见 uavdet3d/utils/rotation_repr.py 的说明与自检。
        rot_repr = self.dataset_cfg.get('ROT_REPR', 'euler6')
        enc_kwargs = encoder_geometry_kwargs(self.dataset_cfg, center_point_encoder, config.ENCODER)

        if 'gt_box9d' not in data_dict:
            return data_dict

        gt_box9d = data_dict['gt_box9d']  # N,9
        intrinsic = data_dict['intrinsic']  # 1, 3,3
        extrinsic = data_dict['extrinsic']  # 1, 4,4
        distortion = data_dict['distortion']  # ,1 5
        stride = data_dict['stride']

        gt_name = data_dict['gt_name']  # 1,

        cls_index = self.map_name_to_index(gt_name, class_name_config).reshape(-1, 1)  # N,1

        gt_box9d_with_cls = np.concatenate([gt_box9d, cls_index], -1)  # N, 10

        raw_im_size = data_dict['raw_im_size']  # 2,
        new_im_size = data_dict['new_im_size']  # 2,

        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = center_point_encoder(gt_box9d_with_cls,
                                                                                   intrinsic,
                                                                                   extrinsic,
                                                                                   distortion,
                                                                                   new_im_size[0],
                                                                                   new_im_size[1],
                                                                                   raw_im_size[0],
                                                                                   raw_im_size[1],
                                                                                   stride,
                                                                                   im_num,
                                                                                   class_name_config,
                                                                                   center_rad,
                                                                                   **enc_kwargs)
        # # 1, Class, W,H
        # # 1, 2, W,H
        # # 1, 1, W,H
        # 'hm': {'out_channels': 1},
        # 'center_res': {'out_channels': 2},
        # 'center_dis': {'out_channels': 1},
        # 'dim': {'out_channels': 3},
        # 'rot': {'out_channels': 6},

        data_dict['hm'] = gt_hm
        data_dict['center_res'] = gt_center_res

        # 前景掩码另出一张（有目标的格子为 1）：标准化后 center_dis 会有负数，
        # 不能再拿 `center_dis > 0` 当前景指示
        data_dict['fg_mask'] = (np.abs(gt_center_dis) > 0).astype(np.float32)

        # TARGET_NORM（2026-09-16）：
        #   'maxdis'  （缺省，逐位等同历史）真值 = 量 / MAX_DIS、量 / MAX_SIZE，两个手填常数
        #   'standard' 真值 = (量 − mu) / sigma，mu/sigma 由 tools/calc_target_stats.py 从训练集自身算
        # 为什么要换：回归头输出层没有 bias（见 center_head.py 的说明），整个预测值都挂在特征幅值上，
        # 跨域一缩放就整体缩放。标准化后「典型深度/典型尺寸」这个大头在解码时才加回去，
        # 永远不经过特征通路，域漂移乘不掉它。而且 mu/sigma 只来自源域，换任何部署数据都不用重调。
        if str(self.dataset_cfg.get('TARGET_NORM', 'maxdis')) == 'standard':
            dmu = float(self.dataset_cfg.DEPTH_MEAN)
            dsd = max(float(self.dataset_cfg.DEPTH_STD), 1e-6)
            smu = np.asarray(self.dataset_cfg.SIZE_MEAN, np.float32).reshape(1, 3, 1, 1)
            ssd = np.maximum(np.asarray(self.dataset_cfg.SIZE_STD, np.float32).reshape(1, 3, 1, 1), 1e-6)
            m = data_dict['fg_mask']
            data_dict['center_dis'] = ((gt_center_dis - dmu) / dsd * m).astype(np.float32)
            data_dict['dim'] = ((gt_dim - smu) / ssd * m).astype(np.float32)
        else:
            data_dict['center_dis'] = gt_center_dis / self.dataset_cfg.MAX_DIS
            data_dict['dim'] = gt_dim / self.dataset_cfg.MAX_SIZE
        data_dict['rot'] = gt_rot

        # ---- size2d：目标 8 角点投影 2D 包围盒的 log(宽), log(高)，单位 = 网络输入像素 ----
        # 为什么（2026-09-16）：深度头 center_dis 是自由回归的标量，和 dim/rot 之间没有任何约束，
        # 跨域时它自己退回训练先验 -> 预测出「自己说 0.53 m、自己说 7.5 m」这种投影回去盖不住目标的框。
        # 有了 2D 跨度就能按 Z = f·L_3d/s_2d 把深度解出来，误差只由【可见量】决定。
        # 实测上界（用 GT 的 2D 跨度代替这个头）：P1 在 MAV6D 上位置中位 4.601 -> 1.012 m。
        # 缺省关（MAKE_SIZE2D 未设时不产出这个键），行为与 2026-09-16 之前逐位一致。
        if self.dataset_cfg.get('MAKE_SIZE2D', False):
            data_dict['size2d'] = make_size2d_map(
                gt_box9d, intrinsic, distortion, new_im_size[0], new_im_size[1], stride, im_num,
                euler_seq=self.dataset_cfg.get('EULER_SEQ', 'xyz'),
                use_distortion=enc_kwargs.get('use_distortion', True))

        return data_dict

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        data_dict['image'] /= 255

        # 可选：按本域统计量归一化 rgb 三通道（BGR 顺序）。源域与目标域亮度差近一倍，
        # 只除 255 的话两域第一层卷积看到的分布相差很多。
        nm, ns = self.dataset_cfg.get('NORM_MEAN', None), self.dataset_cfg.get('NORM_STD', None)
        if nm and ns:
            img = data_dict['image']                     # (K, C, H, W) 或 (C, H, W)
            mean = np.asarray(nm, dtype=np.float32).reshape(-1, 1, 1)
            std = np.asarray(ns, dtype=np.float32).reshape(-1, 1, 1)
            if img.ndim == 4:
                img[:, :3] = (img[:, :3] - mean[None]) / std[None]
            else:
                img[:3] = (img[:3] - mean) / std
            data_dict['image'] = img

        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        gt_box9d = data_dict['gt_box9d']  # N,9

        raw_im_width, raw_im_hight = data_dict['raw_im_size'][0], data_dict['raw_im_size'][1]

        intrinsic = data_dict['intrinsic']  # 1,3,3

        im_num = self.dataset_cfg.IM_NUM

        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']
        gt_name = data_dict['gt_name']
        gt_diff = data_dict['gt_diff']

        valid_gt = []
        valid_name = []
        valid_gt_diff = []

        for i in range(im_num):

            if len(gt_box9d) > 0:
                gt_pts = copy.deepcopy(gt_box9d[:, 0:3])

                corners_2d, _ = cv2.projectPoints(gt_pts, extrinsic[i, :3, :3], extrinsic[i, :3, 3], intrinsic[i],
                                                  distortion[i])
                corners_2d = corners_2d.reshape(-1, 2).astype(int)
                mask_x_low = corners_2d[:, 0] > 0
                mask_x_high = corners_2d[:, 0] < raw_im_width
                mask_y_low = corners_2d[:, 1] > 0
                mask_y_high = corners_2d[:, 1] < raw_im_hight

                mask = mask_x_low * mask_x_high * mask_y_low * mask_y_high

                valid_gt.append(gt_box9d[mask])
                valid_name.append(gt_name[mask])
                valid_gt_diff.append(gt_diff[mask])

        if len(valid_gt) > 0:
            valid_gt = np.concatenate(valid_gt)
            valid_name = np.concatenate(valid_name)
            valid_gt_diff = np.concatenate(valid_gt_diff)
        else:
            valid_gt = np.empty(shape=(0, 9))
            valid_name = np.empty(shape=(0))
            valid_gt_diff = np.empty(shape=(0))

        data_dict['gt_box9d'] = valid_gt
        data_dict['gt_name'] = valid_name
        data_dict['gt_diff'] = valid_gt_diff

        return data_dict

    def __call__(self, data_dict):

        for func in self.data_processor_queue:
            data_dict = func(data_dict)

        return data_dict
