from uavdet3d.datasets import DatasetTemplate
from .mav6d_utils import *
from uavdet3d.datasets.mav6d.eval import eval_6d_pose
import numpy as np
import os
import torch
import pandas as pd
import cv2

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')  
except Exception:
    # 无界面环境 fallback 到 Agg（服务器上 tkAgg 缺 tkinter 时抛的不一定是 ImportError）
    matplotlib.use('Agg')

class MAV6D_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(MAV6D_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg=dataset_cfg
        self.root_path=root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training=training
        self.logger=logger

        self.im_path_name = 'JPEGImages'
        # 自训练（tools/mav6d_pseudo_label.py）用：标签与 split 目录可换成 labels_<suffix> / split_<suffix>，缺省不变
        self.label_path_name = dataset_cfg.get('LABEL_DIR', 'labels')
        self.split_dir_name = dataset_cfg.get('SPLIT_DIR', 'split')

        self.intrinsic = np.array([[1979.4, 0.3984, 976.8189, ],
                              [0.0, 1979.1, 533.9717, ],
                              [0.0, 0.0, 1.0, ], ])
        self.distortion_matrix = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
        self.extrinsic = np.eye(4)

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]

        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]

        self.im_num = self.dataset_cfg.IM_NUM

        self.obj_size = np.array(self.dataset_cfg.OB_SIZE)

        self.center_rad = self.dataset_cfg.CENTER_RAD

        self.stride = self.dataset_cfg.STRIDE

        self.class_name_dict = self.dataset_cfg.CLASS_NAMES


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        all_splits = []
        self.sample_scene_list = []

        for cls in self.class_name_dict:

             split_dir = os.path.join( self.root_path, cls, self.split_dir_name, self.split + '.txt')
             all_splits.append(split_dir)

             self.sample_scene_list += [[cls]+x.strip().split('/')[-3:] for x in open(split_dir).readlines()]

        # 在线增广（只在训练时；仓库原有的 DATA_AUGMENTOR 从来没有被数据集调用过）。
        # MAV6D 训练集 15202 帧只来自 77 个序列，5% 预算抽 761 帧时同序列内高度相关，
        # 实测 rot 训练损失能压到 0.01 而测试角度中位仍有 31°，是典型的过拟合，所以这里补上增广。
        self.aug = dict(dataset_cfg.get('AUG', {}) or {}) if training else {}

        self.infos = []
        self.include_MAV6D_data(self.mode)

    def _augment(self, img, boxes9d, K, D):
        """几何增广同步改 内参 / 畸变 / 相机系 3D 框；光度增广只动像素。

        img: (h, w, 3) float32 0-255，已经 resize 到 IM_RESIZE；K/D 是【原始分辨率】下的，
        编码器按 new/raw 的比例缩放投影结果，所以这里改 K 时要换算回原始像素。

        水平翻转：像素 u -> W-1-u 等价于相机系 x -> -x。
          - K: cx -> (raw_w - 1) - cx
          - 畸变: 切向项 p2 = D[3] 变号（x_dist 里 p2*(r2+2xp^2) 项在 x->-x 下不变号，
            所以要翻 p2 才能保持等价；p1 项含 xp*yp 自动变号）。径向项对称，不变。
          - 框: x -> -x，R -> M R S（M = diag(-1,1,1) 相机系镜像，S = diag(1,-1,1) 无人机左右对称面；
            不能用 M R M，那会把机头机尾对调，见 camera_geometry.hflip_rotation）
        随机尺度：在 resize 后的图上缩放再裁/补回原尺寸，K 的 fx,fy,cx,cy 相应变换；
          目标中心出画就重试，重试不到就不缩放（MAV6D 每帧只有一个目标，不能丢）。
        """
        rng = np.random
        a = self.aug
        h, w = img.shape[:2]
        raw_w, raw_h = float(self.raw_im_width), float(self.raw_im_hight)
        K = K.copy()
        D = D.copy()
        boxes9d = np.array(boxes9d, dtype=np.float64, copy=True)

        def center_uv(K_, D_, box):
            xyz = box[None, :3]
            xp, yp = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
            r2 = xp * xp + yp * yp
            rad = 1.0 + D_[0] * r2 + D_[1] * r2 * r2 + D_[4] * r2 * r2 * r2
            xd = xp * rad + 2.0 * D_[2] * xp * yp + D_[3] * (r2 + 2.0 * xp * xp)
            yd = yp * rad + D_[2] * (r2 + 2.0 * yp * yp) + 2.0 * D_[3] * xp * yp
            return float(K_[0, 0] * xd + K_[0, 2]), float(K_[1, 1] * yd + K_[1, 2])

        if rng.rand() < float(a.get('hflip', 0.0)):
            img = img[:, ::-1].copy()
            K[0, 2] = (raw_w - 1.0) - K[0, 2]
            D[3] = -D[3]
            boxes9d[:, 0] *= -1.0
            M = np.diag([-1.0, 1.0, 1.0])
            S = np.diag([1.0, -1.0, 1.0])
            for i in range(len(boxes9d)):
                Rm = R.from_euler('xyz', boxes9d[i, 6:9]).as_matrix()
                boxes9d[i, 6:9] = R.from_matrix(M @ Rm @ S).as_euler('xyz')

        sr = a.get('scale', None)
        if sr:
            for _ in range(5):
                s = float(rng.uniform(sr[0], sr[1]))
                if abs(s - 1.0) < 1e-3:
                    break
                nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
                res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                if s >= 1.0:
                    oy, ox = rng.randint(0, nh - h + 1), rng.randint(0, nw - w + 1)
                    out = res[oy:oy + h, ox:ox + w]
                    dx, dy = -ox, -oy
                else:
                    out = np.zeros_like(img)
                    oy, ox = rng.randint(0, h - nh + 1), rng.randint(0, w - nw + 1)
                    out[oy:oy + nh, ox:ox + nw] = res
                    dx, dy = ox, oy
                K2 = K.copy()
                K2[0, 0] *= s; K2[1, 1] *= s
                K2[0, 2] = K2[0, 2] * s + dx * raw_w / w
                K2[1, 2] = K2[1, 2] * s + dy * raw_h / h
                u, v = center_uv(K2, D, boxes9d[0])
                mg = float(a.get('scale_margin', 24.0))       # 原始像素下的留边
                if mg <= u < raw_w - mg and mg <= v < raw_h - mg:
                    img, K = out, K2
                    break

        if a.get('photometric', False):
            gain = rng.uniform(0.75, 1.25)
            contrast = rng.uniform(0.8, 1.25)
            gamma = rng.uniform(0.8, 1.25)
            cgain = rng.uniform(0.92, 1.08, size=(1, 1, 3))
            m = img.mean()
            x = np.clip(((img - m) * contrast + m) * gain * cgain, 0, 255) / 255.0
            img = (np.power(x, gamma) * 255.0).astype(np.float32)
            if a.get('noise', 0.0) > 0:
                img = np.clip(img + rng.randn(*img.shape).astype(np.float32) * float(a['noise']) * 255.0, 0, 255)
        return img.astype(np.float32), boxes9d.astype(np.float32), K, D

    def set_split(self, split):
        super(MAV6D_Det_Dataset, self).__init__(dataset_cfg=self.dataset_cfg, training=self.training,
                                                 root_path=self.root_path, logger=self.logger)

        self.split = split
        all_splits = []
        self.sample_scene_list = []

        for cls in self.class_name_dict:
            split_dir = os.path.join(self.root_path, cls, self.split_dir_name, self.split + '.txt')
            all_splits.append(split_dir)

            self.sample_scene_list += [[cls]+x.strip().split('/')[-3:] for x in open(split_dir).readlines()]

        self.infos = []
        self.include_MAV6D_data(self.mode)

    def include_MAV6D_data(self, mode):

        self.logger.info('Loading MAV6D dataset')

        MAV6D_infos = []

        root_path = str(self.root_path)

        for i in range(0, len(self.sample_scene_list), self.dataset_cfg.SAMPLED_INTERVAL[mode]):

            info_item = self.sample_scene_list[i]

            cls_name = info_item[0]

            scene_name = info_item[1]

            seq_name = info_item[2]

            frame_name = info_item[3]

            each_im_path = os.path.join(root_path, cls_name, self.im_path_name, scene_name, seq_name, frame_name)
            each_label_path = os.path.join(root_path, cls_name, self.label_path_name, scene_name, seq_name,
                                           os.path.splitext(frame_name)[0] + '.txt')

            data_info = {'im_path': each_im_path, 'label_path': each_label_path, 'cls_name': cls_name, 'scene_id': scene_name,
                         'seq_id': seq_name, 'frame_id': frame_name}

            MAV6D_infos.append(data_info)

        self.infos = MAV6D_infos


    def __len__(self):

        return len(self.infos)

    def __getitem__(self, item):

        each_info = self.infos[item]

        im_path = each_info['im_path']
        cls_name = each_info['cls_name']
        label_path = each_info['label_path']
        scene_id = each_info['scene_id']
        seq_id = each_info['seq_id']
        frame_id = each_info['frame_id']

        image = cv2.imread(im_path).astype(np.float32)
        resized_img = cv2.resize(
            image,
            (self.new_im_width, self.new_im_hight),  # 目标尺寸 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小推荐使用区域插值
        )

        rotation_matrix, translation_matrix = read_truth_Rt(label_path)
        rotation_matrix = np.array(rotation_matrix.reshape(3, 3), dtype=np.float32)
        translation_matrix = np.array(translation_matrix.reshape(3, ), dtype=np.float32)
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=False)
        box9d = np.concatenate([translation_matrix, self.obj_size[0], euler_angles], 0)

        boxes9d = np.array([box9d])

        K = np.array(self.intrinsic, dtype=np.float64)
        D = np.array(self.distortion_matrix, dtype=np.float64)
        if self.training and self.aug:
            resized_img, boxes9d, K, D = self._augment(resized_img, boxes9d, K, D)

        resized_img = resized_img.transpose(2,0,1)
        C,W,H = resized_img.shape
        resized_img = resized_img.reshape(1,C,W,H)

        data_dict = {}


        data_dict['intrinsic'] = np.array([K])
        data_dict['extrinsic'] = np.array([self.extrinsic])
        data_dict['distortion'] = np.array([D])
        data_dict['raw_im_size'] = np.array([self.raw_im_width, self.raw_im_hight])
        data_dict['new_im_size'] = np.array([self.new_im_width, self.new_im_hight])
        data_dict['obj_size'] = np.array(self.dataset_cfg.OB_SIZE)
        data_dict['scene_id'] = scene_id
        data_dict['seq_id'] = seq_id
        data_dict['frame_id'] = frame_id
        data_dict['stride'] = self.stride
        data_dict['image'] = resized_img
        data_dict['gt_box9d'] = boxes9d
        # 不分机型训练时（配置 HM_CLASS_NAMES: ['drone']）标签统一成一类；数据仍按机型目录读
        hm_names = self.dataset_cfg.get('HM_CLASS_NAMES', None)
        data_dict['gt_name'] = np.array([hm_names[0] if hm_names else cls_name]*len(boxes9d))

        data_dict = self.data_pre_processor(data_dict)


        return data_dict


    def name_from_code(self, name_indices):

        names = [self.class_name_dict[int(x)] for x in name_indices]

        return names


    def generate_prediction_dicts(self, batch_dict, output_path):

        batch_size = batch_dict['batch_size']

        annos = []

        for batch_id in range(batch_size):

            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id] # 1, 1, 4, W, H

            scene_id = batch_dict['scene_id'][batch_id]
            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]

            intrinsic = batch_dict['intrinsic'][batch_id] # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id] # 4, 4
            distortion = batch_dict['distortion'][batch_id] # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id] # 2,
            new_im_size = batch_dict['new_im_size'][batch_id] # 2,
            obj_size = batch_dict['obj_size'][batch_id] # 3,

            # key_points_2d = batch_dict['key_points_2d'][batch_id]
            confidence = batch_dict['confidence'][batch_id]
            im_path = os.path.join(self.root_path, self.im_path_name, scene_id, seq_id, frame_id)

            pred_name = self.name_from_code(pred_boxes9d[:,-1])

            frame_dict = {'scene_id': scene_id,
                          'seq_id': seq_id,
                          'frame_id': frame_id,
                          'intrinsic': intrinsic,
                          'extrinsic': extrinsic,
                          'distortion': distortion,
                          'raw_im_size': raw_im_size,
                          'obj_size': obj_size,
                          #'key_points_2d': key_points_2d,
                          'confidence': confidence,
                          'pred_box9d': pred_boxes9d,
                          'pred_name': pred_name,
                          'im_path': im_path
                          }

            if 'gt_box9d' in batch_dict:
                gt_box9d = batch_dict['gt_box9d'][batch_id].cpu().numpy() # 1, 9
                #gt_pts2d = batch_dict['gt_pts2d'][batch_id].cpu().numpy() # 1, 2
                frame_dict['gt_box9d'] = gt_box9d
                gt_name = batch_dict['gt_name'][batch_id]
                frame_dict['gt_name'] = gt_name

                # frame_dict['gt_pts2d']=gt_pts2d

            annos.append(frame_dict)

            # image = cv2.imread(im_path)
            # #
            # # image = draw_2d_points_on_image(key_points_2d,
            # #                                 image,
            # #                                 color=(0,255,0),
            # #                                 radius=8)
            #
            # image = draw_box9d_on_image(pred_boxes9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(255, 0, 0),
            #                             intrinsic_mat=intrinsic[0],
            #                             extrinsic_mat=extrinsic[0],
            #                             distortion_matrix=distortion[0],
            #                             offset=np.array([0., 0., 0.]))
            #
            #
            #
            # image = draw_box9d_on_image(gt_box9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(0, 0, 255),
            #                             intrinsic_mat=intrinsic[0],
            #                             extrinsic_mat=extrinsic[0],
            #                             distortion_matrix=distortion[0],
            #                             offset=np.array([0.2, 0.2, 0.]))
            #
            # cv2.imshow('im', image)
            # cv2.waitKey(1)

        return annos

    def evaluation_by_name(self, annos, metric_root_path, cls_name):

        orin_error, pos_error = eval_6d_pose(annos, max_dis=self.dataset_cfg.MAX_DIS)

        plt.scatter(np.arange(0,len(pos_error)), pos_error)
        plt.savefig(os.path.join(metric_root_path, cls_name+'_pos_error.png'))

        print(cls_name+'_orin_error： ', orin_error.mean())
        print(cls_name+'_pose_error: ', pos_error.mean())

        print(cls_name+'_orin_error_median: ', np.median(orin_error))
        print(cls_name+'_pose_error_median: ', np.median(pos_error))

        print(cls_name+'_orin_error_min： ', orin_error.min())
        print(cls_name+'_pose_error_min: ', pos_error.min())

        print(cls_name+'_orin_error_max： ', orin_error.max())
        print(cls_name+'_pose_error_max: ', pos_error.max())

        return_str = '\n'+cls_name+': \n orin_error： ' + str(orin_error.mean()) + '\n'\
                     + 'pose_error: ' + str(pos_error.mean()) + '\n'\
                     + 'orin_error_median: ' + str(np.median(orin_error)) +'\n' \
                     + 'pos_error_median: ' + str(np.median(pos_error)) + '\n' \
                     + 'orin_error_min: ' + str(orin_error.min()) + '\n' \
                     + 'pos_error_min: ' + str(pos_error.min()) + '\n' \
                     + 'orin_error_max: ' + str(orin_error.max()) + '\n' \
                     + 'pose_error_max: ' + str(pos_error.max()) + '\n'

        orin_error_out_path = os.path.join(str(metric_root_path),cls_name+'_orin_error.csv')
        orin_error_pd = pd.DataFrame({cls_name+'_orin_error': orin_error.tolist()})
        orin_error_pd.to_csv(orin_error_out_path)

        pos_error_out_path = os.path.join(str(metric_root_path),cls_name+'_pos_error.csv')
        pos_error_pd = pd.DataFrame({cls_name+'_pos_error': pos_error.tolist()})
        pos_error_pd.to_csv(pos_error_out_path)

        mean_error_out_path = os.path.join(str(metric_root_path), cls_name+'_mean_error.csv')
        mean_error_pd = pd.DataFrame({ cls_name+'_orin_error': [orin_error.mean()], cls_name+'_pose_error': [pos_error.mean()] })
        mean_error_pd.to_csv(mean_error_out_path)

        return return_str


    def evaluation(self, annos, metric_root_path):

        all_names = self.class_name_dict

        final_str = ''

        for cls_n in all_names:

            this_annos_pred = copy.deepcopy(annos)

            new_annos = []

            for each_anno in this_annos_pred:

                new_anno = {}
                gt_name = each_anno['gt_name']
                gt_box = each_anno['gt_box9d']
                pre_box = each_anno['pred_box9d']

                # gt 用名字筛；预测不能用同一个掩码 —— name_mask 的长度是 gt 的
                # 个数，而 pre_box 的行数是检测出的目标数，两者一般不等，
                # pre_box[name_mask] 会直接 IndexError。预测的类别在最后一列。
                name_mask = gt_name == cls_n
                new_anno['gt_box9d'] = gt_box[name_mask]

                cls_idx = list(all_names).index(cls_n)
                if len(pre_box) > 0:
                    pred_mask = np.round(np.asarray(pre_box)[:, -1]).astype(int) == cls_idx
                    new_anno['pred_box9d'] = np.asarray(pre_box)[pred_mask]
                else:
                    new_anno['pred_box9d'] = np.asarray(pre_box)

                if new_anno['gt_box9d'].shape[0]>=1:

                    new_annos.append(new_anno)

            if len(new_annos)>=2:
                this_str = self.evaluation_by_name(new_annos, metric_root_path, cls_name = cls_n)

                final_str+=this_str

        this_str =  self.evaluation_by_name(annos, metric_root_path, cls_name='all')

        final_str += this_str

        return final_str



