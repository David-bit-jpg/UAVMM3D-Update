import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from uavdet3d.datasets.mav6d.mav6d_utils import draw_box9d_on_image

from uavdet3d.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network, model_fn_decorator
from uavdet3d.utils import common_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model
from tools.test import repeat_eval_ckpt
import torch.utils.data as torch_data
from collections import defaultdict
from pathlib import Path
import numpy as np
from uavdet3d.model import load_data_to_gpu
import cv2


from uavdet3d.datasets.pre_processor.pre_processor import DataPreProcessor

class Dataset(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=False, root_path=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.obj_num = self.dataset_cfg.OBJ_NUM
        self.im_num = self.dataset_cfg.IM_NUM
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError


class Open3DUAVDet():
    def __init__(self, config='cfgs/models/uavdet_3d/laa3d/centerdet.yaml',
                 mode = 'pose_estimation',
                 model_path=r"..\output\models\uavdet_3d\laa3d\centerdet_30\default\ckpt/checkpoint_epoch_20.pth"):

        self.config = config
        self.model_path = model_path
        self.mode = mode

        self.extrinsic_mat = None
        self.intrinsic_mat = None
        self.distortion_mat = None
        cfg_from_yaml_file(config, cfg)

        self.cfg = cfg

        data_temp = Dataset(cfg.DATA_CONFIG)

        self.model = build_network(model_cfg=cfg.MODEL, dataset=data_temp)
        self.model.load_params_from_file(model_path,'cpu')
        self.model.cuda()
        self.model.eval()

        self.data_pro = DataPreProcessor(cfg.DATA_CONFIG, training=False)

    def get_camera_intrinsic_matrix(self, image_width=1280, image_height=720, fov=90):
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = fx  # Assuming square pixels
        cx = image_width / 2.0
        cy = image_height / 2.0
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic_matrix

    def __call__(self, image, extrinsic_mat=None, in_trinsic_mat=None, distortion_mat=None):

        img_height, img_width, c = image.shape

        if in_trinsic_mat is None:
            in_trinsic_mat = self.get_camera_intrinsic_matrix()

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4)

        if distortion_mat is None:
            distortion_mat = np.zeros(5, dtype=np.float32)


        resized_img = cv2.resize(
            image,
            (self.cfg.DATA_CONFIG.IM_RESIZE[0], self.cfg.DATA_CONFIG.IM_RESIZE[1]),  # 目标尺寸 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小推荐使用区域插值
        )

        resized_img = resized_img.transpose(2,0,1)
        C,W,H = resized_img.shape
        resized_img = resized_img.reshape(1,C,W,H).astype(np.float32)

        self.extrinsic_mat = extrinsic_mat
        self.intrinsic_mat = in_trinsic_mat
        self.distortion_mat = distortion_mat

        input_data = {'intrinsic': np.array([[in_trinsic_mat]]),
                      'extrinsic': np.array([[extrinsic_mat]]),
                      'distortion': np.array([[distortion_mat]]),
                      'raw_im_size': np.array([[img_width, img_height]]),
                      'new_im_size': np.array([[self.cfg.DATA_CONFIG.IM_RESIZE[0], self.cfg.DATA_CONFIG.IM_RESIZE[1]]]),
                      'stride': np.array([self.cfg.DATA_CONFIG.STRIDE]),
                      'image': np.array([resized_img]),
                      'batch_size': 1}

        input_data = self.data_pro(input_data)

        self.model.eval()
        load_data_to_gpu(input_data)


        batch_dict = self.model(input_data)

        if self.mode == 'pose_estimation':

            pred_boxes9d = batch_dict['pred_boxes9d'][0]
            confidence = batch_dict['confidence'][0]

            pred_dict = {'pred_box_9dof': pred_boxes9d,
                         'pred_confidence': confidence}

        elif self.mode == '3d_detection':
            pred_dict = {'pred_box_9dof': None,
                         'pred_name': None,
                         'pred_confidence': None}

        elif self.mode == '2d_detection':
            pred_dict = {'pred_box_2d': None,
                         'pred_name': None,
                         'pred_confidence': None}

        else:
            raise NotImplementedError

        return pred_dict

    def plot_on_image(self, image, boxes9d, extrinsic_mat=None, in_trinsic_mat=None, distortion_mat=None):

        img_height, img_width, c = image.shape

        if in_trinsic_mat is None:
            in_trinsic_mat = self.intrinsic_mat

        if extrinsic_mat is None:
            extrinsic_mat = self.extrinsic_mat

        if distortion_mat is None:
            distortion_mat = self.distortion_mat

        if in_trinsic_mat is None:
            in_trinsic_mat = np.array([[img_width, 0, img_width / 2],
                                      [0, img_height, img_height / 2],
                                      [0, 0, 1]], dtype=np.float32)

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4)

        if distortion_mat is None:
            distortion_mat = np.zeros(5, dtype=np.float32)

        new_im = draw_box9d_on_image(boxes9d, image, img_width=img_width, img_height=img_height, color=(255, 0, 0),
                                intrinsic_mat=in_trinsic_mat, extrinsic_mat=extrinsic_mat, distortion_matrix=distortion_mat,
                                offset=np.array([0, 0, 0]))

        return new_im



