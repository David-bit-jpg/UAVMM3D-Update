from collections import defaultdict
from uavdet3d.datasets import DatasetTemplate
import numpy as np
import os
import torch
import pandas as pd
import cv2
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import copy
import random
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image

try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')
except ImportError:
    # 无界面环境 fallback 到 Agg
    matplotlib.use('Agg')


class LAAM6D_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(LAAM6D_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path,
                                                 logger=logger)

        self.dataset_cfg = dataset_cfg
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training = training
        self.logger = logger
        self.im_num = dataset_cfg.IM_NUM
        self.lidar_offset = dataset_cfg.LIDAR_OFFSET
        self.radar_offset = dataset_cfg.RADAR_OFFSET

        self.sequence_frames = {}
        self.modalities = ['rgb', 'ir', 'dvs']
        self.intrinsics = {}
        self.extrinsics = {}
        self.distortions = {}
        self.default_distortion = np.zeros(5, dtype=np.float32)

        self.im_path_names = ['images_rgb', 'images_ir', 'images_dvs']
        self.label_path_names = ['boxes_rgb', 'boxes_ir', 'boxes_dvs']

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]
        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]
        self.obj_size = np.array(self.dataset_cfg.OB_SIZE)
        self.stride = self.dataset_cfg.STRIDE

        self.seq_list = self._build_seq_list_all_maps(split_ratio=0.9)

        self.sample_scene_list = []
        self.sorted_namelist = self.dataset_cfg.CLASS_NAMES

        for seq_name in self.seq_list:
            seq_path = os.path.join(str(self.root_path), seq_name, 'images_rgb')
            if not os.path.exists(seq_path):
                self.logger.warning(f"No image path: {seq_path}")
                continue

            all_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame in all_frames:
                self.sample_scene_list.append([seq_name, frame])

        self.infos = []
        self.include_CARLA_data(self.mode)

    def include_CARLA_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        CARLA_infos = []

        from collections import defaultdict
        seq_groups = defaultdict(list)  # {seq_name: [(frame_name1), (frame_name2), ...], ...}
        for seq_name, frame_name in self.sample_scene_list:
            seq_groups[seq_name].append(frame_name)

        for seq_name, frame_list in seq_groups.items():
            total_frames_in_seq = len(frame_list)

            valid_end = total_frames_in_seq - max(self.lidar_offset, self.radar_offset)
            if valid_end <= 0:
                self.logger.warning(
                    f"序列 {seq_name} 帧数量不足（共{total_frames_in_seq}帧，偏移量{max(self.lidar_offset, self.radar_offset)}），跳过该序列")
                continue

            base_path = os.path.join(self.root_path, seq_name)
            for i in range(0, valid_end, self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                curr_frame = frame_list[i]

                lidar_frame = frame_list[i + self.lidar_offset]
                radar_frame = frame_list[i + self.radar_offset]

                im_paths = {}
                for mode_name in self.modalities:
                    image_dir = f'images_{mode_name}'
                    image_path = os.path.join(base_path, image_dir, curr_frame)
                    if not os.path.exists(image_path):
                        self.logger.warning(f"序列 {seq_name} 未找到图像: {image_path}")
                    im_paths[mode_name] = image_path

                label_paths = {}
                for mode_name in self.modalities:
                    label_dir = f'boxes_{mode_name}'
                    label_path = os.path.join(base_path, label_dir, curr_frame.replace('.png', '.pkl'))
                    if not os.path.exists(label_path):
                        self.logger.warning(f"序列 {seq_name} 未找到标签: {label_path}")
                    label_paths[mode_name] = label_path

                lidar_path = os.path.join(base_path, 'lidar_1', lidar_frame.replace('.png', '.npy'))
                radar_path = os.path.join(base_path, 'radar_1', radar_frame.replace('.png', '.npy'))

                im_info_path = os.path.join(base_path, 'im_info.pkl')
                lidar_radar_info_path = os.path.join(base_path, 'lidar_radar_info.pkl')
                if not os.path.exists(lidar_radar_info_path):
                    self.logger.warning(f"序列 {seq_name} 未找到外参文件: {lidar_radar_info_path}")
                    lidar_extrinsic = np.eye(4)
                    radar_extrinsic = np.eye(4)
                else:
                    with open(lidar_radar_info_path, 'rb') as f:
                        lidar_radar_info = pickle.load(f)
                    lidar_extrinsic = np.array(lidar_radar_info['lidars'][0]['extrinsic'], dtype=np.float32)
                    radar_extrinsic = np.array(lidar_radar_info['radars'][0]['extrinsic'], dtype=np.float32)

                data_info = {
                    'im_paths': im_paths,
                    'label_paths': label_paths,
                    'lidar_path': lidar_path,
                    'radar_path': radar_path,
                    'im_info_path': im_info_path,
                    'lidar_extrinsic': lidar_extrinsic,
                    'radar_extrinsic': radar_extrinsic,
                    'seq_id': seq_name,
                    'frame_id': curr_frame,
                    'cls_name': self.sorted_namelist
                }

                CARLA_infos.append(data_info)

        self.infos = CARLA_infos

    def __len__(self):

        return len(self.infos)

    def __getitem__(self, item):
        each_info = self.infos[item]

        im_paths = each_info['im_paths']
        label_paths = each_info['label_paths']
        im_info_path = each_info['im_info_path']
        cls_name = each_info['cls_name']
        seq_id = each_info['seq_id']
        frame_id = each_info['frame_id']

        lidar_path = each_info.get('lidar_path', None)
        if lidar_path is not None and os.path.exists(lidar_path):
            lidar_data = np.load(lidar_path)  # (N, 5): x, y, z, intensity, tag
            assert lidar_data.shape[1] == 5, f"{lidar_data.shape} is not (N,5)"
        else:
            self.logger.warning(f"LiDAR file missing for frame: {frame_id}")
            lidar_data = np.empty((0, 5), dtype=np.float32)
        radar_path = each_info.get('radar_path', None)
        if radar_path is not None and os.path.exists(radar_path):
            radar_data = np.load(radar_path)  # (N, 5)
        else:
            self.logger.warning(f"Radar file missing for frame: {frame_id}")
            radar_data = np.empty((0, 5), dtype=np.float32)

        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)

        intrinsic = np.array(camera_info['rgb']['intrinsic'])
        extrinsic = np.array(camera_info['rgb']['extrinsic'])
        image_modal_stack = []

        box_file = label_paths['rgb']
        gt_boxes, gt_names = [], []

        center_world_cal = []
        if os.path.exists(box_file):
            with open(box_file, 'rb') as f:
                raw_data = pickle.load(f)

            for row in raw_data:
                if isinstance(row[0], str):
                    raw_name = row[0]
                    pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
                else:
                    raw_name = "Unknown"
                    pts = np.array(row, dtype=np.float32).reshape(8, 3)

                pts_world = self.convert_box_opencv_to_world(pts, extrinsic)

                center_world = (
                                       (pts_world[0] + pts_world[6]) +
                                       (pts_world[1] + pts_world[7]) +
                                       (pts_world[2] + pts_world[4]) +
                                       (pts_world[3] + pts_world[5])
                               ) / 8.0
                center_world = center_world.reshape(1, 3)
                center_world_cal.append(center_world)
                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in cls_name:
                    if cls in raw_name:
                        matched = cls
                        break
                gt_names.append(matched)
            assert len(gt_boxes) == len(gt_names), f"Boxes/names mismatch: {len(gt_boxes)} vs {len(gt_names)}"
            gt_boxes = np.array(gt_boxes, dtype=np.float32)  # (N, 9, 3)
            gt_names = np.array(gt_names)
            assert len(gt_boxes) == len(gt_names), f"Boxes/names mismatch: {len(gt_boxes)} vs {len(gt_names)}"
        else:
            self.logger.warning(f"No label file for RGB: {box_file}")
            gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            gt_names = np.empty((0,), dtype='<U32')

        distortion = self.default_distortion.copy()
        intrinsic_rgb = np.array(camera_info['rgb']['intrinsic'])
        extrinsic_rgb = np.array(camera_info['rgb']['extrinsic'])
        intrinsic_ir = np.array(camera_info['ir']['intrinsic'])
        extrinsic_ir = np.array(camera_info['ir']['extrinsic'])
        intrinsic_dvs = np.array(camera_info['dvs']['intrinsic'])
        extrinsic_dvs = np.array(camera_info['dvs']['extrinsic'])

        center_uv_rgb = self.xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_rgb,
                                       extrinsic_rgb, distortion)
        center_uv_ir = self.xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_ir,
                                      extrinsic_ir, distortion)
        center_uv_dvs = self.xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_dvs,
                                       extrinsic_dvs, distortion)

        rgb_image = cv2.imread(im_paths['rgb'], cv2.IMREAD_COLOR)
        ir_img = cv2.imread(im_paths['ir'], cv2.IMREAD_GRAYSCALE)
        dvs_img = cv2.imread(im_paths['dvs'], cv2.IMREAD_COLOR)

        registered = self.register_images_by_center(
            rgb_image, ir_img, dvs_img,
            center_uv_rgb, center_uv_ir, center_uv_dvs,
            intrinsic_rgb, intrinsic_ir, intrinsic_dvs
        )

        # 获取配准后的图像和内参
        aligned_rgb = registered['rgb']
        aligned_ir = registered['ir']
        aligned_dvs = registered['dvs']
        aligned_intrinsics_rgb = registered['intrinsics']['rgb']
        aligned_intrinsics_ir = registered['intrinsics']['ir']
        aligned_intrinsics_dvs = registered['intrinsics']['dvs']

        # 计算缩放比例
        # 原始配准后图像的尺寸
        orig_height, orig_width = aligned_rgb.shape[:2]
        # 目标尺寸
        target_width, target_height = self.new_im_width, self.new_im_hight

        # 计算x和y方向的缩放因子
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        # 缩放图像
        rgb_image = cv2.resize(aligned_rgb, (target_width, target_height))
        ir_img = cv2.resize(aligned_ir, (target_width, target_height))
        dvs_img = cv2.resize(aligned_dvs, (target_width, target_height))

        # 更新RGB内参
        resized_intrinsics_rgb = aligned_intrinsics_rgb.copy()
        resized_intrinsics_rgb[0, 0] *= scale_x  # fx
        resized_intrinsics_rgb[1, 1] *= scale_y  # fy
        resized_intrinsics_rgb[0, 2] *= scale_x  # cx
        resized_intrinsics_rgb[1, 2] *= scale_y  # cy

        # 更新IR内参
        resized_intrinsics_ir = aligned_intrinsics_ir.copy()
        resized_intrinsics_ir[0, 0] *= scale_x
        resized_intrinsics_ir[1, 1] *= scale_y
        resized_intrinsics_ir[0, 2] *= scale_x
        resized_intrinsics_ir[1, 2] *= scale_y

        # 更新DVS内参
        resized_intrinsics_dvs = aligned_intrinsics_dvs.copy()
        resized_intrinsics_dvs[0, 0] *= scale_x
        resized_intrinsics_dvs[1, 1] *= scale_y
        resized_intrinsics_dvs[0, 2] *= scale_x
        resized_intrinsics_dvs[1, 2] *= scale_y

        intrinsic = resized_intrinsics_rgb

        for mode in self.modalities:
            if mode == 'rgb':
                img = rgb_image
            elif mode == 'ir':
                img = ir_img
            elif mode == 'dvs':
                img = dvs_img
            else:
                img = cv2.imread(im_paths[mode], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.new_im_width, self.new_im_hight))

            img = img.astype(np.float32)
            if mode == 'ir':
                if len(img.shape) == 2:
                    img = img[np.newaxis, :, :]  # (1, H, W)
                    img = np.repeat(img, 3, axis=0)  # (3, H, W)
                else:
                    img = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)

            image_modal_stack.append(img)

        lidar_proj_info = self.project_lidar_and_get_uvz_rgb_tag(
            lidar_data,
            lidar_extrinsic=np.array(each_info['lidar_extrinsic']),
            camera_extrinsic=extrinsic,
            camera_intrinsic=intrinsic,
            image=rgb_image
        )  # shape (M, 10): [u, v, z, x_world, y_world, z_world, r, g, b, tag]

        world_coords_map = self.generate_world_coords_map(
            lidar_proj_info,
            image_shape=rgb_image.shape[:2]
        )
        world_coords_resized = cv2.resize(
            world_coords_map,
            (self.new_im_width, self.new_im_hight),
            interpolation=cv2.INTER_NEAREST
        )
        world_coords_tensor = world_coords_resized.transpose(2, 0, 1)  # (3, H, W)
        world_coords_tensor = world_coords_tensor.astype(np.float32)

        image_modal_stack.append(world_coords_tensor)

        radar_mask = self.radar_to_velocity_heatmap(
            radar_data=radar_data,
            radar_extrinsic=np.array(each_info['radar_extrinsic']),
            camera_extrinsic=extrinsic,
            camera_intrinsic=intrinsic,
            image_shape=rgb_image.shape[:2]
        )
        radar_mask_resized = cv2.resize(
            radar_mask[0],
            (self.new_im_width, self.new_im_hight),
            interpolation=cv2.INTER_NEAREST
        )[np.newaxis, :, :]

        radar_mask_tensor = np.repeat(radar_mask_resized, 3, axis=0)

        image_modal_stack.append(radar_mask_tensor)

        image = np.stack(image_modal_stack, axis=0)  # shape: (5, 3, H, W)

        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)

        # image_modal_stack2 = []
        # rgb_image = rgb_image.transpose(2, 0, 1)  # → (3, H, W)
        # image_modal_stack2.append(rgb_image)
        # image_modal_stack2.append(rgb_image)
        # image_modal_stack2.append(rgb_image)
        # image_modal_stack2.append(rgb_image)
        # image_modal_stack2.append(rgb_image)
        # image = np.stack(image_modal_stack2, axis=0)

        data_dict = {
            'image': image,  # (5, 3, H, W)
            'gt_boxes': gt_boxes,  # (N, 9, 3)
            'gt_names': gt_names,  # (N,)
            'intrinsic': intrinsic,  # (3, 3)
            'extrinsic': extrinsic,  # (4, 4)
            'distortion': distortion,  # (5,)
            'raw_im_size': np.array([self.raw_im_width, self.raw_im_hight]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.OB_SIZE),
            'seq_id': seq_id,
            'frame_id': frame_id,
            'stride': self.stride,
            'sorted_namelist': self.sorted_namelist
        }
        data_dict = self.data_pre_processor(data_dict)

        # ==========================================================================================================================
        # ==========================================================================================================================
        # ==========================================================================================================================
        # velocity_img = radar_mask_tensor[0, :, :]
        # if velocity_img.max() != velocity_img.min():
        #     velocity_img = (velocity_img - velocity_img.min()) / (velocity_img.max() - velocity_img.min()) * 255
        # else:
        #     velocity_img = np.zeros_like(velocity_img)
        # velocity_img = velocity_img.astype(np.uint8)
        # rgb_with_gt = self.draw_box9d_on_image_gt(
        #     gt_boxes,
        #     rgb_image.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # ir_with_gt = self.draw_box9d_on_image_gt(
        #     gt_boxes,
        #     ir_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # dvs_with_gt = self.draw_box9d_on_image_gt(
        #     gt_boxes,
        #     dvs_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # velicity_with_gt = self.draw_box9d_on_image_gt(
        #     gt_boxes,
        #     velocity_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # world_coords_resized_with_gt = self.draw_box9d_on_image_gt(
        #     gt_boxes,
        #     world_coords_resized.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # self.vis_save_dir = os.path.join(os.getcwd(), 'multimodal_visualizations_with_gt')
        # os.makedirs(self.vis_save_dir, exist_ok=True)
        # save_filename = f"{seq_id.replace('/', '_')}_{frame_id.replace('.png', '.jpg')}"
        # save_path = os.path.join(self.vis_save_dir, save_filename)
        # self.visualize_multi_modal_align(
        #     rgb_image=rgb_with_gt,
        #     ir_image=ir_with_gt,
        #     dvs_image=dvs_with_gt,
        #     velocity_image=velicity_with_gt,
        #     world_coords_map=world_coords_resized_with_gt,
        #     world_coords_map_original=world_coords_resized,
        #     save_path=save_path
        # )
        #
        return data_dict

    def collate_batch(self, batch_list, _unused=False):
        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(batch_list):
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key == 'image':
                    merged = np.stack(val, axis=0)  # (B, 5, 3, H, W)
                    ret[key] = merged
                elif key == 'gt_boxes':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.full((batch_size, max_gt, 9, 3), np.nan, dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    ret[key] = merged
                elif key == 'gt_names':
                    ret[key] = val
                elif key == 'gt_diffs':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    ret[key] = merged
                elif key in ['intrinsic', 'extrinsic', 'distortion']:
                    stacked = np.stack(val, axis=0)
                    stacked = stacked.squeeze(1)
                    ret[key] = stacked
                elif key in ['raw_im_size', 'new_im_size', 'obj_size']:
                    stacked = np.stack(val, axis=0)
                    if key == 'obj_size':
                        stacked = stacked.squeeze(1)
                    ret[key] = stacked
                elif key in ['seq_id', 'frame_id']:
                    ret[key] = val
                elif key == 'stride':
                    ret[key] = val
                elif key == 'sorted_namelist':
                    continue
                elif key in ['hm', 'center_res', 'center_dis', 'dim', 'rot']:
                    try:
                        if key == 'hm':
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            # print(f"hm shapes before pad: {[v.shape for v in val]}")
                            max_cls = max([v.shape[0] for v in val])
                            padded_val = []
                            for v in val:
                                # print(f"current hm shape: {v.shape}, ndim: {v.ndim}")
                                if v.shape[0] < max_cls:
                                    pad_width = ((0, max_cls - v.shape[0]), (0, 0), (0, 0))
                                    v = np.pad(v, pad_width, mode='constant')
                                padded_val.append(v)
                            stacked = np.stack(padded_val, axis=0)
                        else:
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            # print(f"{key} shapes before stack: {[v.shape for v in val]}")
                            stacked = np.stack(val, axis=0)

                        ret[key] = stacked
                    except Exception as e:
                        raise e
                elif key in ['gt_heatmap', 'gt_res_x', 'gt_res_y', 'gt_pts2d']:
                    sample_dim = val[0].ndim
                    max_num_obj = max([v.shape[0] for v in val])
                    pad_width = []
                    for dim in range(sample_dim):
                        if dim == 0:
                            pad_width.append((0, max_num_obj - val[0].shape[0]))
                        else:
                            pad_width.append((0, 0))
                    pad_width = tuple(pad_width)
                    padded_val = []
                    for v in val:
                        if v.ndim != sample_dim:
                            raise ValueError(f"{v.ndim} vs {sample_dim}")
                        cur_pad = max_num_obj - v.shape[0]
                        if cur_pad > 0:
                            cur_pad_width = list(pad_width)
                            cur_pad_width[0] = (0, cur_pad)
                            v_padded = np.pad(v, tuple(cur_pad_width), mode='constant', constant_values=0)
                            padded_val.append(v_padded)
                        else:
                            padded_val.append(v)
                    stacked = np.stack(padded_val, axis=0)
                    ret[key] = stacked
                else:
                    print(f"Unknown key '{key}' ignored")

            except Exception as e:
                print(f"Error in collate_batch for key={key}: {e}")
                raise e
        ret['batch_size'] = batch_size
        ret['sorted_namelist'] = self.dataset_cfg.CLASS_NAMES
        # print("="*50)
        # print(f"batch_size：{batch_size}")
        # for key in ret:
        #     if isinstance(ret[key], np.ndarray):
        #         print(f"  {key}: {ret[key].shape}")
        #     elif isinstance(ret[key], list):
        #         print(f"  {key}: 列表，长度={len(ret[key])}")
        #     else:
        #         print(f"  {key}: {type(ret[key])}")
        # print("="*50)
        # print("collate_batch finished\n")
        return ret

    def generate_prediction_dicts(self, batch_dict, output_path):
        batch_size = batch_dict['batch_size']
        annos = []

        base_debug_dir = os.path.join(os.getcwd(), 'debug_images')
        os.makedirs(base_debug_dir, exist_ok=True)

        # 仅在本地调试时启用显示（通过环境变量控制，默认关闭）
        enable_imshow = os.environ.get('ENABLE_IMSHOW', 'False').lower() == 'true'

        for batch_id in range(batch_size):
            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]

            raw_im_size = batch_dict['raw_im_size'][batch_id]  # (2,)
            obj_size = batch_dict['obj_size'][batch_id]
            intrinsic = batch_dict['intrinsic'][batch_id]  # (3, 3)
            extrinsic = batch_dict['extrinsic'][batch_id]  # (4, 4)
            distortion = batch_dict['distortion'][batch_id]  # (5,)
            gt_boxes = batch_dict['gt_boxes'][batch_id]
            gt_names = batch_dict['gt_names'][batch_id]

            # [center_x, center_y, center_z, l, w, h, a1, a2, a3, class_id]
            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id]  # (N, 10)
            pred_class_ids = pred_boxes9d[:, -1].astype(int)
            pred_names = self.name_from_code(pred_class_ids)
            pred_boxes9d = self.convert_9params_to_9points(pred_boxes9d[:, :-1])  # world -> world

            confidence = batch_dict['confidence'][batch_id]  # (N,)
            sorted_namelist = batch_dict['sorted_namelist'][batch_id]
            frame_dict = {
                'seq_id': seq_id,
                'frame_id': frame_id,
                'obj_size': obj_size,
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'pred_boxes': pred_boxes9d,
                'pred_names': pred_names,
                'confidence': confidence,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'distortion': distortion,
            }

            # 确保图像路径包含扩展名（假设为png，可根据实际情况调整）
            if not any(frame_id.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                frame_id_with_ext = f"{frame_id}.png"
            else:
                frame_id_with_ext = frame_id
            im_path = os.path.join(self.root_path, seq_id, 'images_rgb', frame_id_with_ext)
            frame_dict['im_path'] = im_path

            image = cv2.imread(im_path)
            if image is None:
                print(f"警告：无法读取图像 {im_path}，跳过该帧")
                continue

            # 原始(网络输入/训练时使用)的目标大小
            orig_w, orig_h = self.new_im_width, self.new_im_hight

            # 实际要画框的读取图像大小 (h, w)
            tgt_h, tgt_w = image.shape[:2]

            # 跳过空图像（异常处理）
            if tgt_h == 0 or tgt_w == 0:
                print(f"警告：图像 {im_path} 尺寸异常，跳过该帧")
                continue

            # 计算缩放因子（确保不为0，避免除零错误）
            scale_x = tgt_w / float(orig_w) if orig_w != 0 else 1.0
            scale_y = tgt_h / float(orig_h) if orig_h != 0 else 1.0

            # 调整内参以匹配实际图像尺寸
            resized_intrinsics_rgb = intrinsic.copy()
            resized_intrinsics_rgb[0, 0] *= scale_x  # fx
            resized_intrinsics_rgb[1, 1] *= scale_y  # fy
            resized_intrinsics_rgb[0, 2] *= scale_x  # cx
            resized_intrinsics_rgb[1, 2] *= scale_y  # cy

            # 无预测框且无GT框时，可跳过绘制（可选优化）
            has_pred = len(pred_boxes9d) > 0
            has_gt = 'gt_boxes' in batch_dict and len(batch_dict['gt_boxes'][batch_id]) > 0
            if not has_pred and not has_gt:
                print(f"帧 {frame_id} 无预测框和GT框，跳过绘制")
                annos.append(frame_dict)
                continue

            # 绘制预测框
            image = self.draw_box9d_on_image(
                pred_boxes9d, image,
                img_width=tgt_w,  # 使用实际图像宽度
                img_height=tgt_h,  # 使用实际图像高度
                color=(255, 0, 0),
                intrinsic_mat=resized_intrinsics_rgb,
                extrinsic_mat=extrinsic,
                distortion_matrix=distortion
            )

            # 绘制GT框（如果存在）
            if 'gt_boxes' in batch_dict:
                gt_box9d = batch_dict['gt_boxes'][batch_id]  # (N, 9, 3)
                gt_names = batch_dict['gt_names'][batch_id]  # (N,)
                frame_dict['gt_boxes'] = gt_box9d
                frame_dict['gt_names'] = gt_names

                image = self.draw_box9d_on_image_gt(
                    gt_box9d, image,
                    img_width=tgt_w,  # 使用实际图像宽度
                    img_height=tgt_h,  # 使用实际图像高度
                    color=(0, 0, 255),
                    intrinsic_mat=resized_intrinsics_rgb,
                    extrinsic_mat=extrinsic,
                    distortion_matrix=distortion
                )

            # 处理安全的文件名和路径（替换特殊字符）
            safe_frame_id = frame_id_with_ext.replace('.', '_').replace('=', '_')
            safe_seq_id = seq_id.replace('/', '_').replace('\\', '_')  # 移除路径分隔符
            save_filename = f"batch_{batch_id}_frame_{safe_frame_id}.jpg"
            save_rel_path = os.path.join(safe_seq_id, save_filename)
            save_path = os.path.join(base_debug_dir, save_rel_path)

            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)

            # 保存图像
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                pil_image.save(save_path)
                print(f"已保存调试图像: {save_path}")
            except Exception as e:
                print(f"保存图像失败: {e}，路径: {save_path}")

            # 可选：本地调试时显示图像（服务器环境自动禁用）
            if enable_imshow:
                cv2.imshow('rgb_image', image)
                cv2.waitKey(1)

            annos.append(frame_dict)

        return annos

    def evaluation_by_name(self, annos, metric_root_path, cls_name):

        orin_error, pos_error = self.eval_6d_pose(annos, max_dis=self.dataset_cfg.MAX_DIS)

        plt.figure(figsize=(10, 4))
        plt.scatter(np.arange(0, len(pos_error)), pos_error, s=5)
        plt.title(f"{cls_name} Position Error")
        plt.xlabel("Instance Index")
        plt.ylabel("Position Error (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(metric_root_path, f'{cls_name}_pos_error.png'))
        plt.close()

        print(f"{cls_name}_orin_error： {orin_error.mean():.4f}")
        print(f"{cls_name}_pos_error: {pos_error.mean():.4f}")
        print(f"{cls_name}_orin_error_median: {np.median(orin_error):.4f}")
        print(f"{cls_name}_pos_error_median: {np.median(pos_error):.4f}")
        print(f"{cls_name}_orin_error_min： {orin_error.min():.4f}")
        print(f"{cls_name}_pos_error_min: {pos_error.min():.4f}")
        print(f"{cls_name}_orin_error_max： {orin_error.max():.4f}")
        print(f"{cls_name}_pos_error_max: {pos_error.max():.4f}")

        return_str = f'''
            {cls_name}: 
            orin_error： {orin_error.mean():.4f}
            pos_error: {pos_error.mean():.4f}
            orin_error_median: {np.median(orin_error):.4f}
            pos_error_median: {np.median(pos_error):.4f}
            orin_error_min: {orin_error.min():.4f}
            pos_error_min: {pos_error.min():.4f}
            orin_error_max: {orin_error.max():.4f}
            pos_error_max: {pos_error.max():.4f}
        '''

        pd.DataFrame({f'{cls_name}_orin_error': orin_error.tolist()}).to_csv(
            os.path.join(metric_root_path, f'{cls_name}_orin_error.csv'), index=False)
        pd.DataFrame({f'{cls_name}_pos_error': pos_error.tolist()}).to_csv(
            os.path.join(metric_root_path, f'{cls_name}_pos_error.csv'), index=False)
        pd.DataFrame({
            f'{cls_name}_orin_error': [orin_error.mean()],
            f'{cls_name}_pos_error': [pos_error.mean()]
        }).to_csv(os.path.join(metric_root_path, f'{cls_name}_mean_error.csv'), index=False)

        return return_str

    def evaluation(self, annos, metric_root_path):
        all_names = self.sorted_namelist
        final_str = ''
        print(f"[DEBUG] Total annotations: {len(annos)}, Available classes: {all_names}")

        for cls_n in all_names:
            cls_annos = []
            print(f"\n[DEBUG] Processing class: {cls_n}")

            for i, each_anno in enumerate(annos):
                gt_boxes = each_anno['gt_boxes']  # (N,9,3)
                pred_boxes = each_anno['pred_boxes']
                gt_names = each_anno['gt_names']  # (N,)
                pred_names = each_anno['pred_names']

                print(f"\nAnno {i}:")
                print(f"  gt_boxes shape: {gt_boxes.shape}")
                print(f"  pred_boxes shape: {pred_boxes.shape}")
                print(f"  gt_names: {gt_names}")
                print(f"  pred_names: {pred_names}")
                if len(gt_boxes) > 0:
                    gt_centers = gt_boxes[:, 0, :]
                    print(f"  所有真实框中心点 ({len(gt_centers)} 个):")
                    for idx in range(len(gt_centers)):
                        x, y, z = gt_centers[idx]
                        print(f"    真实框 {idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

                if len(pred_boxes) > 0:
                    pred_centers = pred_boxes[:, 0, :]
                    print(f"  所有预测框中心点 ({len(pred_centers)} 个):")
                    for idx in range(len(pred_centers)):
                        x, y, z = pred_centers[idx]
                        print(f"    预测框 {idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

                if len(gt_boxes) > len(gt_names):
                    if len(gt_boxes) > 0:
                        print("  gt_boxes content:")
                        for box_idx in range(gt_boxes.shape[0]):
                            print(f"    Box {box_idx}:")
                            for point_idx in range(gt_boxes.shape[1]):
                                x, y, z = gt_boxes[box_idx, point_idx]
                                print(f"      Point {point_idx:2d}: x={x:8.4f}, y={y:8.4f}, z={z:8.4f}")
                    else:
                        print("  gt_boxes content: [No boxes]")
                    gt_boxes = gt_boxes[:len(gt_names)]

                mask_gt = (gt_names == cls_n)
                mask_pred = (pred_names == cls_n)

                if mask_gt.sum() > 0:
                    filtered_gt = gt_boxes[mask_gt]
                    filtered_names = gt_names[mask_gt]

                    cls_annos.append({
                        'gt_boxes': filtered_gt,
                        'gt_names': filtered_names,
                        'pred_boxes': pred_boxes[mask_pred],
                        'pred_names': pred_names[mask_pred]
                    })

            if len(cls_annos) >= 2:
                final_str += self.evaluation_by_name(cls_annos, metric_root_path, cls_name=cls_n)

        merged_annos = []
        for each_anno in annos:
            gt_boxes = each_anno['gt_boxes']
            pred_boxes = each_anno['pred_boxes']

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()

            if len(gt_boxes.shape) != 3 or gt_boxes.shape[1:] != (9, 3):
                print(f"[警告] 跳过非9点格式的gt_boxes，形状: {gt_boxes.shape}")
                continue
            if len(pred_boxes.shape) != 3 or pred_boxes.shape[1:] != (9, 3):
                print(f"[警告] 跳过非9点格式的pred_boxes，形状: {pred_boxes.shape}")
                continue

            if len(gt_boxes) > 0:
                merged_annos.append({
                    'gt_boxes': gt_boxes,
                    'gt_names': each_anno['gt_names'],
                    'pred_boxes': pred_boxes,
                    'pred_names': each_anno['pred_names']
                })

        if merged_annos:
            final_str += self.evaluation_by_name(merged_annos, metric_root_path, cls_name='all')
        else:
            print("merged_annos empty")

        return final_str

    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    ## Util funtcions

    def project_lidar_and_get_uvz_rgb_tag(self,
                                          lidar_points,  # (N, 5): x, y, z, intensity, tag
                                          lidar_extrinsic,  # 4x4
                                          camera_extrinsic,  # 4x4
                                          camera_intrinsic,  # 3x3
                                          image  # (H, W, 3)
                                          ):
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        h, w = image.shape[:2]

        xyz = lidar_points[:, :3]
        tags = lidar_points[:, 4]

        # Step 1: LiDAR → World
        lidar_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        points_world = (lidar_extrinsic @ lidar_homo.T).T

        # Step 2: World → Camera
        cam_ext_inv = np.linalg.inv(camera_extrinsic)
        points_cam = (cam_ext_inv @ points_world.T).T
        points_cam_xyz = points_cam[:, :3]

        # Step 3: Camera → OpenCV
        points_cam_homo = np.hstack([points_cam_xyz, np.ones((points_cam_xyz.shape[0], 1))])
        points_opencv = (carla_to_opencv @ points_cam_homo.T).T[:, :3]

        in_front = points_opencv[:, 2] > 0
        points_opencv = points_opencv[in_front]
        points_cam_xyz = points_cam_xyz[in_front]
        tags = tags[in_front]

        uv, _ = cv2.projectPoints(points_opencv, np.eye(3), np.zeros(3), camera_intrinsic, None)
        if uv is None:
            print("[DEBUG ERROR] LIDAR UV = NONE")
            return np.array([])
        uv = uv.reshape(-1, 2)

        results = []
        for i in range(uv.shape[0]):
            u, v = uv[i]
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                z = points_cam_xyz[i, 2]
                r, g, b = image[v_int, u_int].tolist()
                tag = tags[i]
                x_world, y_world, z_world = points_world[i, :3]
                results.append([u, v, z, x_world, y_world, z_world, r, g, b, tag])

        return np.array(results)  # shape: (M, 10)

    def radar_to_velocity_heatmap(self,
                                  radar_data,
                                  radar_extrinsic,
                                  camera_extrinsic,
                                  camera_intrinsic,
                                  image_shape=(720, 1280),
                                  method='max'  # 可选：'max'、'mean'、'sum'
                                  ):
        # 解析雷达数据
        velocity = radar_data[:, 0]
        azim = radar_data[:, 1]
        alt = radar_data[:, 2]
        depth = radar_data[:, 3]

        # 极坐标 → 笛卡尔
        x = depth * np.cos(alt) * np.cos(azim)
        y = depth * np.cos(alt) * np.sin(azim)
        z = depth * np.sin(alt)
        radar_xyz = np.stack([x, y, z], axis=1)

        # 雷达 → 世界
        radar_homo = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
        pts_world = (radar_extrinsic @ radar_homo.T).T[:, :3]

        # 世界 → 相机（再转 Carla → OpenCV）
        cam_extr_inv = np.linalg.inv(camera_extrinsic)
        world_homo = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
        pts_cam = (cam_extr_inv @ world_homo.T).T[:, :3]

        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        pts_cam_homo = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
        pts_opencv = (carla_to_opencv @ pts_cam_homo.T).T[:, :3]

        # 投影
        uv, _ = cv2.projectPoints(pts_opencv, np.zeros(3), np.zeros(3), camera_intrinsic, None)
        if uv is None:
            print("[Debug ERROR] RADAR UV = NONE")
            return np.zeros((1, image_shape[0], image_shape[1]), dtype=np.float32)
        # 关键修复：确保uv是二维数组 (N, 2)
        uv = uv.squeeze()  # 先去除多余维度
        # 如果是一维数组（单一点的情况），转为二维数组
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)
        uv = uv.astype(int)  # 转换为整数坐标

        # 构建热力图
        H, W = image_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32) if method == 'mean' else None

        for i in range(len(uv)):
            # 额外安全检查：确保每个元素都是长度为2的坐标
            if uv[i].size != 2:
                continue  # 跳过无效坐标
            u, v = uv[i]
            # 检查坐标是否在图像范围内
            if 0 <= u < W and 0 <= v < H:
                v_val = velocity[i]
                if method == 'max':
                    heatmap[v, u] = max(heatmap[v, u], v_val)
                elif method == 'sum':
                    heatmap[v, u] += v_val
                elif method == 'mean':
                    heatmap[v, u] += v_val
                    count_map[v, u] += 1.0

        if method == 'mean':
            valid = count_map > 0
            heatmap[valid] /= count_map[valid]

        # 可选平滑
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

        return heatmap[np.newaxis, ...]  # shape: (1, H, W)

    def draw_box9d_on_image(self, boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
                            intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):

        if intrinsic_mat is None:
            intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                      [0, img_height, img_height / 2],
                                      [0, 0, 1]], dtype=np.float32)

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4)

        if distortion_matrix is None:
            distortion_matrix = np.zeros(5, dtype=np.float32)

        # Carla → OpenCV 相机坐标变换
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        extrinsic_inv = np.linalg.inv(extrinsic_mat)

        for i, box in enumerate(boxes9d):
            box_cv = []
            for pt in box:
                if isinstance(pt, torch.Tensor):
                    pt = pt.cpu().numpy()  # 仅对 PyTorch 张量进行转换
                else:
                    pt = np.array(pt)  # 处理其他类型（如列表）
                pt_hom = np.append(pt, 1.0)  # 世界坐标 → 齐次
                pt_cam = extrinsic_inv @ pt_hom  # 世界 → 相机
                pt_cv = carla_to_opencv @ pt_cam  # Carla 相机 → OpenCV 相机
                box_cv.append(pt_cv[:3])

            box_cv = np.array(box_cv, dtype=np.float32)  # (9, 3)

            corners_2d, _ = cv2.projectPoints(box_cv[1:], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
            top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

            for edge in bottom_edges:
                color = (255, 0, 0)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)
            for idx, edge in enumerate(middle_edges):
                intensity = int(255 - (idx / 3) * 150)
                color = (0, intensity, 0)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)
            for edge in top_edges:
                color = (0, 0, 255)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)

            diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
            mid_points = []
            for i, j in diagonals:
                pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
                mid = (pt1 + pt2) / 2
                mid_points.append(mid)
                proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat,
                                                distortion_matrix)
                p1, p2 = proj_pts.reshape(-1, 2).astype(int)
                cv2.line(image, tuple(p1), tuple(p2), (0, 255, 255), 1)

        return image

    def draw_box9d_on_image_gt(self, boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
                               intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):

        if intrinsic_mat is None:
            intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                      [0, img_height, img_height / 2],
                                      [0, 0, 1]], dtype=np.float32)

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4)

        if distortion_matrix is None:
            distortion_matrix = np.zeros(5, dtype=np.float32)

        # Carla → OpenCV 相机坐标变换
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        extrinsic_inv = np.linalg.inv(extrinsic_mat)

        for i, box in enumerate(boxes9d):
            box_cv = []
            for pt in box:
                if isinstance(pt, torch.Tensor):
                    pt = pt.cpu().numpy()  # 仅对 PyTorch 张量进行转换
                else:
                    pt = np.array(pt)  # 处理其他类型（如列表）
                pt_hom = np.append(pt, 1.0)  # 世界坐标 → 齐次
                pt_cam = extrinsic_inv @ pt_hom  # 世界 → 相机
                pt_cv = carla_to_opencv @ pt_cam  # Carla 相机 → OpenCV 相机
                box_cv.append(pt_cv[:3])

            box_cv = np.array(box_cv, dtype=np.float32)  # (9, 3)

            corners_2d, _ = cv2.projectPoints(box_cv[1:], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
            top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

            for edge in bottom_edges:
                color = (255, 255, 255)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)
            for idx, edge in enumerate(middle_edges):
                intensity = int(255 - (idx / 3) * 150)
                color = (255, 255, 255)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)
            for edge in top_edges:
                color = (255, 255, 255)
                idx1, idx2 = edge
                pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
                pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
                cv2.line(image, pt1, pt2, color, 1)

            diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
            mid_points = []
            for i, j in diagonals:
                pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
                mid = (pt1 + pt2) / 2
                mid_points.append(mid)
                proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat,
                                                distortion_matrix)
                p1, p2 = proj_pts.reshape(-1, 2).astype(int)
                cv2.line(image, tuple(p1), tuple(p2), (255, 255, 255), 1)

        return image

    def convert_9params_to_9points(self, box9d_params):
        """
        预测阶段可直接使用的版本：不依赖 self.local_prototypes。
        输入: (N, 9) -> [x,y,z,l,w,h,a1,a2,a3]
        输出: (N, 9, 3) -> [center, eight corners] in world frame
        角点顺序采用固定的立方体原型，保证与投影/可视化一致。
        """
        box9d_params = np.asarray(box9d_params, dtype=np.float32)
        if box9d_params.size == 0:
            return np.empty((0, 9, 3), dtype=np.float32)

        out = []
        for p in box9d_params:
            x, y, z, l, w, h, a1, a2, a3 = p.astype(np.float32)

            # 1) 局部坐标系下的固定角点（以中心为原点）
            # 底面: 0-3, 顶面: 4-7
            corners_obj = np.array([
                [-l / 2, -w / 2, -h / 2],
                [l / 2, -w / 2, -h / 2],
                [l / 2, w / 2, -h / 2],
                [-l / 2, w / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, w / 2, h / 2],
                [-l / 2, w / 2, h / 2],
            ], dtype=np.float32)

            # 2) 旋转矩阵（zyx）
            Rm = R.from_euler('zyx', [a1, a2, a3], degrees=False).as_matrix().astype(np.float32)
            # 保持右手系（健壮性处理）
            if np.linalg.det(Rm) < 0:
                Rm[:, 2] *= -1

            # 3) 旋转 + 平移到世界坐标
            corners_world = (Rm @ corners_obj.T).T + np.array([x, y, z], dtype=np.float32)

            # 4) 拼 9 点（中心点 + 8 角点）
            out.append(np.vstack((np.array([x, y, z], dtype=np.float32), corners_world)))

        return np.stack(out, axis=0).astype(np.float32)

    def _build_seq_list_all_maps(self, split_ratio=0.8):
        all_samples = set()
        map_dirs = sorted([
            d for d in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, d))
        ])

        for map_name in map_dirs:

            carla_data_root = os.path.join(self.root_path, map_name, "carla_data")
            if not os.path.exists(carla_data_root):
                continue

            seq_dirs = sorted(glob(os.path.join(carla_data_root, "0000*")))
            for seq_path in seq_dirs:
                seq_id = os.path.basename(seq_path)

                weather_dirs = [
                    w for w in os.listdir(seq_path)
                    if os.path.isdir(os.path.join(seq_path, w))
                ]
                for weather_name in weather_dirs:
                    weather_path = os.path.join(seq_path, weather_name)
                    drone_dirs = [
                        d for d in os.listdir(weather_path)
                        if os.path.isdir(os.path.join(weather_path, d))
                    ]
                    for drone_name in drone_dirs:
                        full_path = f"{map_name}/carla_data/{seq_id}/{weather_name}/{drone_name}"
                        # full_path = f"{map_name}/carla_data/{seq_id}/clear_day/{drone_name}"
                        # full_path = f"Town01_Opt/carla_data/00001/clear_day/m210-rtk"
                        all_samples.add(full_path)

        all_samples = sorted(list(all_samples))
        self.logger.info(f"Total sequences: {len(all_samples)}")

        split_idx = int(len(all_samples) * split_ratio)

        train_list = all_samples[:split_idx]
        val_list = all_samples[split_idx:]
        random.shuffle(train_list)
        random.shuffle(val_list)

        if self.training:
            self.logger.info(f"Train Sequences: {len(train_list)}")
            return train_list
        else:
            self.logger.info(f"Test Sequences: {len(val_list)}")
            return val_list

    def set_split(self, split):
        super(LAAM6D_Det_Dataset, self).__init__(
            dataset_cfg=self.dataset_cfg,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )

        self.sample_scene_list = []

        self.seq_list = self._build_seq_list_all_maps(split_ratio=0.8)

        for seq_name in self.seq_list:
            seq_path = os.path.join(str(self.root_path), seq_name, 'images_rgb')
            if not os.path.exists(seq_path):
                self.logger.warning(f"No image path: {seq_path}")
                continue

            all_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame in all_frames:
                self.sample_scene_list.append([seq_name, frame])
        self.infos = []
        self.include_CARLA_data(self.mode)

    def generate_world_coords_map(self, lidar_proj_info, image_shape):
        h, w = image_shape
        # x_world, y_world, z_world
        x_map = np.zeros((h, w), dtype=np.float32)
        y_map = np.zeros((h, w), dtype=np.float32)
        z_map = np.zeros((h, w), dtype=np.float32)

        for data in lidar_proj_info:
            u, v, z_carla, x_world, y_world, z_world, r, g, b, tag = data
            u_int, v_int = int(round(u)), int(round(v))

            if 0 <= u_int < w and 0 <= v_int < h:
                if z_map[v_int, u_int] == 0 or z_carla < z_map[v_int, u_int]:
                    x_map[v_int, u_int] = x_world
                    y_map[v_int, u_int] = y_world
                    z_map[v_int, u_int] = z_world

        world_coords_map = np.stack([x_map, y_map, z_map], axis=-1)
        return world_coords_map  # (H, W, 3)

    def convert_9points_to_9params(self, box9d_points):
        all_box_params = []
        self.local_prototypes.clear()

        for single_box in box9d_points:
            center = single_box[0].copy()
            x, y, z = center
            corners_local = single_box[1:] - center  # (8, 3)

            # 1. 计算旋转矩阵（物体自身坐标系）
            x_axis = corners_local[0] / np.linalg.norm(corners_local[0]) if np.linalg.norm(
                corners_local[0]) != 0 else np.array([1, 0, 0])
            y_axis = corners_local[1] / np.linalg.norm(corners_local[1]) if np.linalg.norm(
                corners_local[1]) != 0 else np.array([0, 1, 0])
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) != 0 else np.array([0, 0, 1])
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] *= -1

            # 2. 将局部角点投影到物体自身坐标系（获取沿x/y/z轴的坐标）
            corners_self = np.dot(corners_local, rotation_matrix)  # (8, 3)，每个点在物体坐标系下的坐标

            # 3. 计算物体自身坐标系下的尺寸（l: x轴长度, w: y轴长度, h: z轴长度）
            l = np.max(corners_self[:, 0]) - np.min(corners_self[:, 0])
            w = np.max(corners_self[:, 1]) - np.min(corners_self[:, 1])
            h = np.max(corners_self[:, 2]) - np.min(corners_self[:, 2])

            # 4. 记录局部原型（物体坐标系下的单位尺寸角点）
            scale = np.array([l if l != 0 else 1, w if w != 0 else 1, h if h != 0 else 1])
            local_prototype = corners_self / scale  # 物体坐标系下的归一化角点
            self.local_prototypes.append(local_prototype)

            # 5. 计算欧拉角
            r = R.from_matrix(rotation_matrix)
            angles = r.as_euler('zyx')
            angle1, angle2, angle3 = angles

            box_params = np.array([x, y, z, l, w, h, angle1, angle2, angle3])
            all_box_params.append(box_params)

        return np.array(all_box_params)

    def convert_box_opencv_to_world(self, pts_opencv, extrinsic):
        opencv_to_carla = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        pts_world = []
        for pt in pts_opencv:
            pt_h = np.append(pt, 1.0)  # (4,)
            pt_carla = opencv_to_carla @ pt_h
            pt_world = extrinsic @ pt_carla
            pts_world.append(pt_world[:3])
        return np.array(pts_world, dtype=np.float32)

    def name_from_code(self, name_indices):
        names = [self.dataset_cfg.CLASS_NAMES[int(x)] for x in name_indices]
        return np.array(names)

    def convert_box9d_to_box_param(self, boxes_9d):
        if isinstance(boxes_9d, torch.Tensor):
            boxes_9d = boxes_9d.cpu().numpy()
        boxes_9d = np.asarray(boxes_9d)

        all_box_params = []
        for box in boxes_9d:
            if np.isnan(box).any():
                continue

            center = box[0]
            corners = box[1:9]

            l = np.linalg.norm(corners[0] - corners[1])
            w = np.linalg.norm(corners[1] - corners[2])
            h = np.linalg.norm(corners[0] - corners[4])

            if l < 1e-6 or w < 1e-6 or h < 1e-6:
                print(f"[警告] 边界框尺寸异常: l={l}, w={w}, h={h}，跳过该框")
                continue

            x_axis = corners[1] - corners[0]
            y_axis = corners[3] - corners[0]
            z_axis = corners[4] - corners[0]

            x_norm = np.linalg.norm(x_axis)
            y_norm = np.linalg.norm(y_axis)
            z_norm = np.linalg.norm(z_axis)

            if x_norm < 1e-6 or y_norm < 1e-6 or z_norm < 1e-6:
                print(f"[警告] 轴向量模长为0，跳过该框")
                continue

            rot_mat = np.stack([
                x_axis / x_norm,
                y_axis / y_norm,
                z_axis / z_norm
            ], axis=1)

            try:
                r = R.from_matrix(rot_mat)
                euler = r.as_euler('zyx', degrees=False)
            except:
                print(f"[警告] 旋转矩阵无效，使用默认欧拉角")
                euler = np.zeros(3)

            box_param = np.concatenate([center, [l, w, h], euler])
            all_box_params.append(box_param)

        return np.array(all_box_params) if all_box_params else np.empty((0, 9))

    def eval_key_points_error(self, annos):

        all_error = []

        for i, each_anno in enumerate(annos):
            key_points_2d = each_anno['key_points_2d']
            gt_pts2d = each_anno['gt_pts2d']

            all_error.append(np.abs(key_points_2d - gt_pts2d).reshape(-1))

        all_error = np.concatenate(all_error)

        return all_error

    def euler_angle_error(self, pred, gt):
        """
        计算弧度制下欧拉角的角度误差。

        参数:
        pred (numpy.ndarray): 预测的欧拉角，形状为 (N, 3)。
        gt (numpy.ndarray): 真实的欧拉角，形状为 (N, 3)。

        返回:
        numpy.ndarray: 每个样本的角度误差，形状为 (N,)。
        """
        diff = pred - gt

        diff = np.arctan2(np.sin(diff), np.cos(diff))

        error = np.linalg.norm(diff, axis=1)

        return error

    def euler_angle_error_rad(self, pred, gt):

        diff = pred - gt

        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        error = np.sum(np.abs(diff), axis=1)
        return error

    def limit(self, ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self, pred, gt):

        a = np.abs(pred - gt)
        b = 2 * np.pi - np.abs(pred - gt)

        res = np.stack([a, b])

        res = np.min(res, axis=0)

        return res

    def hungarian_match_allow_unmatched(self, x, y, unmatched_cost: float = 1e5):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        x = np.asarray(x)
        y = np.asarray(y)

        assert x.ndim == 2 and x.shape[1] == 3, f"x形状错误，应为(N,3)，实际为{x.shape}"
        assert y.ndim == 2 and y.shape[1] == 3, f"y形状错误，应为(M,3)，实际为{y.shape}"

        original_x_len = x.shape[0]

        valid_x = np.all(np.isfinite(x), axis=1)
        valid_y = np.all(np.isfinite(y), axis=1)
        x = x[valid_x]
        y = y[valid_y]
        N, M = x.shape[0], y.shape[0]

        if N == 0 or M == 0:
            return np.full(original_x_len, -1, dtype=int)

        try:
            cost = cdist(x, y)
        except Exception as e:
            print(f"距离计算失败: {e}")
            return np.full(original_x_len, -1, dtype=int)

        cost[~np.isfinite(cost)] = unmatched_cost

        size = max(N, M)
        padded_cost = np.full((size, size), unmatched_cost, dtype=np.float64)
        padded_cost[:N, :M] = cost

        try:
            row_ind, col_ind = linear_sum_assignment(padded_cost)
        except Exception as e:
            print(f"匈牙利算法失败: {e}")
            return np.full(original_x_len, -1, dtype=int)

        x_to_y = np.full(original_x_len, -1, dtype=int)
        original_x_indices = np.where(valid_x)[0]
        original_y_indices = np.where(valid_y)[0]

        for r, c in zip(row_ind, col_ind):
            if r < N and c < M and padded_cost[r, c] < unmatched_cost:
                x_to_y[original_x_indices[r]] = original_y_indices[c]

        return x_to_y

    def match_gt(self, gt_box9d, pred_boxes9d):
        new_pred = []

        gt_centers = gt_box9d[:, 0, :]
        pred_centers = pred_boxes9d[:, 0, :]

        x_to_y = self.hungarian_match_allow_unmatched(gt_centers, pred_centers)

        for i, j in enumerate(x_to_y):
            if j == -1:
                new_pred.append(np.full((9, 3), np.nan, dtype=np.float32))
            else:
                new_pred.append(pred_boxes9d[j])

        return gt_box9d, np.array(new_pred)

    def val_rotation(self, pred_q, gt_q):
        """
        test model, compute error (numpy)
        input:
            pred_q: [4,]
            gt_q: [4,]
        returns:
            rotation error (degrees):
        """
        if isinstance(pred_q, np.ndarray):
            predicted = pred_q
            groundtruth = gt_q
        else:
            predicted = pred_q.cpu().numpy()
            groundtruth = gt_q.cpu().numpy()

        # d = abs(np.sum(np.multiply(groundtruth, predicted)))
        # if d != d:
        #     print("d is nan")
        #     raise ValueError
        # if d > 1:
        #     d = 1
        # error = 2 * np.arccos(d) * 180 / np.pi
        # d     = abs(np.dot(groundtruth, predicted))
        # d     = min(1.0, max(-1.0, d))

        d = np.abs(np.dot(groundtruth, predicted))
        d = np.minimum(1.0, np.maximum(-1.0, d))
        error = 2 * np.arccos(d) * 180 / np.pi

        return error

    def val_rotation_euler_old(self, pred_q, gt_q):
        rotation_pred = R.from_euler('zyx', pred_q, degrees=False)
        rotation_gt = R.from_euler('zyx', gt_q, degrees=False)

        rotation_pred = rotation_pred.as_quat()
        rotation_gt = rotation_gt.as_quat()

        all_error = [self.val_rotation(p, q) for p, q in zip(rotation_pred, rotation_gt)]

        all_error = np.array(all_error)

        all_error[all_error > 90] = 180 - all_error[all_error > 90]

        return all_error

    def val_rotation_euler(self, pred_euler, gt_euler):  # 改对函数名，明确输入是欧拉角
        """
        输入：
            pred_euler：预测的欧拉角 (3,)，如[a1,a2,a3]
            gt_euler：GT的欧拉角 (3,)
        输出：
            旋转误差（角度）
        """
        try:
            # 1. 欧拉角→旋转对象（关键：确认单位！若输入是角度，degrees=True）
            # 若模型输出的是弧度（范围≈[-3.14,3.14]），用degrees=False；若是角度（≈[-180,180]），用True
            rotation_pred = R.from_euler('zyx', pred_euler, degrees=False)  # 注意旋转顺序是否和GT一致
            rotation_gt = R.from_euler('zyx', gt_euler, degrees=False)

            # 2. 转换为四元数（整体代表旋转，不是分量）
            pred_q = rotation_pred.as_quat()
            gt_q = rotation_gt.as_quat()

            # 3. 单位化四元数（必须步骤）
            pred_q = pred_q / np.linalg.norm(pred_q)
            gt_q = gt_q / np.linalg.norm(gt_q)

            # 4. 计算旋转误差
            d = np.abs(np.dot(gt_q, pred_q))  # 点积的绝对值（避免q和-q的符号问题）
            d = np.clip(d, -1.0, 1.0)  # 确保在[-1,1]内（浮点数误差可能超范围）
            error = 2 * np.arccos(d) * 180 / np.pi  # 转换为角度

            # 5. 取最小角度（旋转误差最大90度，比如120度等价于60度）
            if error > 90:
                error = 180 - error
            return error

        except Exception as e:
            print(f"旋转误差计算失败：{e}，输入欧拉角：pred={pred_euler}, gt={gt_euler}")
            return np.nan  # 出错时返回nan，后续过滤

    def eval_box6d_error(self, annos, max_dis):
        all_orin_error = []
        all_pos_error = []

        for i, each_anno in enumerate(annos):
            gt_box9d = each_anno['gt_boxes']
            pred_boxes9d = each_anno['pred_boxes']

            if isinstance(gt_box9d, torch.Tensor):
                gt_box9d = gt_box9d.cpu().numpy()
            if isinstance(pred_boxes9d, torch.Tensor):
                pred_boxes9d = pred_boxes9d.cpu().numpy()

            if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
                continue

            gt_box9d, pred_boxes9d = self.match_gt(gt_box9d, pred_boxes9d)

            if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
                continue

            pred_xyz = pred_boxes9d[:, 0, :]
            gt_xyz = gt_box9d[:, 0, :]

            # (center, l, w, h, a1, a2, a3)
            pred_params = self.convert_box9d_to_box_param(pred_boxes9d)  # (N, 9)
            gt_params = self.convert_box9d_to_box_param(gt_box9d)  # (N, 9)

            pred_angle = pred_params[:, 6:9]  # (N, 3)
            gt_angle = gt_params[:, 6:9]  # (N, 3)

            dis_error = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)
            # angle_error = self.val_rotation_euler(pred_angle, gt_angle)
            angle_error = np.array([self.val_rotation_euler(pred_angle[i], gt_angle[i])
                                    for i in range(len(pred_angle))])

            dis_error = dis_error[~np.isnan(dis_error) & ~np.isinf(dis_error)]
            angle_error = angle_error[~np.isnan(angle_error) & ~np.isinf(angle_error)]

            if len(dis_error) > 0 and len(angle_error) > 0:
                all_orin_error.append(angle_error)
                all_pos_error.append(dis_error)

        if len(all_orin_error) > 0 and len(all_pos_error) > 0:
            all_orin_error = np.concatenate(all_orin_error)
            all_pos_error = np.concatenate(all_pos_error)
        else:
            all_orin_error = np.array([0.0])
            all_pos_error = np.array([0.0])

        return all_orin_error, all_pos_error

    def eval_6d_pose(self, annos, max_dis):

        orin_error, pos_error = self.eval_box6d_error(annos, max_dis)

        return orin_error, pos_error

    def visualize_multi_modal_align(self,
                                    rgb_image,
                                    ir_image,
                                    dvs_image,
                                    velocity_image,
                                    world_coords_map,
                                    world_coords_map_original,
                                    save_path,
                                    sample_coords=None):
        """
        可视化多模态图像（RGB、IR、DVS、Velocity）与世界坐标映射图的对齐效果
        Args:
            rgb_image: RGB图像 (H, W, 3)
            ir_image: 红外图像 (H, W) 或 (H, W, 3)
            dvs_image: DVS事件图像 (H, W, 3)
            velocity_image: 雷达速度热力图 (H, W) 或 (H, W, 3)
            world_coords_map: 世界坐标映射图 (H, W, 3)
            save_path: 保存路径
            sample_coords: 待验证的采样坐标列表 [(u1, v1), (u2, v2), ...]，默认随机生成3个
        """
        # 确保所有图像尺寸一致
        h, w = rgb_image.shape[:2]
        assert ir_image.shape[:2] == (h, w), f"IR尺寸不匹配: {ir_image.shape[:2]} vs {(h, w)}"
        assert dvs_image.shape[:2] == (h, w), f"DVS尺寸不匹配: {dvs_image.shape[:2]} vs {(h, w)}"
        assert velocity_image.shape[:2] == (h, w), f"Velocity尺寸不匹配: {velocity_image.shape[:2]} vs {(h, w)}"
        assert world_coords_map.shape[:2] == (h, w), f"世界坐标图尺寸不匹配: {world_coords_map.shape[:2]} vs {(h, w)}"

        # 预处理单通道图像为3通道（便于统一显示）
        if len(ir_image.shape) == 2:
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        if len(velocity_image.shape) == 2:
            velocity_image = cv2.applyColorMap(velocity_image, cv2.COLORMAP_JET)  # 热力图上色

        # 提取世界坐标X通道用于可视化
        x_coords = world_coords_map[..., 0]
        valid_mask = x_coords != 0
        x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
        if np.any(valid_mask):
            x_valid = x_coords[valid_mask]
            x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
        world_vis = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)  # 世界坐标可视化

        x_coords = world_coords_map_original[..., 0]
        valid_mask = x_coords != 0
        x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
        if np.any(valid_mask):
            x_valid = x_coords[valid_mask]
            x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
        world_vis2 = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)  # 世界坐标可视化

        # 生成采样坐标（默认随机选3个有效点）
        if sample_coords is None:
            sample_coords = []
            if np.any(valid_mask):
                # 从有效坐标中随机选3个
                valid_uv = np.argwhere(valid_mask)  # (v, u) 格式
                if len(valid_uv) >= 3:
                    sample_indices = np.random.choice(len(valid_uv), 3, replace=False)
                    sample_coords = [(valid_uv[i][1], valid_uv[i][0]) for i in sample_indices]  # 转为 (u, v)
                else:
                    # 若无足够有效点，选图像中心附近
                    sample_coords = [(w // 2, h // 2), (w // 3, h // 3), (2 * w // 3, 2 * h // 3)]
            else:
                sample_coords = [(w // 2, h // 2), (w // 3, h // 3), (2 * w // 3, 2 * h // 3)]

        # 配置支持中文的字体（确保系统中已安装）
        matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 创建画布（2行3列布局）
        plt.figure(figsize=(18, 12))
        modalities = [
            ("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)),
            ("IR", cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)),
            ("DVS", cv2.cvtColor(dvs_image, cv2.COLOR_BGR2RGB)),
            ("Radar HM", cv2.cvtColor(velocity_image, cv2.COLOR_BGR2RGB)),
            ("世界坐标X通道", cv2.cvtColor(world_vis, cv2.COLOR_BGR2RGB)),
            ("世界坐标AA通道", cv2.cvtColor(world_vis2, cv2.COLOR_BGR2RGB)),
        ]

        # 绘制每个模态并标注采样坐标
        for i, (title, img) in enumerate(modalities, 1):
            plt.subplot(2, 3, i)
            plt.title(title, fontsize=10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"已保存多模态对齐可视化图至：{save_path}")

    def decode_box_centers_from_world(self, centers_world, extrinsic_inv):
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        centers_world = np.asarray(centers_world).reshape(-1, 3)
        centers_hom = np.hstack([centers_world, np.ones((centers_world.shape[0], 1))])  # (N, 4)
        centers_cam = (extrinsic_inv @ centers_hom.T).T  # (N, 4)
        centers_cv = (carla_to_opencv @ centers_cam.T).T  # (N, 4)
        return centers_cv[:, :3].astype(np.float32)

    def xyz_to_uv(self, xyz_coords, img_width=1920., img_height=1080.,
                  intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None,
                  return_average=True):
        """
        将XYZ世界坐标转换为图像上的UV像素坐标，并可计算平均偏移量
        
        参数:
            xyz_coords: 3D坐标，可以是单个坐标(列表/数组)或多个坐标的列表 (N, 3)
            img_width: 图像宽度
            img_height: 图像高度
            intrinsic_mat: 相机内参矩阵
            extrinsic_mat: 相机外参矩阵
            distortion_matrix: 畸变系数矩阵
            return_average: 是否返回所有点的平均偏移量
        
        返回:
            如果return_average=True: 平均UV坐标 (1, 2)
            否则: 所有点的UV坐标 (N, 2)
        """
        # 设置默认参数
        if intrinsic_mat is None:
            intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                      [0, img_height, img_height / 2],
                                      [0, 0, 1]], dtype=np.float32)

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4)

        if distortion_matrix is None:
            distortion_matrix = np.zeros(5, dtype=np.float32)

        # 坐标变换矩阵 (Carla → OpenCV)
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        # 计算外参矩阵的逆矩阵
        extrinsic_inv = np.linalg.inv(extrinsic_mat)

        # 处理输入坐标，确保是二维数组形式 (N, 3)
        coords = np.asarray(xyz_coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        # 转换坐标
        uv_coords = []
        for pt in coords:
            # 转换为齐次坐标 (x, y, z, 1)
            pt_hom = np.append(pt, 1.0)

            # 世界坐标 → 相机坐标
            pt_cam = extrinsic_inv @ pt_hom

            # Carla相机坐标 → OpenCV相机坐标
            pt_cv = carla_to_opencv @ pt_cam

            # 投影到图像平面获取UV坐标
            proj_pt, _ = cv2.projectPoints(pt_cv[:3], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
            uv = proj_pt.reshape(2).astype(int)
            uv_coords.append(uv)

        uv_array = np.array(uv_coords)

        # 如果需要，计算并返回平均值
        if return_average and len(uv_array) > 0:
            return np.mean(uv_array, axis=0, keepdims=True).astype(int)
        else:
            return uv_array

    def register_images_by_center(self, rgb_img, ir_img, dvs_img,
                                  center_rgb, center_ir, center_dvs,
                                  intrinsic_rgb, intrinsic_ir, intrinsic_dvs):
        """
        保持RGB图像不变，通过平移IR和DVS图像使其中心点与RGB对齐
        平移方式：左侧超出部分截断，右侧不足部分用黑色填充
        
        参数:
            rgb_img: RGB图像（保持不变）
            ir_img: IR图像（需要平移）
            dvs_img: DVS图像（需要平移）
            center_rgb: RGB中心点坐标 (u, v)
            center_ir: IR中心点坐标 (u, v)
            center_dvs: DVS中心点坐标 (u, v)
            内参矩阵: 3x3矩阵
        """
        # 提取中心点坐标 (u, v)
        center_rgb = (int(center_rgb[0][0]), int(center_rgb[0][1]))
        center_ir = (int(center_ir[0][0]), int(center_ir[0][1]))
        center_dvs = (int(center_dvs[0][0]), int(center_dvs[0][1]))

        # print("\n===== 初始中心点信息 =====")
        # print(f"RGB中心点: {center_rgb}")
        # print(f"IR中心点: {center_ir}")
        # print(f"DVS中心点: {center_dvs}")

        # 获取图像尺寸（假设所有图像尺寸相同）
        h, w = rgb_img.shape[:2]
        # print(f"图像尺寸: {w}x{h}")

        # 计算需要平移的像素数（以RGB为基准）
        # 正值：需要向右平移；负值：需要向左平移
        ir_shift_u = center_rgb[0] - center_ir[0]
        ir_shift_v = center_rgb[1] - center_ir[1]

        dvs_shift_u = center_rgb[0] - center_dvs[0]
        dvs_shift_v = center_rgb[1] - center_dvs[1]

        # print(f"\n===== 平移量计算 =====")
        # print(f"IR需要平移: 水平{ir_shift_u}px, 垂直{ir_shift_v}px")
        # print(f"DVS需要平移: 水平{dvs_shift_u}px, 垂直{dvs_shift_v}px")

        # ------------------------------
        # 平移IR图像
        # ------------------------------
        # 创建平移矩阵 [1,0,dx; 0,1,dy]
        ir_M = np.float32([[1, 0, ir_shift_u], [0, 1, ir_shift_v]])
        # 执行平移：左侧超出部分截断，右侧不足部分用黑色填充
        aligned_ir = cv2.warpAffine(
            ir_img,
            ir_M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,  # 超出部分用常数填充
            borderValue=0  # 填充黑色
        )

        # ------------------------------
        # 平移DVS图像
        # ------------------------------
        dvs_M = np.float32([[1, 0, dvs_shift_u], [0, 1, dvs_shift_v]])
        aligned_dvs = cv2.warpAffine(
            dvs_img,
            dvs_M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # ------------------------------
        # 计算平移后的中心点（应该与RGB一致）
        # ------------------------------
        ir_new_center = (center_ir[0] + ir_shift_u, center_ir[1] + ir_shift_v)
        dvs_new_center = (center_dvs[0] + dvs_shift_u, center_dvs[1] + dvs_shift_v)

        # ------------------------------
        # 更新内参（主点坐标同步更新）
        # ------------------------------
        new_intrinsic_rgb = intrinsic_rgb.copy()  # RGB内参不变
        new_intrinsic_ir = intrinsic_ir.copy()
        new_intrinsic_dvs = intrinsic_dvs.copy()

        # 内参主点坐标随平移量调整
        new_intrinsic_ir[0, 2] += ir_shift_u  # cx
        new_intrinsic_ir[1, 2] += ir_shift_v  # cy

        new_intrinsic_dvs[0, 2] += dvs_shift_u
        new_intrinsic_dvs[1, 2] += dvs_shift_v

        # ------------------------------
        # 输出结果验证
        # ------------------------------
        # print(f"\n===== 对齐结果 =====")
        # print(f"RGB中心点: {center_rgb}")
        # print(f"IR平移后中心点: {ir_new_center}")
        # print(f"DVS平移后中心点: {dvs_new_center}")

        return {
            'rgb': rgb_img,  # RGB保持不变
            'ir': aligned_ir,
            'dvs': aligned_dvs,
            'intrinsics': {
                'rgb': new_intrinsic_rgb,
                'ir': new_intrinsic_ir,
                'dvs': new_intrinsic_dvs
            },
            'shift_info': {
                'ir': (ir_shift_u, ir_shift_v),
                'dvs': (dvs_shift_u, dvs_shift_v)
            }
        }
