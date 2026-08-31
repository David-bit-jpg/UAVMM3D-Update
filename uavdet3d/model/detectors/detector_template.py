import os
import torch
import torch.nn as nn
from uavdet3d.model import backbone_2d, backbone_3d, dense_head_2d, dense_head_3d, roi_head_2d, roi_head_3d


class DetectorTemplate(nn.Module):
    def __init__(self, model_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'backbone_2d', 'backbone_3d', 'dense_head_2d',  'dense_head_3d',  'roi_head_2d',  'roi_head_3d'
        ]

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'image_shape': self.dataset.dataset_cfg.IM_RESIZE,
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        this_module = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        this_module = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_dense_head_2d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_2D', None) is None:
            return None, model_info_dict
        this_module = dense_head_2d.__all__[self.model_cfg.DENSE_HEAD_2D.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_dense_head_3d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_3D', None) is None:
            return None, model_info_dict
        this_module = dense_head_3d.__all__[self.model_cfg.DENSE_HEAD_3D.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_roi_head_2d(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD_2D', None) is None:
            return None, model_info_dict
        this_module = roi_head_2d.__all__[self.model_cfg.ROI_HEAD_2D.NAME](
            model_cfg=self.model_cfg.ROI_HEAD_2D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def build_roi_head_3d(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD_3D', None) is None:
            return None, model_info_dict
        this_module = roi_head_3d.__all__[self.model_cfg.ROI_HEAD_3D.NAME](
            model_cfg=self.model_cfg.ROI_HEAD_3D,
        )
        model_info_dict['module_list'].append(this_module)
        return this_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        raise NotImplementedError

    def load_params_from_file(self, filename, to_cpu, logger=None):
        """载入权重；形状对不上的张量会被跳过而不是报错。

        注意 strict=False 只容忍「多出来 / 缺失的 key」，**形状不匹配依然会抛
        RuntimeError**。跨域迁移时（例如源域 hm 是 7 类、目标域 2 类）必然有几层
        形状不同，所以这里先按 名字+形状 过滤再载入，并把跳过的层打印出来 ——
        迁移最怕的就是以为加载成功、实际大半没载上。
        """
        loc_type = torch.device('cpu') if to_cpu else None
        dict_sta = torch.load(filename, map_location=loc_type, weights_only=False)
        src_sd = dict_sta['model_state'] if 'model_state' in dict_sta else dict_sta

        model_sd = self.state_dict()
        filtered, shape_bad, unexpected = {}, [], []
        for k, v in src_sd.items():
            if k not in model_sd:
                unexpected.append(k)
            elif tuple(model_sd[k].shape) != tuple(v.shape):
                shape_bad.append('%s: ckpt%s vs model%s'
                                 % (k, tuple(v.shape), tuple(model_sd[k].shape)))
            else:
                filtered[k] = v

        self.load_state_dict(filtered, strict=False)

        n_loaded = sum(int(v.numel()) for v in filtered.values())
        n_total = sum(int(v.numel()) for v in model_sd.values())
        not_init = [k for k in model_sd if k not in filtered]

        log = logger.info if logger is not None else print
        log('loading weights: 载入 %d/%d 个张量, %.2fM/%.2fM 参数 (%.1f%%)'
            % (len(filtered), len(model_sd), n_loaded / 1e6, n_total / 1e6,
               100.0 * n_loaded / max(n_total, 1)))
        if shape_bad:
            log('  形状不符已跳过 (%d): %s' % (len(shape_bad), shape_bad[:8]))
        if unexpected:
            log('  ckpt 有、模型没有 (%d): %s' % (len(unexpected), unexpected[:8]))
        if not_init:
            log('  未初始化、保持随机 (%d): %s' % (len(not_init), not_init[:8]))
        return len(filtered), len(model_sd)

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type, weights_only=False)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        return it, epoch

