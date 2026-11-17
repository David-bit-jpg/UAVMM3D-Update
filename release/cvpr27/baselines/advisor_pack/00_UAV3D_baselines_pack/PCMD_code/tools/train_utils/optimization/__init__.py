from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle,CosineWarmup


def build_optimizer(model, optim_cfg, backbone_lr_mult=1.0):
    """backbone_lr_mult < 1 时给骨干单独一个更小的学习率。

    微调时这一条很关键：LR 1e-3 全网络解冻，100 个 iteration 就能把损失从 127
    压到 2.9，预训练权重在第 200 步之前就被冲干净了 —— 迁移臂因此退化成从零训练。
    """
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle' or optim_cfg.OPTIMIZER == 'adam_cosin':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]

        if backbone_lr_mult == 1.0:
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        else:
            def get_layer_groups(m):
                # 拆成 [骨干, 其余]。DetectorTemplate 用 add_module 注册子模块，
                # module_list 只是个普通 list 不会重复注册，named_children 不会重复计参数。
                bb, rest = [], []
                for name, child in m.named_children():
                    (bb if name == 'backbone_2d' else rest).extend(flatten_model(child))
                assert bb, '模型里没有 backbone_2d，无法分层设学习率'
                return [nn.Sequential(*bb), nn.Sequential(*rest)]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
        if backbone_lr_mult != 1.0:
            optimizer.lr_mults = [backbone_lr_mult, 1.0]
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosin':
        lr_scheduler = CosineWarmup(
            optimizer, total_steps, optim_cfg.WARMUP_EPOCH * total_iters_each_epoch, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
