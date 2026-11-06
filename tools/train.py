import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from uavdet3d.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import load_data_to_gpu
from uavdet3d.model import build_network, model_fn_decorator
from uavdet3d.utils import common_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_model, checkpoint_state
from tools.test import repeat_eval_ckpt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/models/uavdet_3d/laa3d/center9d_swin_base.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--pretrained_skip', type=str, nargs='*', default=None,
                        help='跨域迁移时主动不加载的层，按名字子串匹配。'
                             '例如 --pretrained_skip hm rot '
                             '（类别数不同的 hm、旋转约定不同的 rot）')
    parser.add_argument('--freeze', type=str, nargs='*', default=None,
                        help='冻结名字里含这些子串的参数（如 backbone_2d）。'
                             '用于「线性探针」诊断：只训检测头，看预训练特征本身能用到什么程度')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='配合 --freeze 使用：前 N 轮冻结，第 N 轮开始解冻（渐进解冻）。'
                             '0 表示全程冻结')
    parser.add_argument('--backbone_lr_mult', type=float, default=1.0,
                        help='骨干的学习率倍率（分层学习率）。微调时设 0.1 左右，'
                             '避免大学习率把预训练权重冲掉')
    parser.add_argument('--freeze_bn_stats', action='store_true',
                        help='被冻结的模块同时锁在 eval 模式，BN 的 running_mean/var 也不更新。'
                             '不加则允许 BN 统计量适应目标域（相当于 AdaBN，对迁移更有利）')
    parser.add_argument('--pretrained_src_max_dis', type=float, default=None,
                        help='源域 ckpt 训练时的 MAX_DIS。给了就把 center_dis 输出层'
                             '按 src/dst 缩放，米制深度预测原样保持')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--seed', type=int, default=666,
                        help='配合 --fix_random_seed 的随机种子。原来写死 666，「多种子」实验跑出来会完全一样')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                        help='固定种子时仍开 cudnn.benchmark（不追求逐位复现，只要种子决定数据顺序和增广）')
    parser.add_argument('--val_split', type=str, default=None,
                        help='按验证集选轮：DATA_SPLIT 名（如 val）。每次验证后分数更高就存 ckpt/best.pth')
    parser.add_argument('--val_interval', type=int, default=1, help='验证集抽帧间隔')
    parser.add_argument('--val_every', type=int, default=1, help='每 N 轮验证一次（最后一轮一定验证）')
    parser.add_argument('--val_match', type=str, default='top1', choices=['top1', 'greedy'],
                        help='top1 = 每帧 1 个 GT（MAV6D）；greedy = 多目标（仿真）')
    parser.add_argument('--skip_test_eval', action='store_true',
                        help='训练结束后不跑仓库自带的测试集评测（测试集只在选好轮次后由 eval_camnorm.py 测一次）')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='bfloat16 自动混合精度训练（验证/评测仍是 float32）')
    parser.add_argument('--anchor_ckpt', default=None,
                        help='锚定蒸馏：把这些头的输出钉在该权重（教师）上，只让其余部分继续学。'
                             '用途：继续训朝向而位置按构造不退（2026-09-20）')
    parser.add_argument('--anchor_keys', default='hm,center_res,center_dis,dim,size2d',
                        help='被钉住的头，逗号分隔')
    parser.add_argument('--anchor_weight', type=float, default=1.0, help='锚定项的权重')
    # 11-02 检测器整图特征对抗对齐（常规 UDA 对照，DA-Faster 图像级）：仿真图（源域）+ 当前训练图（真实，目标域）
    # 的骨干输出特征逐像素过梯度反转层喂域分类器。仿真图只进对抗项，不进检测损失；仿真前向时 BN 用滑动统计、不更新，
    # 保证与不对齐的版本只差这一项。默认关。
    parser.add_argument('--da_cfg', default=None, help='源域（仿真）数据所用的模型配置，开启整图特征对抗对齐')
    parser.add_argument('--da_weight', type=float, default=0.1, help='对抗项权重')
    parser.add_argument('--da_workers', type=int, default=2)
    # 11-02 原型约束 loss：训练开始时用起点模型（eval、不求梯度）在仿真目标格（fg_mask）特征上做球面 k-means 得 K 个原型（固定不更新）；
    # 训练中每个真实伪标签目标格特征与 K 个原型做对比（温度 softmax，目标 = 最近原型），把真实目标特征拉回仿真上见过的样子。只作用在目标格，背景不碰。默认关。
    parser.add_argument('--proto_cfg', default=None, help='源域（仿真）数据配置，开启原型 loss')
    parser.add_argument('--proto_weight', type=float, default=0.1)
    parser.add_argument('--proto_k', type=int, default=30)
    parser.add_argument('--proto_temp', type=float, default=0.1)
    parser.add_argument('--proto_nsim', type=int, default=8000)
    parser.add_argument('--proto_bank', default=None, help='离线建好的原型库（K×C 原始特征），给了就不在训练开始时现聚')
    parser.add_argument('--proto_min_dis', type=float, default=0.0, help='11-05 距离门控：只对伪标签距离 ≥ 此值（米）的帧算原型 loss；0 = 关（逐位不变）')
    parser.add_argument('--proto_clean', action='store_true', help='11-05 建原型时关掉仿真的模糊/运动模糊/目标光环/光度增强（视角缩放和翻转保留）')
    parser.add_argument('--proto_dist', default='ce', choices=['ce', 'l2'], help='ce=原型对比交叉熵；l2=到最近原型的 L2 距离（老师方案）')
    parser.add_argument('--src_cfg', default=None, help='源域（仿真）配置：自训练时同时加仿真真值监督 L_source')
    parser.add_argument('--src_weight', type=float, default=1.0)
    parser.add_argument('--src_workers', type=int, default=2)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(args.seed + cfg.LOCAL_RANK)
        if args.cudnn_benchmark:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        seed=args.seed if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION,
                                backbone_lr_mult=args.backbone_lr_mult)
    if args.backbone_lr_mult != 1.0:
        logger.info('分层学习率: 骨干 lr x %.3f，检测头 lr x 1.0' % args.backbone_lr_mult)

    # 冻结要放在 build_optimizer【之后】：OptimWrapper.create 用 trainable_params()
    # 按 requires_grad 过滤参数组，先冻结的话这些参数根本进不了优化器，
    # 后面 --freeze_epochs 想解冻也解不了。放在后面则它们留在参数组里，
    # Adam 遇到 grad=None 会跳过，OptimWrapper.step 的权重衰减也显式跳过 requires_grad=False。
    unfreeze_hook = None
    if args.freeze:
        n_frozen = n_total = 0
        for name, param in model.named_parameters():
            n_total += param.numel()
            if any(pat in name for pat in args.freeze):
                param.requires_grad = False
                n_frozen += param.numel()
        logger.info('冻结参数: 模式 %s -> %d / %d (%.1f%%) 不参与训练%s'
                    % (args.freeze, n_frozen, n_total, 100.0 * n_frozen / max(n_total, 1),
                       '，第 %d 轮解冻' % args.freeze_epochs if args.freeze_epochs > 0 else '（全程冻结）'))
        if args.freeze_bn_stats:
            # requires_grad=False 只挡梯度，挡不住 BN 更新 running stats。
            # 把这些子模块永久钉在 eval：覆盖它们的 train()，让上层的 model.train() 无效。
            n_mod = 0
            for name, mod in model.named_modules():
                if name and any(pat in name for pat in args.freeze):
                    mod.eval()
                    mod.train = (lambda mode=True, _m=mod: _m)
                    n_mod += 1
            logger.info('  另有 %d 个子模块锁定在 eval 模式（BN 统计量也冻结）' % n_mod)

        if args.freeze_epochs > 0:
            if args.freeze_bn_stats:
                logger.warning('  --freeze_bn_stats 会把子模块永久钉在 eval，'
                               '解冻后 BN 统计量仍然不更新')

            def unfreeze_hook(cur_epoch, _m=model, _pats=list(args.freeze),
                              _at=args.freeze_epochs, _lg=logger):
                if cur_epoch != _at:
                    return
                n = 0
                for name, param in _m.named_parameters():
                    if any(pat in name for pat in _pats) and not param.requires_grad:
                        param.requires_grad = True
                        n += param.numel()
                _lg.info('第 %d 轮：解冻 %s，%d 个参数重新参与训练' % (cur_epoch, _pats, n))

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        dis_rescale = None
        if args.pretrained_src_max_dis is not None:
            dst_max_dis = float(cfg.DATA_CONFIG.MAX_DIS)
            dis_rescale = float(args.pretrained_src_max_dis) / dst_max_dis
            logger.info('center_dis 归一化尺度换算: 源域 MAX_DIS=%s -> 目标域 %s，比例 %.4f'
                        % (args.pretrained_src_max_dis, dst_max_dis, dis_rescale))
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train,
                                    logger=logger,
                                    skip_patterns=args.pretrained_skip,
                                    dis_rescale=dis_rescale)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(
        f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # ---------------- 按验证集选轮（审查 P9：取最后一轮的比较受过拟合动态主导）----------------
    on_epoch_end = None
    if args.val_split:
        import copy
        import json
        from uavdet3d.utils import pose_eval
        val_cfg = copy.deepcopy(cfg.DATA_CONFIG)
        val_cfg.DATA_SPLIT['test'] = args.val_split
        val_cfg.SAMPLED_INTERVAL['test'] = args.val_interval
        # workers=0：不再多开 spawn 进程（这台机器 >=3 个带 worker 的进程会锁死）
        val_set, val_loader, _ = build_dataloader(dataset_cfg=val_cfg, batch_size=args.batch_size, dist=False,
                                                  workers=0, logger=logger, training=False)
        hist_path = output_dir / 'val_history.json'
        history = json.load(open(hist_path, encoding='utf-8')) if hist_path.exists() else []
        best = {'score': max([h['score'] for h in history], default=-1.0)}
        seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))
        logger.info('验证集选轮: split=%s 间隔 %d -> %d 帧，每 %d 轮验证，匹配 %s；已有记录 %d 条'
                    % (args.val_split, args.val_interval, len(val_set), args.val_every, args.val_match, len(history)))

        def on_epoch_end(trained_epoch, accumulated_iter, _model=model, _opt=optimizer):
            if trained_epoch % args.val_every != 0 and trained_epoch != args.epochs:
                return
            s, _ = pose_eval.evaluate(_model, val_loader, seq=seq, match=args.val_match)
            s['epoch'] = int(trained_epoch)
            history.append(s)
            json.dump(history, open(hist_path, 'w', encoding='utf-8'), indent=1)
            tag = '验证 第 %d 轮' % trained_epoch
            if s['score'] > best['score']:
                best['score'] = s['score']
                state = checkpoint_state(_model, _opt, trained_epoch, accumulated_iter)
                torch.save(state, str(ckpt_dir / 'best.pth'))
                json.dump({'epoch': int(trained_epoch), 'score': s['score']},
                          open(ckpt_dir / 'best.json', 'w', encoding='utf-8'))
                tag += ' [新最优，已存 best.pth]'
            logger.info(pose_eval.format_summary(tag, s))

    # ---------------- 损失权重按轮次爬升（2026-09-20）----------------
    # 动机：朝向与位置对训练的需求不同步。rot 权重一上来就拉高（R16）会伤位置（自标定 0.32 -> 0.44 m），
    # 但低权重下朝向学得太慢。让 rot 权重随轮次线性爬升：前期位置/深度先收敛，后期 OneCycle 学习率已经衰减，
    # 再把朝向推上去，对位置的扰动最小。配置：MODEL.DENSE_HEAD_2D.LOSS_WEIGHT_RAMP: {rot: [起点, 终点]}。
    ramp_cfg = dict((cfg.MODEL.get('DENSE_HEAD_2D', {}) or {}).get('LOSS_WEIGHT_RAMP', {}) or {})
    if ramp_cfg:
        head = (model.module if hasattr(model, 'module') else model).dense_head_2d
        base_hook = unfreeze_hook

        def unfreeze_hook(cur_epoch, _h=head, _r=ramp_cfg, _n=args.epochs, _lg=logger, _base=base_hook):
            if _base is not None:
                _base(cur_epoch)
            f = cur_epoch / max(_n - 1, 1)
            msg = []
            for k, (lo, hi) in _r.items():
                w = float(lo) + (float(hi) - float(lo)) * f
                _h.loss_weights[k] = w
                msg.append('%s %.2f' % (k, w))
            _lg.info('第 %d 轮 损失权重爬升: %s' % (cur_epoch, ', '.join(msg)))
        logger.info('损失权重爬升已启用: %s（按轮次线性，共 %d 轮）' % (ramp_cfg, args.epochs))

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # ---------------- 锚定蒸馏（2026-09-20，用户要求「保全位置的同时提升朝向」）----------------
    # 位置相关的头（hm / center_res / center_dis / dim / size2d）的输出被钉在教师（通常是位置最好的那一轮）上，
    # 旋转头与骨干继续学。教师走 eval（MixStyle 关），学生走 train（MixStyle 开），所以这一项同时是一致性正则。
    train_fn = model_fn_decorator()
    if args.anchor_ckpt:
        import torch.nn.functional as F
        teacher = build_network(model_cfg=cfg.MODEL, dataset=train_set)
        n_l, n_t = teacher.load_params_from_file(filename=args.anchor_ckpt, to_cpu=False, logger=logger)
        assert n_l == n_t, '教师权重没有全部载入'
        teacher.cuda().eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        anchor_keys = [k for k in args.anchor_keys.split(',') if k]
        logger.info('锚定蒸馏: 教师 %s，钉住 %s，权重 %.2f' % (args.anchor_ckpt, anchor_keys, args.anchor_weight))
        base_fn = train_fn

        def train_fn(model, batch, _b=base_fn, _t=teacher, _k=anchor_keys, _w=args.anchor_weight, _F=F):
            loss = _b(model, batch)                      # 基础 model_fn 只返回 loss
            head = (model.module if hasattr(model, 'module') else model).dense_head_2d
            sp = head.forward_loss_dict['pred_center_dict']
            with torch.no_grad():
                tb = {k: v for k, v in batch.items() if k != 'pred_center_dict'}
                for mod in _t.module_list:
                    tb = mod(tb)
                tp = tb['pred_center_dict']
            terms = {}
            fg = batch.get('fg_mask', None)
            if fg is not None:
                fg = fg.reshape(-1, 1, *fg.shape[-2:]).float()
            extra = 0.0
            for k in _k:
                if k not in sp or k not in tp:
                    continue
                a, b = sp[k].float(), tp[k].float()
                if k == 'hm':
                    term = _F.mse_loss(torch.sigmoid(a), torch.sigmoid(b))          # 整张图都钉，热图就是检测本身
                else:
                    d = (a - b) ** 2
                    term = (d * fg).sum() / fg.sum().clamp(min=1.0) / a.shape[1] if fg is not None else d.mean()
                extra = extra + term
                terms['anchor_' + k] = float(term.detach())
            head.loss_terms = dict(getattr(head, 'loss_terms', None) or {}, **terms)   # 进度条会显示
            return loss + _w * extra

    if args.da_cfg:
        from easydict import EasyDict as _ED
        import torch.nn as _nn
        import math as _math
        scfg = _ED()
        scfg.ROOT_DIR = cfg.ROOT_DIR
        scfg.LOCAL_RANK = cfg.LOCAL_RANK
        cfg_from_yaml_file(args.da_cfg, scfg)
        _, src_loader, _ = build_dataloader(dataset_cfg=scfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                            workers=args.da_workers, logger=logger, training=True,
                                            seed=args.seed + 7 if args.fix_random_seed else None)
        _net = model.module if hasattr(model, 'module') else model
        C = int(cfg.MODEL.BACKBONE_2D.OUT_CHANNELS)
        disc = _nn.Sequential(_nn.Conv2d(C, 256, 3, padding=1), _nn.ReLU(inplace=True),
                              _nn.Conv2d(256, 128, 3, padding=1), _nn.ReLU(inplace=True),
                              _nn.Conv2d(128, 1, 1)).cuda()
        dopt = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.9, 0.99))
        total_iters = max(1, args.epochs * len(train_loader))

        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lam):
                ctx.lam = lam
                return x.view_as(x)

            @staticmethod
            def backward(ctx, g):
                return -ctx.lam * g, None

        st = {'it': iter(src_loader), 'n': 0, 'acc': 0.0, 'cnt': 0}
        logger.info('整图特征对抗对齐已开启：源域配置 %s，%d 帧，权重 %.3f，域分类器在 features_2d（%d 通道）'
                    % (args.da_cfg, len(src_loader.dataset), args.da_weight, C))
        base_fn2 = train_fn

        def train_fn(model, batch, _b=base_fn2, _st=st, _F=torch.nn.functional):
            if _st['n'] > 0:                                # 上一步 backward 已给域分类器算好梯度
                _nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
                dopt.step()
            dopt.zero_grad()
            loss = _b(model, batch)
            ft = batch['features_2d']
            ft = ft.reshape(-1, *ft.shape[-3:])
            try:
                sb = next(_st['it'])
            except StopIteration:
                _st['it'] = iter(src_loader)
                sb = next(_st['it'])
            load_data_to_gpu(sb)
            bns = [m for m in _net.backbone_2d.modules() if isinstance(m, _nn.modules.batchnorm._BatchNorm)]
            modes = [m.training for m in bns]
            for m in bns:
                m.eval()                                    # 仿真前向不动 BN 统计
            fs = _net.backbone_2d({'image': sb['image']})['features_2d']
            for m, md in zip(bns, modes):
                m.train(md)
            fs = fs.reshape(-1, *fs.shape[-3:])
            p = min(1.0, _st['n'] / total_iters)
            lam = 2.0 / (1.0 + _math.exp(-10.0 * p)) - 1.0
            with torch.autocast('cuda', enabled=False):
                lt = disc(_GRL.apply(ft.float(), lam))
                ls = disc(_GRL.apply(fs.float(), lam))
                ld = 0.5 * (_F.binary_cross_entropy_with_logits(lt, torch.ones_like(lt)) +
                            _F.binary_cross_entropy_with_logits(ls, torch.zeros_like(ls)))
            with torch.no_grad():
                acc = 0.5 * ((lt > 0).float().mean() + (ls < 0).float().mean())
            _st['n'] += 1
            head = _net.dense_head_2d
            head.loss_terms = dict(getattr(head, 'loss_terms', None) or {}, da=float(ld.detach()), da_acc=float(acc),
                                   da_lam=lam)
            return loss + args.da_weight * ld

    if args.src_cfg:
        # L_source（11-03 老师方案）：每步再取一个仿真 batch 走同一个检测损失。仿真旋转目标是 12 通道（对称两解），
        # 真实配置的 rot 头是 6 通道且自训练 rot 权重 0，所以只截前 6 通道让形状对上（数值不进总损失）。
        from easydict import EasyDict as _ED
        _scfg = _ED()
        _scfg.ROOT_DIR = cfg.ROOT_DIR
        _scfg.LOCAL_RANK = cfg.LOCAL_RANK
        cfg_from_yaml_file(args.src_cfg, _scfg)
        _, _src_loader, _ = build_dataloader(dataset_cfg=_scfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                             workers=args.src_workers, logger=logger, training=True,
                                             seed=args.seed + 13 if args.fix_random_seed else None)
        _sst = {'it': iter(_src_loader)}
        _sn = model.module if hasattr(model, 'module') else model
        logger.info('L_source 已开启：仿真 %s，%d 帧，权重 %.3f' % (args.src_cfg, len(_src_loader.dataset), args.src_weight))
        base_fn_s = train_fn

        def train_fn(model, batch, _b=base_fn_s, _st=_sst):
            loss = _b(model, batch)
            terms = dict(getattr(_sn.dense_head_2d, 'loss_terms', None) or {})
            try:
                sb = next(_st['it'])
            except StopIteration:
                _st['it'] = iter(_src_loader)
                sb = next(_st['it'])
            if 'rot' in sb and sb['rot'].shape[-3] > 6:
                sb['rot'] = sb['rot'][..., :6, :, :]
            ls = _b(model, sb)
            terms['src'] = float(ls.detach())
            _sn.dense_head_2d.loss_terms = terms
            return loss + args.src_weight * ls

    if args.proto_bank:
        _pn = model.module if hasattr(model, 'module') else model
        PROTO_B = torch.load(args.proto_bank).float().cuda().detach()
        logger.info('原型库载入：%s，%d 个原型 × %d 维，距离 %s，权重 %.3f' % (args.proto_bank, PROTO_B.shape[0], PROTO_B.shape[1],
                                                                  args.proto_dist, args.proto_weight))
        base_fn4 = train_fn

        def train_fn(model, batch, _b=base_fn4, _P=PROTO_B, _F=torch.nn.functional):
            loss = _b(model, batch)
            ft = batch['features_2d']
            ft = ft.reshape(-1, *ft.shape[-3:])
            fg = batch['fg_mask'].reshape(ft.shape[0], -1, *ft.shape[-2:]) > 0
            bi, _, yy, xx = torch.nonzero(fg, as_tuple=True)
            if len(bi) == 0:
                return loss
            head = _pn.dense_head_2d
            with torch.autocast('cuda', enabled=False):
                z = ft[bi, :, yy, xx].float()
                if args.proto_dist == 'l2':
                    d = torch.cdist(z, _P)                     # (M, K)
                    lp = d.min(1).values.mean()
                    sim = _F.cosine_similarity(z[:, None], _P[None], dim=-1).max(1).values.mean()
                else:
                    zn, Pn = _F.normalize(z, dim=1), _F.normalize(_P, dim=1)
                    lg = zn @ Pn.T / args.proto_temp
                    lp = _F.cross_entropy(lg, lg.argmax(1).detach())
                    sim = (zn @ Pn.T).max(1).values.mean()
            head.loss_terms = dict(getattr(head, 'loss_terms', None) or {}, proto=float(lp.detach()), proto_sim=float(sim.detach()))
            return loss + args.proto_weight * lp

    if args.proto_cfg and not args.proto_bank:
        from easydict import EasyDict as _ED
        pcfg = _ED()
        pcfg.ROOT_DIR = cfg.ROOT_DIR
        pcfg.LOCAL_RANK = cfg.LOCAL_RANK
        cfg_from_yaml_file(args.proto_cfg, pcfg)
        if args.proto_clean:
            pcfg.DATA_CONFIG.AUG = _ED({'hflip': 0.5, 'scale': [1.0, 1.0], 'photometric': False, 'noise': 0.0, 'blur': 0.0, 'blur_prob': 0.5})
            logger.info('原型用干净仿真图建：AUG=%s，VIEW_AUG 不变' % dict(pcfg.DATA_CONFIG.AUG))
        _, ploader, _ = build_dataloader(dataset_cfg=pcfg.DATA_CONFIG, batch_size=16, dist=False, workers=4, logger=logger,
                                         training=True, seed=args.seed + 11 if args.fix_random_seed else None)
        _pn = model.module if hasattr(model, 'module') else model
        _was = _pn.training
        _pn.eval()
        feats, n = [], 0
        with torch.no_grad():
            for pb in ploader:
                load_data_to_gpu(pb)
                with torch.autocast('cuda'):
                    f = _pn.backbone_2d({'image': pb['image']})['features_2d']
                f = f.float().reshape(-1, *f.shape[-3:])
                fg = pb['fg_mask'].reshape(f.shape[0], -1, *f.shape[-2:]) > 0
                bi, _, yy, xx = torch.nonzero(fg, as_tuple=True)
                feats.append(torch.nn.functional.normalize(f[bi, :, yy, xx], dim=1))
                n += pb['image'].shape[0]
                if n >= args.proto_nsim:
                    break
        _pn.train(_was)
        del ploader
        X = torch.cat(feats)
        _g = torch.Generator(device='cpu').manual_seed(args.seed)
        PROTO = X[torch.randperm(len(X), generator=_g)[:args.proto_k].to(X.device)].clone()
        for _ in range(50):
            asg = (X @ PROTO.T).argmax(1)
            for j in range(args.proto_k):
                m = asg == j
                if m.any():
                    PROTO[j] = torch.nn.functional.normalize(X[m].mean(0), dim=0)
        PROTO = PROTO.detach()
        cnt = torch.bincount((X @ PROTO.T).argmax(1), minlength=args.proto_k)
        logger.info('原型 loss 已开启：%d 帧仿真、%d 个目标格 → %d 个原型（每类 %s），权重 %.3f，温度 %.2f'
                    % (n, len(X), args.proto_k, cnt.tolist(), args.proto_weight, args.proto_temp))
        torch.save(PROTO.cpu(), str(output_dir / 'prototypes.pt'))
        base_fn3 = train_fn

        def train_fn(model, batch, _b=base_fn3, _P=PROTO, _F=torch.nn.functional):
            loss = _b(model, batch)
            ft = batch['features_2d']
            ft = ft.reshape(-1, *ft.shape[-3:])
            fg = batch['fg_mask'].reshape(ft.shape[0], -1, *ft.shape[-2:]) > 0
            if args.proto_min_dis > 0:
                gb = torch.as_tensor(batch['gt_box9d'], device=ft.device).float()
                dd = torch.linalg.norm(gb[..., :3], dim=-1)
                dd = torch.where(dd > 0, dd, torch.full_like(dd, float('inf'))).min(1).values      # 每帧最近目标的伪距离
                keep = (dd >= args.proto_min_dis) & torch.isfinite(dd)
                fg = fg & keep.repeat_interleave(ft.shape[0] // keep.shape[0])[:, None, None, None]
            bi, _, yy, xx = torch.nonzero(fg, as_tuple=True)
            head = _pn.dense_head_2d
            if len(bi) == 0:
                return loss
            with torch.autocast('cuda', enabled=False):
                z = _F.normalize(ft[bi, :, yy, xx].float(), dim=1)
                lg = z @ _P.T / args.proto_temp
                lp = _F.cross_entropy(lg, lg.argmax(1).detach())
            head.loss_terms = dict(getattr(head, 'loss_terms', None) or {}, proto=float(lp.detach()),
                                   proto_sim=float((z @ _P.T).max(1).values.mean().detach()))
            return loss + args.proto_weight * lp

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=train_fn,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        logger=logger,
        log_interval=args.logger_iter_interval,
        on_epoch_start=unfreeze_hook,
        on_epoch_end=on_epoch_end,
        use_amp=args.use_amp,
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    if args.skip_test_eval:
        return

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval,
                           0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()