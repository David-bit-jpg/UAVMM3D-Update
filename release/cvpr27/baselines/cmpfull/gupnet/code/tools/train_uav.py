# -*- coding: utf-8 -*-
"""[UAV] 训练入口（由 tools/train_val.py 改写）。
python tools/train_uav.py --labels D:/cmpfull/sim_train/labels.pkl [更多 pkl ...] --out <dir> --seed 444 --epochs 140
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from lib.datasets.uav import UAV, load_records, mean_size_from_records
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.dataloader_helper import my_worker_init_fn

parser = argparse.ArgumentParser(description='GUPNet on UAV data')
parser.add_argument('--labels', nargs='+', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--seed', type=int, default=444)
parser.add_argument('--epochs', type=int, default=140, help='"轮" = epoch_size 个样本；默认 140 = 官方')
parser.add_argument('--epoch-size', type=int, default=None, help='每轮样本数，默认 config 里的 3712（官方 KITTI train 大小）；<=0 表示整个数据集')
parser.add_argument('--batch-size', type=int, default=None, help='默认官方 32；改小时学习率按线性规则缩放')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--mean-size-from', default='D:/cmpfull/sim_train/labels.pkl', help='类均值尺寸统计来源（单类，sim_train）')
parser.add_argument('--resume', default=None, help='从 checkpoint 接着训（官方 trainer.resume_model）')
parser.add_argument('--max-iters', type=int, default=None, help='仅冒烟测试：总迭代上限')
parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'experiments/config_uav.yaml'))
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.2')))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg['random_seed'] = args.seed

    B = args.batch_size or cfg['dataset']['batch_size']
    lr = args.lr if args.lr is not None else cfg['optimizer']['lr'] * B / float(cfg['dataset']['batch_size'])
    cfg['dataset']['batch_size'] = B
    cfg['optimizer']['lr'] = lr
    E = args.epochs
    # 官方日程按 140 轮写死；改轮数时衰减点按比例缩放（warmup 5 轮、HTL 窗口 5 轮保持官方）
    cfg['lr_scheduler']['decay_list'] = [int(round(d * E / 140.0)) for d in cfg['lr_scheduler']['decay_list']]
    cfg['trainer']['max_epoch'] = E
    cfg['trainer']['log_dir'] = args.out
    if args.resume:
        cfg['trainer']['resume_model'] = args.resume
    if args.max_iters is not None:
        cfg['trainer']['max_iters'] = args.max_iters
        cfg['trainer']['e0_iters'] = max(1, min(10, args.max_iters))
    os.makedirs(args.out, exist_ok=True)
    logger = create_logger(os.path.join(args.out, 'train.log'))

    mean_size = mean_size_from_records(load_records([args.mean_size_from]))
    train_set = UAV(args.labels, 'train', cfg['dataset'], mean_size=mean_size)
    es = args.epoch_size if args.epoch_size is not None else cfg['dataset'].get('epoch_size', 0)
    if es is None or es <= 0 or es >= len(train_set):
        sampler = RandomSampler(train_set)
        es = len(train_set)
    else:
        sampler = RandomSampler(train_set, replacement=False, num_samples=es)
    train_loader = DataLoader(dataset=train_set, batch_size=B, num_workers=args.workers,
                              worker_init_fn=my_worker_init_fn, sampler=sampler,
                              pin_memory=True, drop_last=False, persistent_workers=args.workers > 0)
    run_info = dict(args=vars(args), batch_size=B, lr=lr, epoch_size=es, n_train=len(train_set),
                    mean_size_hwl=mean_size.tolist(), decay_list=cfg['lr_scheduler']['decay_list'],
                    pretrained=os.environ.get('GUPNET_NO_PRETRAIN', '0') != '1')
    logger.info(json.dumps(run_info, ensure_ascii=False))
    json.dump(run_info, open(os.path.join(args.out, 'run_info.json'), 'w'), ensure_ascii=False, indent=1)

    model = build_model(cfg['model'], mean_size)
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    cfg['extra_state'] = dict(mean_size=mean_size, cfg=cfg['dataset'],
                              model_cfg=cfg['model'])
    trainer = Trainer(cfg=cfg, model=model, optimizer=optimizer, train_loader=train_loader, test_loader=None,
                      lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler, logger=logger)
    trainer.train()


if __name__ == '__main__':
    main()
