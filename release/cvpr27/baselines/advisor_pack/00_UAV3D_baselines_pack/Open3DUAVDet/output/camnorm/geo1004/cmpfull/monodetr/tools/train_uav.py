# -*- coding: utf-8 -*-
"""[UAV port] MonoDETR training entry (replaces tools/train_val.py for the cmpfull data; same model/optimizer/scheduler/Trainer).

python tools/train_uav.py --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl [more.pkl[@img_root] ...] --out <dir> --seed 444 --epochs 37 --batch-size 2
  --batch-size B : official 16. If B != 16 the lr is scaled linearly (2e-4 * B / 16).
  --epochs E     : official 195 epochs on KITTI train (3,712) = 723,840 samples; 37 epochs x 19,490 sim_train = 721,130.
                   decay epochs [125, 165] are rescaled proportionally (E=37 -> [24, 31]).
  --max-iters N  : smoke test only (stop after N iterations).
GPU memory cap: torch.cuda.set_per_process_memory_fraction(GPU_FRAC, default 0.2).
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import datetime
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
import yaml

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.utils_helper import create_logger, set_random_seed

OFFICIAL_EPOCHS, OFFICIAL_BS = 195, 16  # configs/monodetr.yaml

p = argparse.ArgumentParser()
p.add_argument('--config', default=os.path.join(ROOT_DIR, 'configs', 'monodetr_uav.yaml'))
p.add_argument('--labels', nargs='+', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=444)
p.add_argument('--epochs', type=int, default=37)
p.add_argument('--batch-size', type=int, default=2)
p.add_argument('--workers', type=int, default=2)
p.add_argument('--resume', action='store_true', help='从 <out>/checkpoint.pth 接着训（官方 resume_model）')
p.add_argument('--max-iters', type=int, default=0)
p.add_argument('--keep-every', type=int, default=4, help='also save model-only snapshot every N epochs (0=off)')
args = p.parse_args()


def main():
    os.environ.setdefault('TORCH_HOME', '@@ROOT@@/torch_home')  # ImageNet resnet50-0676ba61.pth is there; never download
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.2')))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['random_seed'] = args.seed
    cfg['dataset']['label_files'] = args.labels
    cfg['dataset']['batch_size'] = args.batch_size
    cfg['optimizer']['lr'] = cfg['optimizer']['lr'] * args.batch_size / OFFICIAL_BS  # linear scaling rule
    cfg['trainer']['max_epoch'] = args.epochs
    cfg['lr_scheduler']['decay_list'] = [int(round(d * args.epochs / OFFICIAL_EPOCHS)) for d in cfg['lr_scheduler']['decay_list']]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    cfg['trainer']['save_path'] = out
    cfg['trainer']['max_iters'] = args.max_iters
    if args.resume:
        cfg['trainer']['resume_model'] = True
    cfg['trainer']['keep_every'] = args.keep_every
    cfg['trainer']['loss_log'] = os.path.join(out, 'loss_log.jsonl')
    yaml.safe_dump(cfg, open(os.path.join(out, 'config_used.yaml'), 'w'), allow_unicode=True)

    set_random_seed(cfg.get('random_seed', 444))
    logger = create_logger(os.path.join(out, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    logger.info(json.dumps(vars(args)))

    train_loader, _ = build_dataloader(cfg['dataset'], workers=args.workers)
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    trainer = Trainer(cfg=cfg['trainer'], model=model, optimizer=optimizer, train_loader=train_loader, test_loader=None,
                      lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler, logger=logger, loss=loss,
                      model_name=cfg['model_name'])
    logger.info('###################  Training  ##################')
    logger.info('Train frames: %d, Batch Size: %d, Learning Rate: %f, epochs %d, decay %s' % (
        len(train_loader.dataset), args.batch_size, cfg['optimizer']['lr'], args.epochs, cfg['lr_scheduler']['decay_list']))
    trainer.train()
    if torch.cuda.is_available():
        logger.info('max_memory_allocated %.2f GB, max_reserved %.2f GB' % (
            torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30))


if __name__ == '__main__':
    main()
