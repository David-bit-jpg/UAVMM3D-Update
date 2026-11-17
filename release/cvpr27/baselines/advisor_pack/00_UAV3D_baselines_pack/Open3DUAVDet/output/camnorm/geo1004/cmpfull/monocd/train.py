# -*- coding: utf-8 -*-
"""[UAV port] training entry for MonoCD on the cmpfull UAV data.

Mirrors tools/plain_train_net.py::train + engine/trainer.py::do_train (same optimizer/scheduler builders, same
iteration loop: loss = sum(loss_dict), zero_grad, backward, step, scheduler.step(iteration), epoch-based
MAX_ITERATION/STEPS conversion), but replaces the KITTI dataset/evaluation with the UAV dataset and saves a
checkpoint every epoch (KITTI AP evaluation is not applicable).

  python train.py --labels @@ROOT@@/data/cmpfull/sim_train/labels.pkl [more.pkl ...] --out OUT --seed 0 --epochs 19
"""
import os
import sys
import time
import json
import argparse
import datetime
import logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import numpy as np


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=None, help='default: MAX_EPOCHS of the yaml (19, see NOTES)')
    p.add_argument('--config', default=os.path.join(HERE, 'runs', 'monocd_uav.yaml'))
    p.add_argument('--batch', type=int, default=None, help='default: official 8. If changed, BASE_LR is scaled linearly')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-iters', type=int, default=None, help='stop after N iterations (smoke test)')
    p.add_argument('--no-pretrain', action='store_true', help='do not load ImageNet DLA-34 (smoke test only)')
    p.add_argument('--resume', action='store_true', help='resume from OUT/model_checkpoint.pth')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('opts', nargs=argparse.REMAINDER, help='extra yacs overrides KEY VALUE ...')
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)
    frac = float(os.environ.get('GPU_FRAC', '0.2'))
    torch.cuda.set_per_process_memory_fraction(frac)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.out, 'log.txt'), encoding='utf-8')])
    logger = logging.getLogger('monocd.trainer')

    from config import cfg
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.UAV.LABELS = list(args.labels)
    cfg.OUTPUT_DIR = args.out
    cfg.SEED = args.seed
    cfg.DATALOADER.NUM_WORKERS = args.workers
    if args.epochs is not None:
        # keep the official decay points at 80% / 90% of training
        cfg.SOLVER.MAX_EPOCHS = args.epochs
        cfg.SOLVER.DECAY_EPOCH_STEPS = [int(round(0.8 * args.epochs)), int(round(0.9 * args.epochs))]
    if args.batch is not None and args.batch != cfg.SOLVER.IMS_PER_BATCH:
        cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * args.batch / cfg.SOLVER.IMS_PER_BATCH  # linear scaling rule
        cfg.SOLVER.IMS_PER_BATCH = args.batch
    if args.no_pretrain:
        cfg.MODEL.PRETRAIN = False
    elif cfg.MODEL.PRETRAIN and not os.environ.get('DLA34_PTH'):
        logger.warning('MODEL.PRETRAIN=True but DLA34_PTH is not set: the official code will try to download '
                       'dla34-ba72cf86.pth from dl.yf.io')
    cfg.MODEL.USE_SYNC_BN = False  # single GPU (official main() does the same when not distributed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    from utils.envs import seed_all_rng
    seed_all_rng(args.seed)

    from data.datasets.uav import UAVDataset
    from data.transforms import build_transforms
    from data.samplers import TrainingSampler
    from data.collate_batch import BatchCollator
    from data.build import worker_init_reset_seed
    from model.detector import KeypointDetector
    from solver import build_optimizer, build_scheduler

    dataset = UAVDataset(cfg, cfg.UAV.LABELS, is_train=True, transforms=build_transforms(cfg, True), augment=True)
    bs = cfg.SOLVER.IMS_PER_BATCH
    sampler = TrainingSampler(len(dataset), seed=args.seed)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, bs, drop_last=True)
    loader = torch.utils.data.DataLoader(dataset, num_workers=args.workers, batch_sampler=batch_sampler,
                                         collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), pin_memory=True,
                                         worker_init_fn=worker_init_reset_seed, persistent_workers=args.workers > 0)

    total_iters_each_epoch = len(dataset) // bs
    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
        cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * total_iters_each_epoch
        cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = total_iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
        cfg.SOLVER.STEPS = [total_iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
        cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * total_iters_each_epoch
    cfg.freeze()
    with open(os.path.join(args.out, 'config.yaml'), 'w', encoding='utf-8') as f:
        f.write(cfg.dump())

    device = torch.device(cfg.MODEL.DEVICE)
    model = KeypointDetector(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_scheduler = build_scheduler(optimizer, total_iters_each_epoch=total_iters_each_epoch, optim_cfg=cfg.SOLVER)

    start_iter = 0
    ck_path = os.path.join(args.out, 'model_checkpoint.pth')
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_iter = ck['iteration']
        logger.info('resumed from %s @ iteration %d' % (ck_path, start_iter))

    max_iter = cfg.SOLVER.MAX_ITERATION
    if args.max_iters is not None:
        max_iter = min(max_iter, start_iter + args.max_iters)
    warmup_iters = cfg.SOLVER.WARMUP_STEPS if cfg.SOLVER.LR_WARMUP else -1
    grad_norm_clip = cfg.SOLVER.GRAD_NORM_CLIP
    logger.info('frames %d, batch %d, iters/epoch %d, epochs %d, max_iter %d (this run stops at %d), lr %g, decay at %s'
                % (len(dataset), bs, total_iters_each_epoch, cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.MAX_ITERATION, max_iter,
                   cfg.SOLVER.BASE_LR, list(cfg.SOLVER.STEPS)))

    def save(name, iteration):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'iteration': iteration, 'iter_per_epoch': total_iters_each_epoch}, os.path.join(args.out, name + '.pth'))

    model.train()
    hist = []
    t0 = time.time(); end = time.time(); torch.cuda.reset_peak_memory_stats()
    win = []
    for data, iteration in zip(loader, range(start_iter, max_iter)):
        data_time = time.time() - end
        images = data["images"].to(device)
        targets = [target.to(device) for target in data["targets"]]
        loss_dict, log_loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        if torch.isfinite(losses):
            losses.backward()
            if grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()
        else:  # [UAV] 损失非有限（官方会进 pdb）：跳过这一步的参数更新，学习率日程照走
            logger.warning('iter %d 损失非有限 %s，跳过本步更新' % (iteration, {k: float(v) for k, v in loss_dict.items()}))
        if iteration < warmup_iters:
            warmup_scheduler.step(iteration)
        else:
            scheduler.step(iteration)
        batch_time = time.time() - end
        end = time.time()
        iteration += 1
        if iteration == start_iter + 5:
            torch.cuda.reset_peak_memory_stats()  # exclude cudnn.benchmark autotuning workspaces from the peak
        rec = {'iter': iteration, 'loss': float(losses.detach()), 'time': batch_time, 'data': data_time,
               'lr': optimizer.param_groups[0]['lr']}
        rec.update({k: float(v) for k, v in log_loss_dict.items()})
        hist.append(rec); win.append(rec)
        if iteration % args.log_every == 0 or iteration == max_iter:
            m = {k: np.mean([w[k] for w in win if k in w]) for k in win[-1]}
            eta = (time.time() - t0) / max(iteration - start_iter, 1) * (cfg.SOLVER.MAX_ITERATION - iteration)
            logger.info('iter %d/%d loss %.4f | %s | %.3fs/it (data %.3f) lr %.2e mem %.2f GB eta %s' % (
                iteration, cfg.SOLVER.MAX_ITERATION, m['loss'],
                ' '.join('%s %.3f' % (k, m[k]) for k in m if k not in ('iter', 'loss', 'time', 'data', 'lr')),
                m['time'], m['data'], m['lr'], torch.cuda.max_memory_allocated() / 2 ** 30,
                str(datetime.timedelta(seconds=int(eta)))))
            win = []
        if iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0:
            save('model_checkpoint', iteration)
            logger.info('epoch %d done, checkpoint saved' % (iteration // total_iters_each_epoch))
        if iteration == cfg.SOLVER.MAX_ITERATION:
            save('model_final', iteration)
    if args.max_iters is not None:
        save('model_smoke', max_iter)
    json.dump(hist, open(os.path.join(args.out, 'train_hist.json'), 'w'))
    logger.info('done: %d iters in %.1fs, peak mem %.2f GB' % (max_iter - start_iter, time.time() - t0,
                                                               torch.cuda.max_memory_allocated() / 2 ** 30))


if __name__ == '__main__':
    main()
