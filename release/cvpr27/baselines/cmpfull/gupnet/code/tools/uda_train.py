# -*- coding: utf-8 -*-
"""[UDA] Two classic sim->real UDA baselines on top of the GUPNet port (new entry; train_uav.py is untouched).

  --method mt   : mean-teacher self-training (STMono3D-style, Li et al. ECCV'22).
                  teacher = EMA(student); teacher (eval mode, weak view = no flip/crop/jitter) predicts each target image,
                  top-1 box kept if score >= tau (tau fixed once = (1-keep) quantile of the *source model's* top-1 scores on the
                  unlabeled target train images; no real label is ever read); student is trained on a strongly augmented view
                  (dataset flip/crop/scale/shift + colour jitter) with the pseudo box pushed through the dataset's own label code.
                  Final prediction model = teacher.
  --method dann : adversarial feature alignment (Ganin & Lempitsky ICML'15). Domain classifier on the DLAUp neck feature map
                  (stride 4, 64 ch) through a gradient reversal layer; source batch: GT detection loss + domain loss (label 0),
                  target batch: domain loss only (label 1).

Both start from the source-only checkpoint, constant lr, fixed number of epochs, final checkpoint used (no model selection).
Real target records are loaded with every label field stripped (only file/K/W/H/seq/frame kept).

Exit code 3 = CUDA OOM (the launcher halves the batch and resumes from checkpoints/checkpoint_latest.pth).
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import argparse
import copy
import json
import logging
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, default_collate

import lib.datasets.uav as uavmod
from lib.datasets.uav import UAV, load_records, IMG_H_VALID, NET_W
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, get_heading_angle
from lib.helpers.dataloader_helper import my_worker_init_fn
from lib.losses.loss_function import GupnetLoss
from lib.datasets.uav_utils import code_angles_to_ypr

ap = argparse.ArgumentParser(description='[UDA] GUPNet mean-teacher / DANN')
ap.add_argument('--method', required=True, choices=['mt', 'dann'])
ap.add_argument('--init', default='D:/cmpfull_runs/gupnet_s0/checkpoints/checkpoint_final.pth', help='source-only checkpoint')
ap.add_argument('--src-labels', nargs='+', default=['D:/cmpfull/sim_train/labels.pkl'])
ap.add_argument('--tgt-labels', default='D:/cmpfull/real_train/labels.pkl', help='unlabeled target images (labels stripped on load)')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--epoch-size', type=int, default=3712, help='source samples per epoch (same definition as train_uav.py)')
ap.add_argument('--batch-size', type=int, default=8, help='source batch = target batch')
ap.add_argument('--base-lr', type=float, default=3.125e-5, help='lr at batch 8 (= source-only lr 3.125e-4 x 0.1); scaled linearly with batch')
ap.add_argument('--weight-decay', type=float, default=1e-5)
ap.add_argument('--workers', type=int, default=0, help='0 = no worker processes; data loaded by --threads threads instead')
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--resume', default=None, help="'auto' = resume from <out>/checkpoints/checkpoint_latest.pth if it exists")
ap.add_argument('--max-iters', type=int, default=None, help='smoke only: total iteration cap')
ap.add_argument('--tau-frames', type=int, default=None, help='smoke only: use the first N target frames to fix tau')
ap.add_argument('--log-every', type=int, default=50)
# mean teacher
ap.add_argument('--ema', type=float, default=0.999)
ap.add_argument('--keep', type=float, default=0.8, help='tau = (1-keep) quantile of source-model top-1 scores on target train (same 80%% rule as +ST)')
ap.add_argument('--tau', type=float, default=None, help='fixed absolute threshold instead of the quantile rule')
ap.add_argument('--tgt-weight', type=float, default=1.0)
ap.add_argument('--jitter', type=float, nargs=4, default=[0.4, 0.4, 0.4, 0.05], help='strong view colour jitter b c s h')
ap.add_argument('--quality-weight', action='store_true', help='scale target loss by mean teacher score of the kept boxes (off by default)')
# dann
ap.add_argument('--grl-max', type=float, default=0.1, help='GRL coefficient max; ramp lambda_p = max*(2/(1+exp(-10p))-1)')
ap.add_argument('--disc-lr-mult', type=float, default=10.0, help='domain classifier lr = mult x lr (DANN: 10x for new layers)')
ap.add_argument('--bn', default='frozen', choices=['frozen', 'train'], help='BatchNorm statistics during fine-tuning (default frozen = source stats)')
ap.add_argument('--config', default=os.path.join(ROOT_DIR, 'experiments/config_uav.yaml'))
args = ap.parse_args()

TGT_KEEP_KEYS = ('file', 'root', 'K', 'W', 'H', 'seq', 'frame')


# ----------------------------------------------------------------------------- utils
def create_logger(log_file):
    lg = logging.getLogger('uda')
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    for h in (logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg


def load_target_records(path):
    """[UDA] target records with ALL label content removed: boxes = [] always."""
    recs = []
    for r in load_records([path]):
        d = {k: r[k] for k in TGT_KEEP_KEYS if k in r}
        d['boxes'] = []
        recs.append(d)
    return recs


_orig_load_resized = uavmod.load_resized


def _patched_load_resized(rec):
    """[UDA] lets the train-mode dataset reuse an already-loaded weak image (pseudo-labelled target frame) and adds
    colour jitter on the valid 960x540 region for the strong view. Other records go through the original loader."""
    if '_pil' not in rec:
        return _orig_load_resized(rec)
    img = rec['_pil'].copy()
    if rec.get('_jitter') is not None:
        region = rec['_jitter'](img.crop((0, 0, NET_W, IMG_H_VALID)))
        img.paste(region, (0, 0))
    return img, rec['_K'].copy()


uavmod.load_resized = _patched_load_resized

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def tensor_to_pil(t):
    """normalised CHW tensor (from uint8 image) -> exact uint8 PIL."""
    a = t.detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return Image.fromarray(np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8))


def forward_feat(m, x, coord_ranges, calibs, targets=None, K=50, mode='train'):
    """= GUPNet.forward, additionally returning the neck feature map (for the DANN domain classifier)."""
    feat = m.backbone(x)
    feat = m.feat_up(feat[m.first_level:])
    ret = {'heatmap': m.heatmap(feat), 'offset_2d': m.offset_2d(feat), 'size_2d': m.size_2d(feat)}
    if mode == 'train':
        inds, cls_ids, masks = targets['indices'], targets['cls_ids'], targets['mask_2d']
    else:
        raise NotImplementedError
    ret.update(m.get_roi_feat(feat, inds, masks, ret, calibs, coord_ranges, cls_ids))
    return ret, feat


def decode_top1(dets, info, calibs, mean_size):
    """= decode_detections_uav, but returns per image the top-1 (center, lwh, ypr, score) (ypr kept for the dataset)."""
    out = []
    for i in range(dets.shape[0]):
        best = None
        for j in range(dets.shape[1]):
            score = float(dets[i, j, 1])
            if best is not None and score <= best[3]:
                continue
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            depth = dets[i, j, 6]
            angles = np.array([get_heading_angle(dets[i, j, 7 + 24 * k: 7 + 24 * (k + 1)]) for k in range(3)])
            ypr = code_angles_to_ypr(angles, x, calibs[i])
            dims = dets[i, j, 79:82].copy() + mean_size[int(dets[i, j, 0])]
            if True in (dims < 0.0):
                continue
            x3d = dets[i, j, 82] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 83] * info['bbox_downsample_ratio'][i][1]
            loc = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            if not (np.all(np.isfinite(loc)) and np.all(np.isfinite(ypr)) and loc[2] > 0.05):
                continue
            best = (loc.astype(np.float64), dims[::-1].astype(np.float64), np.asarray(ypr, np.float64), score)
        out.append(best)
    return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainClassifier(nn.Module):
    """small conv head (3 x conv3x3 stride 2, no BN since every batch is single-domain) + global avg pool + MLP -> 1 logit."""
    def __init__(self, cin=64, c=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, c, 3, 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(nn.Linear(c, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, f):
        return self.mlp(self.conv(f).mean((2, 3))).squeeze(1)


@torch.no_grad()
def ema_update(teacher, student, m):
    sp = dict(student.named_parameters())
    for n, p in teacher.named_parameters():
        p.mul_(m).add_(sp[n].detach(), alpha=1 - m)
    sb = dict(student.named_buffers())
    for n, b in teacher.named_buffers():
        if b.dtype.is_floating_point:
            b.mul_(m).add_(sb[n].detach(), alpha=1 - m)
        else:
            b.copy_(sb[n])


def freeze_bn_stats(m):
    """[UDA] BatchNorm layers use the source running statistics (eval mode); affine weights still train.
    Reason (measured, see UDA_NOTES.md): with train-mode BN on single-domain target batches the student's output on the
    *identical, unaugmented* target image differs from the eval-mode teacher by a median 2.6 m in depth, so the pseudo-label
    loss would mostly fit BN-statistics noise. Frozen BN is also the usual setting for small-batch detector fine-tuning
    (Detectron2 FrozenBN; DA-Faster R-CNN implementations)."""
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def to_dev(batch, dev):
    inputs, calibs, coord_ranges, targets, info = batch
    inputs, calibs, coord_ranges = inputs.to(dev), calibs.to(dev), coord_ranges.to(dev)
    targets = {k: v.to(dev) for k, v in targets.items()}
    if 'mask_2d' in targets:
        targets['mask_2d'] = targets['mask_2d'].bool()
    return inputs, calibs, coord_ranges, targets, info


def finite(x):
    return bool(torch.isfinite(x).all())


class ThreadBatchLoader(object):
    """[UDA] DataLoader replacement with a thread pool (no spawned worker processes: Windows locks up with >=3 spawn
    trainings, see gpu_slot.sh). PIL decode/resize/affine and numpy release the GIL, so 6 threads give ~DataLoader(workers=2+)
    throughput. order: 'random' = fresh permutation per pass (= RandomSampler without replacement, first num_samples),
    'seq' = 0..n-1. Batches are prefetched `depth` batches ahead."""
    def __init__(self, ds, batch_size, pool, order='random', num_samples=None, drop_last=False, depth=3, indices=None):
        self.ds, self.B, self.pool, self.order, self.drop_last, self.depth = ds, batch_size, pool, order, drop_last, depth
        self.indices = list(range(len(ds))) if indices is None else list(indices)
        self.n = len(self.indices) if num_samples is None else min(num_samples, len(self.indices))

    def __len__(self):
        return self.n // self.B if self.drop_last else (self.n + self.B - 1) // self.B

    def __iter__(self):
        if self.order == 'random':
            perm = torch.randperm(len(self.indices)).tolist()[:self.n]
            idx = [self.indices[k] for k in perm]
        else:
            idx = self.indices[:self.n]
        chunks = [idx[k:k + self.B] for k in range(0, len(idx), self.B)]
        if self.drop_last and chunks and len(chunks[-1]) < self.B:
            chunks = chunks[:-1]
        q = []
        for c in chunks:
            q.append([self.pool.submit(self.ds.__getitem__, i) for i in c])
            if len(q) > self.depth:
                yield default_collate([f.result() for f in q.pop(0)])
        while q:
            yield default_collate([f.result() for f in q.pop(0)])


def save_atomic(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main
def main():
    os.environ.setdefault('GUPNET_NO_PRETRAIN', '1')  # all weights come from the source checkpoint
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.35')))
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    logger = create_logger(os.path.join(args.out, 'train.log'))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    dcfg = cfg['dataset']
    B = args.batch_size
    lr = args.base_lr * B / 8.0
    dcfg['batch_size'] = B

    src_ck = torch.load(args.init, map_location='cpu', weights_only=False)
    mean_size = np.asarray(src_ck['mean_size'], np.float32)
    model_cfg = src_ck.get('model_cfg', cfg['model'])

    # ---- data
    src_set = UAV(args.src_labels, 'train', dcfg, mean_size=mean_size)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max(1, args.threads))
    if args.workers > 0:   # original multiprocess DataLoader path (not used by the launcher)
        src_loader = DataLoader(src_set, batch_size=B, num_workers=args.workers, worker_init_fn=my_worker_init_fn,
                                sampler=RandomSampler(src_set, replacement=False, num_samples=args.epoch_size),
                                pin_memory=True, drop_last=False)
    else:
        src_loader = ThreadBatchLoader(src_set, B, pool, 'random', num_samples=args.epoch_size)
    tgt_recs = load_target_records(args.tgt_labels)
    assert all(len(r['boxes']) == 0 for r in tgt_recs)
    tgt_weak = UAV(None, 'test', dcfg, mean_size=mean_size, records=tgt_recs)   # no augmentation, no labels
    if args.workers > 0:
        tgt_loader = DataLoader(tgt_weak, batch_size=B, num_workers=args.workers, sampler=RandomSampler(tgt_weak),
                                pin_memory=True, drop_last=True)
    else:
        tgt_loader = ThreadBatchLoader(tgt_weak, B, pool, 'random', drop_last=True)
    tgt_strong = UAV(None, 'train', dcfg, mean_size=mean_size, records=[])       # label encoder for pseudo boxes
    from torchvision.transforms import ColorJitter
    jitter = ColorJitter(*args.jitter)
    iters_per_epoch = len(src_loader)
    total_iters = iters_per_epoch * args.epochs

    # ---- models
    model = build_model(model_cfg, mean_size)
    model.load_state_dict(src_ck['model_state'])
    model.to(dev)
    teacher, disc = None, None
    params = [{'params': model.parameters(), 'lr': lr}]
    if args.method == 'mt':
        teacher = copy.deepcopy(model).to(dev)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
    else:
        disc = DomainClassifier(cin=model.backbone.channels[model.first_level]).to(dev)
        params.append({'params': disc.parameters(), 'lr': lr * args.disc_lr_mult})
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    del src_ck

    start_epoch, global_iter, tau = 0, 0, None
    latest = os.path.join(args.out, 'checkpoints', 'checkpoint_latest.pth')
    if args.resume == 'auto' and os.path.exists(latest) or (args.resume not in (None, 'auto')):
        path = latest if args.resume == 'auto' else args.resume
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['student_state'])
        if teacher is not None:
            teacher.load_state_dict(ck['model_state'])
        if disc is not None:
            disc.load_state_dict(ck['disc_state'])
        try:
            optimizer.load_state_dict(ck['optimizer_state'])
            for g, base in zip(optimizer.param_groups, [lr, lr * args.disc_lr_mult]):
                g['lr'] = base   # batch may have been halved after an OOM: lr follows the current batch
        except Exception as e:
            logger.info('optimizer state not restored (%s)' % e)
        start_epoch, global_iter, tau = ck['epoch'], ck['global_iter'], ck.get('tau')
        try:
            random.setstate(ck['rng']['py']); np.random.set_state(ck['rng']['np']); torch.set_rng_state(ck['rng']['torch'])
        except Exception:
            pass
        logger.info('resumed from %s at epoch %d iter %d' % (path, start_epoch, global_iter))
        del ck

    # ---- MT: fix tau once from the source model (= initial teacher) on unlabeled target train images
    if args.method == 'mt' and tau is None:
        tau_file = os.path.join(args.out, 'tau.json')
        if args.tau is not None:
            tau = float(args.tau); tau_info = dict(tau=tau, rule='fixed')
        elif os.path.exists(tau_file):
            tau_info = json.load(open(tau_file)); tau = tau_info['tau']
        else:
            n = len(tgt_weak) if args.tau_frames is None else min(args.tau_frames, len(tgt_weak))
            dl = ThreadBatchLoader(tgt_weak, B, pool, 'seq', num_samples=n)
            sc = []
            t0 = time.time()
            with torch.no_grad():
                for inputs, calibs, coord_ranges, _, info in dl:
                    outs = teacher(inputs.to(dev), coord_ranges.to(dev), calibs.to(dev), K=50, mode='test')
                    dets = extract_dets_from_outputs(outs, K=50).cpu().numpy()
                    info = {k: v.numpy() for k, v in info.items()}
                    cal = [tgt_weak.get_calib(int(i)) for i in info['img_id']]
                    sc += [(-np.inf if b is None else b[3]) for b in decode_top1(dets, info, cal, mean_size)]
            sc = np.array(sc)
            tau = float(np.quantile(sc[np.isfinite(sc)], 1 - args.keep))
            tau_info = dict(tau=tau, rule='(1-keep) quantile of source-model top-1 score on target train', keep=args.keep,
                            n_frames=int(n), score_pcts={str(q): float(np.quantile(sc[np.isfinite(sc)], q)) for q in (0.05, 0.2, 0.5, 0.8, 0.95)},
                            sec=time.time() - t0)
            json.dump(tau_info, open(tau_file, 'w'), indent=1)
        logger.info('[MT] tau = %.4f  %s' % (tau, json.dumps(tau_info)))

    run_info = dict(args=vars(args), batch_size=B, lr=lr, iters_per_epoch=iters_per_epoch, total_iters=total_iters,
                    n_src=len(src_set), n_tgt=len(tgt_recs), tau=tau)
    logger.info(json.dumps(run_info, ensure_ascii=False))
    json.dump(run_info, open(os.path.join(args.out, 'run_info.json'), 'w'), ensure_ascii=False, indent=1)

    tgt_iter = [iter(tgt_loader)]

    def next_tgt():
        try:
            return next(tgt_iter[0])
        except StopIteration:
            tgt_iter[0] = iter(tgt_loader)
            return next(tgt_iter[0])

    def make_ckpt(epoch):
        st = dict(epoch=epoch, global_iter=global_iter, tau=tau, method=args.method,
                  student_state=model.state_dict(),
                  model_state=(teacher if teacher is not None else model).state_dict(),   # predict.py loads model_state
                  optimizer_state=optimizer.state_dict(),
                  rng=dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state()),
                  mean_size=mean_size, cfg=dcfg, model_cfg=model_cfg)
        if disc is not None:
            st['disc_state'] = disc.state_dict()
        return st

    stop = False
    t_last, n_skip = time.time(), 0
    acc = {}
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.bn != 'train':
            freeze_bn_stats(model)
        if disc is not None:
            disc.train()
        np.random.seed(args.seed * 1000 + epoch)
        logger.info('------ UDA[%s] EPOCH %03d / %d  lr %.3g ------' % (args.method, epoch + 1, args.epochs, lr))
        for bi, sb in enumerate(src_loader):
            p = min(1.0, (epoch + bi / float(iters_per_epoch)) / args.epochs)   # training progress (batch-size independent)
            optimizer.zero_grad(set_to_none=True)
            logs = {}
            # ---------------- source: GT detection loss (all task weights 1 = HTL weights at the end of source training)
            x, cal, cr, tg, _ = to_dev(sb, dev)
            if tg['mask_2d'].sum() > 0:
                ret, feat = forward_feat(model, x, cr, cal, tg)
                det, stat = GupnetLoss(epoch)(ret, tg)
                loss = det
                if disc is not None:
                    lam = args.grl_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
                    ds = disc(GradReverse.apply(feat, lam))
                    dl_s = F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
                    loss = loss + dl_s
                    logs['dom_src'] = dl_s.item(); logs['lam'] = lam
                    logs['dacc_src'] = (ds < 0).float().mean().item()
                if finite(loss):
                    loss.backward()
                    logs['src_det'] = det.item()
                    for k, v in stat.items():
                        logs['s_' + k.replace('_loss', '')] = float(v.detach())
                else:
                    n_skip += 1
                del ret, feat, loss
            # ---------------- target
            tb = next_tgt()
            if args.method == 'mt':
                xt, calt, crt, _, infot = tb
                with torch.no_grad():
                    outs = teacher(xt.to(dev), crt.to(dev), calt.to(dev), K=50, mode='test')
                    dets = extract_dets_from_outputs(outs, K=50).cpu().numpy()
                infon = {k: v.numpy() for k, v in infot.items()}
                ids = [int(i) for i in infon['img_id']]
                top1 = decode_top1(dets, infon, [tgt_weak.get_calib(i) for i in ids], mean_size)
                prs, kept_sc = [], []
                for j, (i, b) in enumerate(zip(ids, top1)):
                    if b is None or b[3] < tau:
                        continue          # frames without a confident pseudo box are not used (not trained as negatives)
                    r = dict(tgt_recs[i])
                    r['boxes'] = [dict(center=b[0], lwh=b[1], ypr=b[2])]
                    r['_pil'], r['_K'], r['_jitter'] = tensor_to_pil(xt[j]), tgt_weak._scaled_K(i), jitter
                    prs.append(r)
                    kept_sc.append(b[3])
                tgt_strong.records = prs  # pseudo-labelled records of this batch only, encoded by the dataset's own train path
                items = list(pool.map(tgt_strong.__getitem__, range(len(prs))))
                tgt_strong.records = []
                logs['kept'] = len(items)
                if items:
                    xs, cals, crs, tgs, _ = to_dev(default_collate(items), dev)
                    if tgs['mask_2d'].sum() > 0:
                        ret = model(xs, crs, cals, tgs)
                        tl, tstat = GupnetLoss(epoch)(ret, tgs)
                        w = args.tgt_weight * (float(np.mean(kept_sc)) if args.quality_weight else 1.0)
                        if finite(tl):
                            (w * tl).backward()
                            logs['tgt_det'] = tl.item()
                            logs['t_seg'] = float(tstat['seg_loss'].detach()); logs['t_depth'] = float(tstat['depth_loss'].detach())
                        else:
                            n_skip += 1
                        del ret, tl
            else:
                xt = tb[0].to(dev)
                ft = model.feat_up(model.backbone(xt)[model.first_level:])
                dt = disc(GradReverse.apply(ft, lam if 'lam' in logs else args.grl_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)))
                dl_t = F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt))
                if finite(dl_t):
                    dl_t.backward()
                    logs['dom_tgt'] = dl_t.item(); logs['dacc_tgt'] = (dt > 0).float().mean().item()
                del ft, dt, dl_t
            optimizer.step()
            if teacher is not None:
                ema_update(teacher, model, args.ema)
            global_iter += 1
            for k, v in logs.items():
                acc.setdefault(k, []).append(v)
            if global_iter % args.log_every == 0:
                now = time.time(); dt_ = now - t_last; t_last = now
                s = 'it %d/%d ep %d [%.2f it/s, peakmem %.2fGB, skipped %d] ' % (
                    global_iter, total_iters, epoch + 1, args.log_every / max(dt_, 1e-6),
                    torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0, n_skip)
                s += ' '.join('%s:%.4f' % (k, np.mean(v)) for k, v in sorted(acc.items()))
                logger.info(s)
                acc = {}
            if args.max_iters is not None and global_iter >= args.max_iters:
                stop = True
                break
        save_atomic(make_ckpt(epoch + 1), latest)
        logger.info('saved %s (epoch %d, iter %d)' % (latest, epoch + 1, global_iter))
        if (epoch + 1) % 10 == 0:
            st = make_ckpt(epoch + 1); st.pop('optimizer_state'); st.pop('rng')
            save_atomic(st, os.path.join(args.out, 'checkpoints', 'checkpoint_epoch_%d.pth' % (epoch + 1)))
        if stop:
            break
    if True:   # final model = last iterate (no model selection on any real data)
        st = make_ckpt(min(args.epochs, epoch + 1)); st.pop('optimizer_state'); st.pop('rng')
        save_atomic(st, os.path.join(args.out, 'checkpoints', 'checkpoint_final.pth'))
        logger.info('saved checkpoint_final (prediction model = %s)' % ('EMA teacher' if teacher is not None else 'model'))


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        print('[UDA] CUDA OOM: %s' % e, flush=True)
        sys.exit(3)
    except RuntimeError as e:
        if any(t in str(e).lower() for t in ('out of memory', 'alloc_failed')):
            print('[UDA] CUDA OOM (RuntimeError): %s' % e, flush=True)
            sys.exit(3)
        raise
