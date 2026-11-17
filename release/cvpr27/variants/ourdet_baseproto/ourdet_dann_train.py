# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] "CenterNet + DANN" row = the protocol of the GUPNet + DANN baseline
(cmpfull/gupnet/code/tools/uda_train.py --method dann; run record D:/cmpfull_runs/gupnet_uda_dann_s0/run_info.json), on OUR detector:
  * start from the sim-only single detector, seed 0 (same checkpoint as row 1);
  * domain classifier = the same DomainClassifier as uda_train.py (3 x conv3x3/s2 + ReLU, GAP, MLP 64, dropout 0.5, 1 logit; no BN) on the
    neck feature map (TvResNet FPN output features_2d, stride 8) through a gradient reversal layer,
    lambda_p = grl_max * (2 / (1 + exp(-10 p)) - 1), grl_max 0.1, p = training progress;
  * each step: source batch (sim train, the recipe's own augmentation) = GT detection loss + domain BCE (label 0); target batch (unlabeled real
    train images of the target cache, no augmentation, every label ignored -- only 'image' is read) = domain BCE only (label 1);
  * detection loss weights = those at the END of the sim-only training (rot ramp 2->16 ended at 16; GUPNet: HTL weights at the end);
  * constant LR = 0.1 x the sim-only peak LR (OneCycle peak 1e-3 -> 1e-4), AdamW (the recipe's adam_onecycle optimizer family: betas
    (0.9, 0.99), weight decay 0.01), domain classifier LR x10 (disc_lr_mult 10), grad-norm clip 35 (recipe); BN statistics frozen (bn=frozen);
  * budget: as many iterations as our 5-epoch sim recipe (5 x len(sim train loader) source batches of 8; GUPNet ran 40 x 3712 source samples
    = 18,560 its); last checkpoint, no model selection.
usage (cwd tools):  python ourdet_dann_train.py --combo in2mav|out2antifs --out DIR [--batch 8] [--epochs 5]"""
import argparse, copy, json, math, os, sys, time
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
os.chdir('E:/Open3DUAVDet/tools')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler
from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset
from uavdet3d.model import build_network, load_data_to_gpu
from uavdet3d.utils import common_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import thread_loader as TL  # 11-16 thread-pool batches (no spawn workers)

CF = 'cfgs/models/uavdet_3d/camnorm/'
COMBOS = {
    'in2mav': dict(sim_cfg=CF + 'sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml', sim_path='F:/mmcache/full_v2',
                   real_cfg=CF + 'srcx/real_in2mav.yaml', real_path='D:/S2MUAV3D/ourdet_baseproto_cache/mav6d_cn_obj',
                   init='E:/Open3DUAVDet/output/release_20260920/ramp_s0.pth'),
    'out2antifs': dict(sim_cfg=CF + 'sim_out_r34_geo_ramp_out.yaml', sim_path=None,
                       real_cfg=CF + 'srcx/real_out2antifs.yaml', real_path='D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs',
                       init='E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out/DET_s0/ckpt/checkpoint_epoch_5.pth')}


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainClassifier(nn.Module):   # = uda_train.py
    def __init__(self, cin=64, c=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, c, 3, 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(nn.Linear(c, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, f):
        return self.mlp(self.conv(f).mean((2, 3))).squeeze(1)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combo', required=True, choices=list(COMBOS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grl-max', type=float, default=0.1)
    ap.add_argument('--disc-lr-mult', type=float, default=10.0)
    ap.add_argument('--lr-mult', type=float, default=0.1)
    ap.add_argument('--log-every', type=int, default=100)
    a = ap.parse_args()
    C = COMBOS[a.combo]
    os.makedirs(a.out + '/ckpt', exist_ok=True)
    logger = common_utils.create_logger(a.out + '/train_dann.log')
    common_utils.set_random_seed(a.seed)
    cfg = EasyDict(); cfg_from_yaml_file(C['sim_cfg'], cfg)
    if C['sim_path']:
        cfg.DATA_CONFIG.DATA_PATH = C['sim_path']
    src_set, src_loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=a.batch, dist=False, workers=0, logger=logger,
                                              training=True, seed=a.seed)
    rc = EasyDict(); cfg_from_yaml_file(C['real_cfg'], rc)
    tc = copy.deepcopy(rc.DATA_CONFIG); tc.DATA_PATH = C['real_path']; tc.DATA_SPLIT['test'] = 'train'; tc.SAMPLED_INTERVAL['test'] = 1
    tgt_set = MMCache_Det_Dataset(dataset_cfg=tc, training=False, logger=logger)   # no augmentation; only 'image' is used below
    src_loader = TL.ThreadLoader(src_set, a.batch, shuffle=True, drop_last=False, threads=6, seed=a.seed)
    tgt_loader = TL.ThreadLoader(tgt_set, a.batch, shuffle=True, drop_last=True, threads=4, seed=a.seed + 1)
    model = build_network(model_cfg=cfg.MODEL, dataset=src_set)
    model.load_params_from_file(filename=C['init'], to_cpu=False, logger=logger)
    model.cuda()
    head = model.dense_head_2d
    ramp = dict((cfg.MODEL.get('DENSE_HEAD_2D', {}) or {}).get('LOSS_WEIGHT_RAMP', {}) or {})
    for k, (lo, hi) in ramp.items():
        head.loss_weights[k] = float(hi)
    logger.info('loss weights (end of sim-only training): %s' % dict(head.loss_weights))
    with torch.no_grad():
        model.eval(); b0 = next(iter(tgt_loader)); load_data_to_gpu(b0)
        cin = model.backbone_2d({'image': b0['image']})['features_2d'].shape[-3]
    disc = DomainClassifier(cin=cin).cuda()
    lr = a.lr_mult * float(cfg.OPTIMIZATION.LR)
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': lr}, {'params': disc.parameters(), 'lr': lr * a.disc_lr_mult}],
                            lr=lr, betas=(0.9, 0.99), weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY))
    ipe = len(src_loader); total = ipe * a.epochs
    start_ep, it = 0, 0
    lat = a.out + '/ckpt/latest.pth'
    if os.path.exists(lat):
        ck = torch.load(lat, map_location='cuda')
        model.load_state_dict(ck['model_state']); disc.load_state_dict(ck['disc_state']); opt.load_state_dict(ck['opt_state'])
        start_ep, it = ck['epoch'], ck['it']; logger.info('resumed epoch %d it %d' % (start_ep, it))
    info = dict(vars(a), init=C['init'], sim_cfg=C['sim_cfg'], sim_path=cfg.DATA_CONFIG.DATA_PATH, real_path=C['real_path'], lr=lr,
                disc_lr=lr * a.disc_lr_mult, feat_channels=cin, iters_per_epoch=ipe, total_iters=total, n_src=len(src_set), n_tgt=len(tgt_set),
                loss_weights=dict(head.loss_weights), bn='frozen', grad_clip=float(cfg.OPTIMIZATION.GRAD_NORM_CLIP))
    json.dump(info, open(a.out + '/run_info.json', 'w'), indent=1, default=str); logger.info(info)
    tit = iter(tgt_loader); acc = {}; t0 = time.time()
    for ep in range(start_ep, a.epochs):
        model.train(); freeze_bn(model); disc.train()
        skip = it - ep * ipe                         # mid-epoch resume: run only the remaining batches of this epoch
        for bi, sb in enumerate(src_loader):
            if bi + skip >= ipe:
                break
            bi = bi + skip
            p = min(1.0, (ep + bi / float(ipe)) / a.epochs)
            lam = a.grl_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            opt.zero_grad(set_to_none=True)
            load_data_to_gpu(sb)
            det = model(sb)['loss']
            fs = sb['features_2d']; fs = fs.reshape(-1, *fs.shape[-3:])
            ds = disc(GradReverse.apply(fs, lam))
            dl_s = F.binary_cross_entropy_with_logits(ds, torch.zeros_like(ds))
            (det + dl_s).backward()
            try:
                tb = next(tit)
            except StopIteration:
                tit = iter(tgt_loader); tb = next(tit)
            xt = torch.from_numpy(tb['image']).float().cuda() if isinstance(tb['image'], np.ndarray) else tb['image'].cuda()
            ft = model.backbone_2d({'image': xt})['features_2d']; ft = ft.reshape(-1, *ft.shape[-3:])
            dt = disc(GradReverse.apply(ft, lam))
            dl_t = F.binary_cross_entropy_with_logits(dt, torch.ones_like(dt))
            dl_t.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.OPTIMIZATION.GRAD_NORM_CLIP))
            opt.step()
            it += 1
            for k, v in (('det', det.item()), ('dom_src', dl_s.item()), ('dom_tgt', dl_t.item()),
                         ('dacc', 0.5 * ((ds < 0).float().mean().item() + (dt > 0).float().mean().item())), ('lam', lam)):
                acc.setdefault(k, []).append(v)
            if it % a.log_every == 0:
                logger.info('it %d/%d ep %d %.2f it/s peak %.1fGB ' % (it, total, ep + 1, a.log_every / (time.time() - t0),
                            torch.cuda.max_memory_allocated() / 2 ** 30) + ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in acc.items()))
                acc = {}; t0 = time.time()
            if it % 1000 == 0:
                torch.save(dict(model_state=model.state_dict(), disc_state=disc.state_dict(), opt_state=opt.state_dict(), epoch=ep, it=it), lat)
        torch.save(dict(model_state=model.state_dict(), disc_state=disc.state_dict(), opt_state=opt.state_dict(), epoch=ep + 1, it=it), lat)
        logger.info('epoch %d done (it %d)' % (ep + 1, it))
    torch.save(dict(model_state=model.state_dict()), a.out + '/ckpt/checkpoint_final.pth')
    logger.info('DONE')


if __name__ == '__main__':
    main()
