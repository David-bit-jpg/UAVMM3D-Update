# -*- coding: utf-8 -*-
"""[sup1] AdaBN baseline for the clean MAV6D chain's sim-only detector ensemble (inference only, no gradient step).
  1. the 3 sim-only detectors behind srcx/in2mav/preds/srconly_* (sx.S['det'] = release_20260920/ramp_s{0,1,2}.pth), built exactly as
     srcx/lib/det_infer_x.build (target config srcx/real_in2mav.yaml, SIZE_SOURCE pred, SIZE2D_MODE width);
  2. each member's BatchNorm statistics re-estimated on UNLABELED MAV6D train images with the existing detector AdaBN routine
     tools/eval_camnorm.adapt_bn (reset running stats, momentum None = cumulative average, only BN layers in train mode, shuffled with
     generator seed 0, no augmentation, torch.no_grad -> weights untouched). Default: all 12,866 train frames (--interval 1, every batch);
  3. ensemble hook + horizontal-flip hook installed AFTER the adaptation, s_bar self-calibrated on unlabeled val (every 2nd frame,
     conf >= 0.7 median) by this adapted ensemble, then val + test decoded with that constant size = det_infer_x.main unchanged.
Output: same records format as det_infer_x (dict(size, ckpt, recs={split: [...]}), + adabn info). Adapted BN states are cached in
--bn-dir (bn_s<k>.pth) so a restart skips the adaptation. Batch = $DET_BS (default 8; halved by the wrapper on OOM).
Memory: torch.cuda.set_per_process_memory_fraction($SX_INFER_MEMFRAC, default 0.2) before anything touches CUDA.
  python det_infer_adabn.py --out recs/det0_adabn.pkl --bn-dir bn [--splits val test] [--interval 1] [--smoke N]"""
import argparse, os, pickle, sys, time
import numpy as np
import torch
FR = float(os.environ.get('SX_INFER_MEMFRAC', '0.2')); assert 0 < FR <= 0.2, FR
BS = int(os.environ.get('DET_BS', '8'))
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import det_infer_x as DI  # noqa  (chdir to tools; sx with SX_SRC=in SX_TGT=mav -> reads only)
import eval_camnorm as EC  # noqa
from easydict import EasyDict  # noqa
from uavdet3d.config import cfg_from_yaml_file, cfg_from_list  # noqa
from uavdet3d.model import build_network  # noqa
from uavdet3d.utils import common_utils  # noqa
sx = DI.sx

_orig_bd = DI.build_dataloader


def _bd(**kw):   # det_infer_x.run / build call build_dataloader(... batch_size=8 ...) with keywords only -> batch = $DET_BS
    kw['batch_size'] = BS
    return _orig_bd(**kw)


DI.build_dataloader = _bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--bn-dir', required=True)
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--adabn-split', default='train')
    ap.add_argument('--interval', type=int, default=1, help='1 = all 12,866 unlabeled train frames')
    ap.add_argument('--smoke', type=int, default=0, help='>0: adapt on that many batches only and write nothing but the BN cache to --bn-dir')
    a = ap.parse_args()
    assert sx.SRC == 'in' and sx.TGT == 'mav' and sx.TEACH == 'full', (sx.SRC, sx.TGT, sx.TEACH)
    torch.cuda.set_per_process_memory_fraction(FR)
    os.makedirs(a.bn_dir, exist_ok=True)
    ckpts = list(sx.S['det'])
    print('[sup1] sim-only detectors (sx.S[det]): %s | memory fraction %.2f | batch %d' % (ckpts, FR, BS), flush=True)
    logger = common_utils.create_logger()
    cfg = EasyDict(); cfg_from_yaml_file(DI.CFG, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)   # = det_infer_x.build
    ds, _, _ = _bd(dataset_cfg=cfg.DATA_CONFIG, batch_size=BS, dist=False, workers=0, logger=logger, training=False)
    ms, info = [], []
    for k, ck in enumerate(ckpts):
        m = build_network(model_cfg=cfg.MODEL, dataset=ds)
        n1, n2 = m.load_params_from_file(filename=ck, to_cpu=False, logger=logger); assert n1 == n2, ck
        m.cuda().eval()
        bn = '%s/bn_s%d.pth' % (a.bn_dir, k)
        if os.path.exists(bn) and not a.smoke:
            z = torch.load(bn, map_location='cuda', weights_only=False); m.load_state_dict(z['state']); info.append(z['info'])
            print('member %d: adapted BN state loaded from %s (%s)' % (k, bn, z['info']), flush=True)
        else:
            t0 = time.time()
            nb, nbatch = EC.adapt_bn(cfg, m, a.adabn_split, a.interval, BS, a.smoke or 10 ** 9, logger)
            m.eval()
            inf = dict(ckpt=ck, split=a.adabn_split, interval=a.interval, bn_layers=nb, batches=nbatch, batch_size=BS, seconds=round(time.time() - t0))
            print('member %d AdaBN: %d BN layers re-estimated on %d batches x %d unlabeled %s frames (interval %d), %.0f s' % (
                k, nb, nbatch, BS, a.adabn_split, a.interval, time.time() - t0), flush=True)
            torch.save(dict(state=m.state_dict(), info=inf), bn + ('.smoke' if a.smoke else '')); info.append(inf)
        ms.append(m)
    if a.smoke:
        print('smoke done (nothing else written)'); return
    model = ms[0]
    EC.install_ensemble_hook(model, ms[1:])   # = det_infer_x.build: ensemble first, then flip TTA
    EC.install_flip_hook(model)
    # s_bar self-calibration and decoding = det_infer_x.main (unlabeled val every 2nd frame, conf >= 0.7 median)
    model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'pred'
    r = DI.run(cfg, model, logger, 'val', 2)
    got = [np.abs(x['box'][3:6]) for x in r if x['box'] is not None and x['conf'] >= 0.7 and x['box'][2] > 0]
    if got:
        size = np.median(np.array(got), 0); print('s_bar (val unlabeled, %d boxes conf>=0.7 of %d) = %s' % (len(got), len(r), np.round(size, 4)), flush=True)
    else:   # same fallback as det_infer_x (top-10% by confidence, >= 20)
        ok = sorted([x for x in r if x['box'] is not None and x['box'][2] > 0], key=lambda x: -x['conf'])[:max(20, len(r) // 10)]
        size = np.median(np.array([np.abs(x['box'][3:6]) for x in ok]), 0); print('s_bar: no conf>=0.7 val box; top-%d by conf = %s' % (len(ok), np.round(size, 4)), flush=True)
    model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
    model.model_cfg.POST_PROCESSING.KNOWN_SIZE = [float(v) for v in size]
    R = {}
    for sp in a.splits:
        R[sp] = DI.run(cfg, model, logger, sp)
        c = np.array([x['conf'] for x in R[sp]])
        print('%s: %d frames, conf median %.3f, >=0.5 %.1f%%, >=0.3 %.1f%%' % (sp, len(c), np.median(c), 100 * (c >= .5).mean(), 100 * (c >= .3).mean()), flush=True)
    pickle.dump(dict(size=size, ckpt=ckpts, recs=R, adabn=info), open(a.out, 'wb'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
