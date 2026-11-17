# [review1113 copy of srcx/lib/det_infer_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s20/lib] batch = $DET_BS (default 8 = deployed)
# -*- coding: utf-8 -*-
"""[outdoor->srcx] detector inference on the real outdoor cache (no labels anywhere: the cache only holds placeholder boxes).
Same post-processing as tools/eval_camnorm.py: seed ensemble hook + horizontal-flip TTA hook, one box per frame (MAX_OBJ 1, no threshold),
width-only geometric depth with the detector's constant size s_bar.
s_bar (self-calibration, MODULES §0.1): per-component median of predicted lwh of top boxes with conf >= 0.7 on UNLABELED val images
(every 2nd frame), computed by this same ensemble+flip; or passed with --size.

  python det_infer_out.py --ckpt A [B ...] --splits train val test --out recs_<tag>.pkl [--size l w h]
Output: dict(size=s_bar, recs={split: [dict(seq, frame, box(9), conf, K_in(3x3 at 512x288), w2d(px at 512 input))]})"""
import argparse, os, pickle, sys
import numpy as np
import torch
from easydict import EasyDict
os.environ.setdefault('UAV_PNP_EPNP_FALLBACK', '1')
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
os.chdir('E:/Open3DUAVDet/tools')
from uavdet3d.config import cfg_from_yaml_file, cfg_from_list  # noqa
from uavdet3d.datasets import build_dataloader  # noqa
from uavdet3d.model import build_network, load_data_to_gpu  # noqa
from uavdet3d.utils import common_utils  # noqa
import eval_camnorm as EC  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa

sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s20/lib'); import sx  # noqa
CFG = sx.REAL_CFG   # [srcx] target data + source decoder stats
P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])


def w2d_of(box, K):
    C = (Rot.from_euler('xyz', box[6:9]).as_matrix() @ (P8 * box[3:6]).T).T + box[:3]
    uv = (K @ C.T).T
    return float(np.ptp(uv[:, 0] / uv[:, 2]))


def build(ckpts, cfg_path=CFG, sets=None):
    logger = common_utils.create_logger()
    cfg = EasyDict(); cfg_from_yaml_file(cfg_path, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'] + (sets or []), cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=int(os.environ.get('DET_BS', '8')), dist=False, workers=0, logger=logger, training=False)
    ms = []
    for ck in ckpts:
        m = build_network(model_cfg=cfg.MODEL, dataset=ds)
        a, b = m.load_params_from_file(filename=ck, to_cpu=False, logger=logger)
        assert a == b, ck
        ms.append(m.cuda().eval())
    model = ms[0]
    if len(ms) > 1:
        EC.install_ensemble_hook(model, ms[1:])
    EC.install_flip_hook(model)
    return cfg, model, logger


def run(cfg, model, logger, split, interval=1):
    import copy
    dc = copy.deepcopy(cfg.DATA_CONFIG); dc.DATA_SPLIT['test'] = split; dc.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(dataset_cfg=dc, batch_size=int(os.environ.get('DET_BS', '8')), dist=False, workers=0, logger=logger, training=False)
    out, k = [], 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            res = model(batch)
            for b in range(res['batch_size']):
                meta = ds.metas[int(ds.valid_idx[k])]; k += 1
                p = np.asarray(res['pred_boxes9d'][b], np.float64).reshape(-1, 10)
                c = np.asarray(res['confidence'][b], np.float64).reshape(-1)
                K = np.asarray(meta['K_in'], np.float64)
                if not len(c):
                    out.append(dict(seq=meta['seq'], frame=meta['frame'], box=None, conf=0.0, K_in=K, w2d=np.nan)); continue
                j = int(np.argmax(c))
                out.append(dict(seq=meta['seq'], frame=meta['frame'], box=p[j, :9], conf=float(c[j]), K_in=K, w2d=w2d_of(p[j, :9], K)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', nargs=3, type=float, default=None)
    ap.add_argument('--cfg', default=CFG)
    ap.add_argument('--size-fallback', default=None, help='[srcx] recs.pkl whose s_bar is used if no conf>=0.7 val box exists')
    a = ap.parse_args()
    cfg, model, logger = build(a.ckpt, a.cfg)
    if a.size is None:
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'pred'
        r = run(cfg, model, logger, 'val', 2)
        got = [np.abs(x['box'][3:6]) for x in r if x['box'] is not None and x['conf'] >= 0.7 and x['box'][2] > 0]
        if not got and a.size_fallback:
            size = np.asarray(pickle.load(open(a.size_fallback, 'rb'))['size'], np.float64); print('s_bar: no conf>=0.7 val box; fallback to %s s_bar = %s' % (a.size_fallback, np.round(size, 4)))
        elif not got:
            # [srcx 11-12] weak source (indoor sim -> AntiUAV3D v2 val has no conf>=0.7 box): median over the top-10% most confident
            # val boxes (>= 20), still unlabeled-image-only. Logged so the table can footnote it.
            ok = sorted([x for x in r if x['box'] is not None and x['box'][2] > 0], key=lambda x: -x['conf'])
            ok = ok[:max(20, len(ok) // 10)]; assert ok, 'self-calibration: no detections on val'
            size = np.median(np.array([np.abs(x['box'][3:6]) for x in ok]), 0)
            print('s_bar: no conf>=0.7 val box; top-%d by conf (min conf %.3f) = %s' % (len(ok), ok[-1]['conf'], np.round(size, 4)))
        else:
            size = np.median(np.array(got), 0); got and print('s_bar (val unlabeled, %d boxes conf>=0.7 of %d) = %s' % (len(got), len(r), np.round(size, 4)))
    else:
        size = np.array(a.size)
    model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
    model.model_cfg.POST_PROCESSING.KNOWN_SIZE = [float(v) for v in size]
    R = {}
    for sp in a.splits:
        R[sp] = run(cfg, model, logger, sp)
        c = np.array([x['conf'] for x in R[sp]])
        print('%s: %d frames, conf median %.3f, >=0.5 %.1f%%, >=0.3 %.1f%%' % (sp, len(c), np.median(c), 100 * (c >= .5).mean(), 100 * (c >= .3).mean()), flush=True)
    pickle.dump(dict(size=size, ckpt=a.ckpt, recs=R), open(a.out, 'wb'))


if __name__ == '__main__':
    main()
