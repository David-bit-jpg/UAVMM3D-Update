# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] SINGLE detector inference on a real target split, as the sim-only detector is normally evaluated without
known size: no ensemble, no flip TTA, no s_bar; decode = the target real config real_<combo>.yaml (source decoder stats) with
SIZE_SOURCE pred + SIZE2D_MODE width (depth from the predicted 2D width and the model's OWN predicted 3D size).
Per-frame post-processing = srcx/lib/det_infer_x.run (one box per frame, highest confidence, no threshold) -- same code, hooks not installed.
env SX_SRC / SX_TGT select the combo (srcx/lib/sx.py).
  python ourdet_infer.py --ckpt X.pth --splits train test --out recs.pkl [--pred-test preds_test.pkl]"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import det_infer_x as DI  # noqa: E402  (chdir tools, imports sx)
import sx  # noqa: E402
from easydict import EasyDict  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file, cfg_from_list  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.model import build_network  # noqa: E402
from uavdet3d.utils import common_utils  # noqa: E402


def build_single(ck, cfg_path, data_path=None):
    logger = common_utils.create_logger()
    cfg = EasyDict(); cfg_from_yaml_file(cfg_path, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
    if data_path:   # byte-identical cache copy (train split only; E: is saturated)
        cfg.DATA_CONFIG.DATA_PATH = data_path; cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'train'   # (network build only; run() sets the split)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=logger, training=False)
    m = build_network(model_cfg=cfg.MODEL, dataset=ds)
    a, b = m.load_params_from_file(filename=ck, to_cpu=False, logger=logger)
    assert a == b, ck
    return cfg, m.cuda().eval(), logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--pred-test', default=None)
    ap.add_argument('--data-path', default=None)
    a = ap.parse_args()
    cfg, model, logger = build_single(a.ckpt, sx.REAL_CFG, a.data_path)
    assert model.model_cfg.POST_PROCESSING.SIZE_SOURCE == 'pred'
    R = {}
    for sp in a.splits:
        R[sp] = DI.run(cfg, model, logger, sp)
        c = np.array([x['conf'] for x in R[sp]])
        print('%s: %d frames, conf median %.3f, no box %d' % (sp, len(c), np.median(c), sum(x['box'] is None for x in R[sp])), flush=True)
    pickle.dump(dict(ckpt=a.ckpt, cfg=sx.REAL_CFG, data_path=cfg.DATA_CONFIG.DATA_PATH, size_source='pred', flip=False, ensemble=False, recs=R), open(a.out, 'wb'))
    if a.pred_test:
        P = [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']]))
             for r in R['test'] if r['box'] is not None]
        pickle.dump(P, open(a.pred_test, 'wb')); print(len(P), 'test preds ->', a.pred_test)


if __name__ == '__main__':
    main()
