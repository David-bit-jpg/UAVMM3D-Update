# -*- coding: utf-8 -*-
"""[review1113 lidraw] smoke of the raw-point head (arm b): builds data + net exactly as train.py (workers 0), runs N steps with AMP like
train.py, prints loss terms (incl. pts), target coverage, Chamfer finiteness and it/s.  python smoke_lidpts.py [steps]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools'); sys.path.insert(0, HERE)
os.chdir('E:/Open3DUAVDet/tools')
import torch                                   # noqa: E402
import numpy as np                             # noqa: E402
import uavdet3d.model.detectors as dets        # noqa: E402
from center_det_lidpts import CenterDetLidPts  # noqa: E402
dets.__all__['CenterDetLidPts'] = CenterDetLidPts
from easydict import EasyDict                  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file # noqa: E402
from uavdet3d.datasets import build_dataloader # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils        # noqa: E402
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
cfg = EasyDict()
cfg_from_yaml_file(HERE + '/cfgs/sim_full_r34_kp_w12_zoom_lidpts.yaml', cfg)
print('MODEL.NAME', cfg.MODEL.NAME, '| LIDAR_AUX', dict(cfg.MODEL.LIDAR_AUX), '| DISTILL', dict(cfg.MODEL.DISTILL), '| MODALITIES', cfg.DATA_CONFIG.MODALITIES, flush=True)
common_utils.set_random_seed(0)
logger = common_utils.create_logger()
ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=logger, training=True, seed=0)
model = build_network(model_cfg=cfg.MODEL, dataset=ds).cuda().train()
opt = torch.optim.Adam(model.parameters(), 1e-4)
scaler = torch.cuda.amp.GradScaler()
it = iter(loader); t0 = None
for k in range(steps):
    if k == 2:
        torch.cuda.synchronize(); t0 = time.time()
    b = next(it)
    load_data_to_gpu(b)
    with torch.cuda.amp.autocast():
        ret = model(b)
    loss = ret['loss']
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    terms = {kk: round(float(v), 3) for kk, v in model.dense_head_2d.loss_terms.items()}
    assert np.isfinite(terms['pts']) and np.isfinite(float(loss)), terms
    if k < 3 or k % 5 == 0 or k == steps - 1:
        T = model.pts_targets(b)
        n_ok = sum(t is not None for t in T)
        sz = [len(t) for t in T if t is not None]
        print(k, 'loss %.3f' % float(loss), terms, '| crops with a LiDAR point target %d/%d, points/crop median %s, max %s | grad pts_head.mlp %.2e' % (
            n_ok, len(T), int(np.median(sz)) if sz else 0, max(sz) if sz else 0,
            float(model.pts_head['mlp'][2].weight.grad.abs().mean())), flush=True)
torch.cuda.synchronize()
print('it/s (workers 0, batch 8, AMP, steps 2..%d): %.2f | peak GPU mem %.2f GB' % (steps - 1, (steps - 2) / (time.time() - t0), torch.cuda.max_memory_allocated() / 2**30), flush=True)
print('SMOKE OK', flush=True)
