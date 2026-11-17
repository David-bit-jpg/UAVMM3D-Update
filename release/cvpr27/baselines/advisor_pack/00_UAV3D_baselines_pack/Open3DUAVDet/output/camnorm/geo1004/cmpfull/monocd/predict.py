# -*- coding: utf-8 -*-
"""[UAV port] MonoCD inference on a cmpfull split (official post-processing: heatmap max-pool NMS, top-50,
score threshold TEST.DETECTIONS_THRESHOLD = 0.2, score = heatmap * (1 - clamp(depth uncertainty, 0.01, 1))).

  python predict.py --ckpt OUT/model_final.pth --split real_test --out pred.pkl [--limit 50]

Output: list in the same order as @@ROOT@@/data/cmpfull/<split>/labels.pkl; per frame dict(seq, frame, boxes9d (n x 9:
center, lwh, euler_xyz in the cmpfull definition), scores (n,)), at most 10 boxes (highest scores).
"""
import os, sys, argparse, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # cmpfull/angles.py
import numpy as np
import torch
from angles import euler_xyz_from_ypr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', required=True, help='real_test | sim_val | real_train (or a path to a labels.pkl)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=None, help='default: config.yaml next to the checkpoint, else runs/monocd_uav.yaml')
    ap.add_argument('--data-root', default='@@ROOT@@/data/cmpfull')
    ap.add_argument('--limit', type=int, default=None, help='only the first N frames (smoke test)')
    ap.add_argument('--max-det', type=int, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--thresh', type=float, default=None, help='override TEST.DETECTIONS_THRESHOLD (official 0.2); smoke only')
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.2')))

    from config import cfg
    cfgf = a.config or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), 'config.yaml')
    if not os.path.exists(cfgf):
        cfgf = os.path.join(HERE, 'runs', 'monocd_uav.yaml')
    cfg.merge_from_file(cfgf)
    cfg.DATASETS.TEST_SPLIT = 'test'      # official test mode: no targets
    cfg.MODEL.PRETRAIN = False            # weights come from the checkpoint
    cfg.MODEL.USE_SYNC_BN = False
    if a.thresh is not None:
        cfg.TEST.DETECTIONS_THRESHOLD = a.thresh
    cfg.freeze()
    from data.datasets.uav import UAVDataset
    from data.transforms import build_transforms
    from data.collate_batch import BatchCollator
    from model.detector import KeypointDetector

    lab = a.split if a.split.endswith('.pkl') else os.path.join(a.data_root, a.split, 'labels.pkl')
    recs = pickle.load(open(lab, 'rb'))
    ds = UAVDataset(cfg, lab, is_train=False, transforms=build_transforms(cfg, False), augment=False, with_targets=False)
    n = len(ds) if a.limit is None else min(a.limit, len(ds))
    sub = torch.utils.data.Subset(ds, list(range(n)))
    loader = torch.utils.data.DataLoader(sub, batch_size=1, shuffle=False, num_workers=a.workers,
                                         collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY))
    model = KeypointDetector(cfg).cuda()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    # [UAV] 训练时 DLA34 按 ImageNet 权重构建，带分类头 fc（检测不用）；预测时没建这层 → 只丢掉这两项，其余仍严格加载
    sd = {k: v for k, v in sd.items() if not k.startswith('backbone.base.fc.')}
    model.load_state_dict(sd)
    model.eval()

    out = []
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            idx = int(data['img_ids'][0])
            images = data['images'].to('cuda')
            targets = [t.to('cuda') for t in data['targets']]
            result, _, _ = model(images, targets)
            res = result.detach().cpu().numpy().astype(np.float64)   # [cls, score, center(3), lwh(3), ypr(3), box2d(4)]
            res = res[np.argsort(-res[:, 1])][:a.max_det] if len(res) else res.reshape(0, 15)
            b9 = np.zeros((len(res), 9))
            for j, r in enumerate(res):
                b9[j, :6] = r[2:8]
                b9[j, 6:] = euler_xyz_from_ypr(r[8:11])
            out.append(dict(seq=recs[idx]['seq'], frame=recs[idx]['frame'], boxes9d=b9, scores=res[:, 1].copy()))
    pickle.dump(out, open(a.out, 'wb'))
    print('%d frames -> %s (%.1f ms/frame), %d boxes' % (len(out), a.out, 1000 * (time.time() - t0) / max(len(out), 1),
                                                         sum(len(o['scores']) for o in out)))


if __name__ == '__main__':
    main()
