# -*- coding: utf-8 -*-
"""[UAV] GUPNet 预测（由 lib/helpers/tester_helper.py 改写）。
python predict.py --ckpt <ckpt.pth> --split real_test --out pred.pkl
输出：list，与 @@ROOT@@/data/cmpfull/<split>/labels.pkl 同序；每帧 dict(seq, frame, boxes9d (n,9)=center,lwh,euler_xyz, scores (n,))，
按分数降序最多 10 个（GUPNet 的 NMS 就是 heatmap 3x3 max-pool + top-K=50，原样保留）。
"""
import os, sys, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lib.datasets.uav import UAV, load_records
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections_uav

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--split', required=True, choices=['real_test', 'sim_val', 'real_train', 'sim_train', 'outdoor_test', 'outdoor_train', 'outdoor_val', 'sim_outdoor_train', 'sim_outdoor_val'])
ap.add_argument('--out', required=True)
ap.add_argument('--root', default='@@ROOT@@/data/cmpfull')
ap.add_argument('--thresh', type=float, default=0.0, help='官方 tester 用 0.2；默认 0 不过滤，交给评测按分数取')
ap.add_argument('--topk', type=int, default=10)
ap.add_argument('--n', type=int, default=None, help='只预测前 n 帧（冒烟测试）')
ap.add_argument('--batch-size', type=int, default=8)
ap.add_argument('--workers', type=int, default=2)
args = ap.parse_args()


def main():
    os.environ.setdefault('GUPNET_NO_PRETRAIN', '1')  # 推理时骨干权重来自 ckpt，不需要 ImageNet 权重
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.2')))
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    mean_size = np.asarray(ck['mean_size'], np.float32)
    dcfg = dict(ck['cfg'])
    recs = load_records([os.path.join(args.root, args.split, 'labels.pkl')])
    ds = UAV(None, 'test', dcfg, mean_size=mean_size, records=recs)
    n = len(ds) if args.n is None else min(args.n, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_model(ck.get('model_cfg', {'type': 'gupnet', 'backbone': 'dla34', 'neck': 'DLAUp'}), mean_size)
    model.load_state_dict(ck['model_state'])
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for inputs, calibs, coord_ranges, _, info in dl:
            outputs = model(inputs.to(dev), coord_ranges.to(dev), calibs.to(dev), K=50, mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=50).cpu().numpy()
            ids = info['img_id'].numpy()
            info = {k: v.numpy() for k, v in info.items()}
            cal = [ds.get_calib(int(i)) for i in ids]
            res = decode_detections_uav(dets, info, cal, mean_size, args.thresh)
            for i in ids:
                p = res[i]
                p = sorted(p, key=lambda r: -r[9])[:args.topk]
                b = np.array([r[:9] for r in p], np.float64).reshape(-1, 9)
                s = np.array([r[9] for r in p], np.float64).reshape(-1)
                out.append(dict(seq=recs[i]['seq'], frame=recs[i]['frame'], boxes9d=b, scores=s))
            print('%d/%d' % (len(out), n), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump(out, open(args.out, 'wb'))
    print('saved', args.out, len(out))


if __name__ == '__main__':
    main()
