# -*- coding: utf-8 -*-
"""把几个 mmcache 格式的缓存（train split）合并成一个训练缓存，可对每个输入做可见度过滤和数量上限，
并从「原版」帧里切一小块出来当 test split（域内 sanity）。

    D:/Miniconda3/envs/city/python.exe tools/merge_mmcaches.py --out E:/data_collect/aug_paste_v1/mixed_cache \
        --input E:/data_collect/aug_paste_v1/cache:0:0:aug --input E:/mmcache/paste_src:5:2000:orig --test-from orig --test-n 100
--input 的格式：<缓存根目录>:<min_vis>:<最多帧数(0=不限)>:<标签>
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--input', action='append', required=True)
    ap.add_argument('--test-from', default='', help='从哪个标签的输入里切 test split')
    ap.add_argument('--test-n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    picks = []      # (root, idx_in_src, tag)
    src_meta = {}
    for spec in args.input:
        root, min_vis, cap, tag = spec.split(':')
        d = os.path.join(root, 'train')
        idx = pickle.load(open(os.path.join(d, 'index.pkl'), 'rb'))
        valid = np.asarray(idx['valid_idx'])
        vis = np.load(os.path.join(d, 'vis_score.npy')) if os.path.exists(os.path.join(d, 'vis_score.npy')) else np.full(len(idx['metas']), 100.0)
        keep = [int(i) for i in valid if vis[i] >= float(min_vis)]
        rng.shuffle(keep)
        if int(cap) > 0:
            keep = keep[:int(cap)]
        src_meta[root] = {'idx': idx, 'mm': {k: np.load(os.path.join(d, k + '.npy'), mmap_mode='r') for k in ('rgb', 'ir', 'depth', 'tag')},
                          'vis': vis}
        picks += [(root, i, tag) for i in keep]
        print('%-6s %s: 可用 %d（vis>=%s，上限 %s）' % (tag, root, len(keep), min_vis, cap))
    W, H = src_meta[picks[0][0]]['idx']['W'], src_meta[picks[0][0]]['idx']['H']
    for r in src_meta:
        assert (src_meta[r]['idx']['W'], src_meta[r]['idx']['H']) == (W, H), '分辨率不一致'
    # test split
    test = []
    if args.test_from:
        cand = [p for p in picks if p[2] == args.test_from]
        rng.shuffle(cand)
        test = cand[:args.test_n]
        picks = [p for p in picks if p not in set(test)]
    rng.shuffle(picks)
    for split, items in (('train', picks), ('test', test)):
        if not items:
            continue
        od = os.path.join(args.out, split)
        os.makedirs(od, exist_ok=True)
        N = len(items)
        mm = {'rgb': np.lib.format.open_memmap(os.path.join(od, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3)),
              'ir': np.lib.format.open_memmap(os.path.join(od, 'ir.npy'), 'w+', np.uint8, (N, H, W)),
              'depth': np.lib.format.open_memmap(os.path.join(od, 'depth.npy'), 'w+', np.uint16, (N, H, W)),
              'tag': np.lib.format.open_memmap(os.path.join(od, 'tag.npy'), 'w+', np.uint8, (N, H, W))}
        metas, vis = [], np.zeros(N, np.float32)
        t0 = time.time()
        for k, (root, i, tag) in enumerate(items):
            s = src_meta[root]
            for key in mm:
                mm[key][k] = s['mm'][key][i]
            m = dict(s['idx']['metas'][i])
            m['origin'] = tag
            m['origin_root'] = root
            metas.append(m)
            vis[k] = s['vis'][i]
            if (k + 1) % 500 == 0:
                print('  %s %d/%d  %.0fs' % (split, k + 1, N, time.time() - t0), flush=True)
        for v in mm.values():
            v.flush()
        base = src_meta[items[0][0]]['idx']
        pickle.dump({'W': W, 'H': H, 'classes': base['classes'], 'split': split, 'metas': metas, 'valid_idx': np.arange(N),
                     'merged_from': args.input}, open(os.path.join(od, 'index.pkl'), 'wb'))
        np.save(os.path.join(od, 'vis_score.npy'), vis)
        np.save(os.path.join(od, 'translated.npy'), np.ones(N, bool))
        import collections
        print('%s：%d 帧 %s' % (split, N, dict(collections.Counter(p[2] for p in items))))
    open(os.path.join(args.out, 'READY'), 'w').write('ok\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
