# -*- coding: utf-8 -*-
"""PREREG_dcalign.md 第 2 步：第二段（dc/{DC0,DCS}_s{k mod 3}.pt，不重训）在真实验证集 P0 检测上推理 → 对齐量 c_val；
测试集的第二段原样结果直接用 dc/eval/{DC0,DCS}_real_s{k}.pkl（与 dc_eval.py 同一流程）→ c_test。
c = 被改动检测（|δ̂| ≤ ln 1.4）上 δ̂ 的中位数，δ̂ 从「第二段原样输出 / P0」的深度比读出（两者检测一一对应）。
输出（dcalign/）：DCA0 / DCAS 的 test_s{k}（用 c_val）与 val_s{k}（用 c_test，反向交叉复核）；selfcheck_test_s{k}（用 c_test，须复现 white/fusion_test.json）。"""
import json, os, pickle, sys, time
import numpy as np
import torch
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
import dc_eval as de  # noqa: E402  （会切工作目录，下面全用绝对路径）
import dc_train as dt  # noqa: E402
OUT = G + '/dcalign'
ARMS = {'DCA0': 'DC0', 'DCAS': 'DCS'}

class ImagesVal:
    def __init__(self):
        b = '@@MMCACHE@@/mav6d_cn_obj/val'
        self.arr = np.load(b + '/rgb.npy', mmap_mode='r')
        d = pickle.load(open(b + '/index.pkl', 'rb'))
        self.metas = d['metas']
        self.key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
    def get(self, seq, frame):
        i = self.key[(seq, frame)]
        m = self.metas[i]
        return np.ascontiguousarray(self.arr[i]), np.asarray(m['K_in'], np.float64), m

def ratios(base, h2):
    """每个检测的 log(z2/z1)；未改动的为 nan"""
    out = []
    for a, b in zip(base, h2):
        assert a['seq_id'] == b['seq_id'] and a['frame_id'] == b['frame_id'] and len(a['pred']) == len(b['pred'])
        if len(a['pred']) == 0:
            out.append(np.zeros(0)); continue
        r = np.log(np.maximum(b['pred'][:, 2], 1e-9) / np.maximum(a['pred'][:, 2], 1e-9))
        r[np.abs(r) <= 1e-7] = np.nan
        out.append(r)
    return out

def apply(base, rs, c):
    new = []
    for a, r in zip(base, rs):
        p = a['pred'].copy()
        if len(r):
            f = np.where(np.isfinite(r), np.exp(np.nan_to_num(r) - c), 1.0)
            p[:, :3] *= f[:, None]
        new.append(dict(a, pred=p))
    return new

def main():
    os.makedirs(OUT + '/raw', exist_ok=True)
    imgs = ImagesVal()
    info = {}
    for k in range(5):
        t0 = time.time()
        zv = pickle.load(open(G + '/irloc_val_recs/P0_s%d.pkl' % k, 'rb'))
        zt = pickle.load(open(G + '/irloc_recs/P0_s%d.pkl' % k, 'rb'))
        raw_val = {}
        need = [a for a in ARMS.values() if not os.path.exists('%s/raw/%s_val_s%d.pkl' % (OUT, a, k))]
        if need:
            nets = {}
            for a in ARMS.values():
                n = dt.DCNet().cuda()
                n.load_state_dict(torch.load('%s/dc/%s_s%d.pt' % (G, a, k % 3), map_location='cuda'))
                nets[a] = n.eval()
            outs, st = de.process(zv['records'], imgs, nets)
            for a in ARMS.values():
                pickle.dump(dict(zv, records=outs[a]), open('%s/raw/%s_val_s%d.pkl' % (OUT, a, k), 'wb'))
            del nets; torch.cuda.empty_cache()
        for arm, a in ARMS.items():
            rv = ratios(zv['records'], pickle.load(open('%s/raw/%s_val_s%d.pkl' % (OUT, a, k), 'rb'))['records'])
            rt = ratios(zt['records'], pickle.load(open('%s/dc/eval/%s_real_s%d.pkl' % (G, a, k), 'rb'))['records'])
            cv = float(np.nanmedian(np.concatenate(rv))); ct = float(np.nanmedian(np.concatenate(rt)))
            pickle.dump(dict(zt, records=apply(zt['records'], rt, cv)), open('%s/%s_test_s%d.pkl' % (OUT, arm, k), 'wb'))
            pickle.dump(dict(zv, records=apply(zv['records'], rv, ct)), open('%s/%s_val_s%d.pkl' % (OUT, arm, k), 'wb'))
            pickle.dump(dict(zt, records=apply(zt['records'], rt, ct)), open('%s/selfcheck_%s_test_s%d.pkl' % (OUT, arm, k), 'wb'))
            info['%s_s%d' % (arm, k)] = dict(c_val=cv, c_test=ct, n_changed_val=int(np.isfinite(np.concatenate(rv)).sum()),
                                             n_changed_test=int(np.isfinite(np.concatenate(rt)).sum()))
            print(arm, k, json.dumps(info['%s_s%d' % (arm, k)]), '%.0fs' % (time.time() - t0), flush=True)
    json.dump(info, open(OUT + '/c_values.json', 'w'), indent=1)
    print('全部完成', flush=True)

if __name__ == '__main__':
    main()
