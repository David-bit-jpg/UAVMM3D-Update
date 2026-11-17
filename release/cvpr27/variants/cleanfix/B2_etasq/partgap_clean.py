# -*- coding: utf-8 -*-
"""[cleanfix B2] Copy of geo1004/partgap/partgap.py (10-29, PREREG_partgap.md) with ONLY the real-crop source changed to the clean chain.
Replicated unchanged (partgap.py line refs):
  networks  KZ-KPS simulation weights KPSs0..2 (KE.E.CK, no AdaBN)                         partgap.py:144-147, kps_eval.py:17
  sim side  the SAME 885 simulated samples (cached partgap/sim_samples.pkl, read only;     partgap.py:27-50
            sim test zoom windows, part points = LiDAR template x GT pose; simulation only)
  layers    layer2/3/4 + head;  pools win/ctr/motor/nose/tail/top/bottom/body8             partgap.py:61-62, 71-104
  metrics   domain_gap.metrics (eta2 / PCA64 logistic AUC 5-fold by sequence / Frechet)    partgap.py:148
  batch 16, bf16 autocast, real part points = the model's own structure-point head         partgap.py:76-93
Changed (clean protocol):
  real side 2,000 crops evenly spaced (np.linspace, as partgap.py:56) from the CLEAN MAV6D chain's final-detector orientation crops on
            unlabeled TRAIN frames: D:/S2MUAV3D/srcx_work/in2mav/oc_tr1.npy (12,866 crops, det1 windows, srcx/run_x.sh Module 5),
            normalised with the same real stats (= sx 'mav' rn = partgap REAL_MEAN/STD); group = sequence.  No label is read.
  dropped   the 'hard_vs_realness' Spearman (needed dev pseudo-label errors from kzst9; not in the paper sentence).
  batch     $PG_BS (default 16) for OOM halving; memory fraction 0.2 as original; SMOKE=1 -> CPU, 12 sim + 24 real, seed 0 only.
  saves     raw pooled features per seed (feats_KPSs<seed>.npz: every layer x pool, sim + real, + domain/group) for the t-SNE figure.
Outputs (this dir): partgap_clean.json, PARTGAP_CLEAN_SUMMARY.md, feats_KPSs*.npz.   usage: python partgap_clean.py"""
import os, sys, json, pickle, time
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
O = os.environ.get('PG_OUT', G + '/gproto/review1113/cleanfix/B2_etasq')
SMOKE = os.environ.get('SMOKE') == '1'
for p in ('E:/Open3DUAVDet', G, G + '/gap', G + '/lidar/kpsize', G + '/lidar/kzbody'):
    sys.path.insert(0, p)
import numpy as np, torch, torch.nn.functional as F
dev = torch.device('cpu' if SMOKE else 'cuda')
if dev.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('PG_MEMFRAC', '0.2')))
import kps_eval as KE
KE.register()
import domain_gap as DG                     # metrics() (chdir to tools on import)
BS = int(os.environ.get('PG_BS', '16'))
REAL_MEAN = np.array([0.436, 0.4418, 0.4374], np.float32)[:, None, None]
REAL_STD = np.array([0.2239, 0.2244, 0.22], np.float32)[:, None, None]

SIM = pickle.load(open(G + '/partgap/sim_samples.pkl', 'rb'))
CR = 'D:/S2MUAV3D/srcx_work/in2mav'
crops = np.load(CR + '/oc_tr1.npy', mmap_mode='r'); info = pickle.load(open(CR + '/oc_tr1.pkl', 'rb'))['info']
ridx = np.linspace(0, len(crops) - 1, 2000).astype(int)
if SMOKE:
    SIM = SIM[::len(SIM) // 12][:12]; ridx = ridx[::len(ridx) // 24][:24]
REAL = [dict(i=int(i), grp='r:' + str(info[i]['seq'])) for i in ridx]
print('sim samples', len(SIM), '| real clean train crops', len(REAL), 'of', len(crops), flush=True)

LAYERS = ['layer2', 'layer3', 'layer4', 'head']
POOLS = ['win', 'ctr', 'motor', 'nose', 'tail', 'top', 'bottom', 'body8']


def sample(f, pts, H=288, W=512):
    g = torch.stack([(pts[..., 0] + 0.5) / W * 2 - 1, (pts[..., 1] + 0.5) / H * 2 - 1], -1)[:, :, None]
    return F.grid_sample(f, g, mode='bilinear', align_corners=False)[..., 0].permute(0, 2, 1)


def run(m, X, pts_given):
    feats = {}
    hooks = [getattr(m.backbone_2d, n).register_forward_hook(lambda mod, i, o, n=n: feats.__setitem__(n, o)) for n in LAYERS[:3]]
    out = {L: {p: [] for p in POOLS} for L in LAYERS}
    for c0 in range(0, len(X), BS):
        x = torch.from_numpy(np.stack(X[c0:c0 + BS]).astype(np.float32)).to(dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            tb = {'image': x[:, None], 'batch_size': len(x)}
            for mod in m.module_list:
                tb = mod(tb)
            f = tb['features_2d'][:, 0]
            kk = m.kps_head(f).float()
        feats['head'] = f
        hm = tb['pred_center_dict']['hm'].float()[:, 0]
        cres = tb['pred_center_dict']['center_res'].float()
        if pts_given is not None:
            pts = torch.from_numpy(np.stack(pts_given[c0:c0 + BS])).to(dev)
        else:
            B, h, w = hm.shape
            j = hm.reshape(B, -1).argmax(1); r, c = j // w, j % w
            ar = torch.arange(B, device=dev)
            uc = (c.float() + cres[ar, 0, r, c]) * 8; vc = (r.float() + cres[ar, 1, r, c]) * 8
            off = kk[ar, :16, r, c].reshape(B, 8, 2) * 64.0
            pts = torch.stack([uc, vc], -1)[:, None] + off
        for L in LAYERS:
            fl = feats[L].float()
            h, w = fl.shape[-2:]
            out[L]['win'].append(fl.mean((2, 3)).cpu())
            out[L]['ctr'].append(fl[:, :, int(h * .33):max(int(h * .67), int(h * .33) + 1), int(w * .39):max(int(w * .61), int(w * .39) + 1)].mean((2, 3)).cpu())
            s = sample(fl, pts.float())
            out[L]['motor'].append(s[:, :4].mean(1).cpu())
            for k, nm in enumerate(['nose', 'tail', 'top', 'bottom']):
                out[L][nm].append(s[:, 4 + k].cpu())
            out[L]['body8'].append(s.mean(1).cpu())
    for h_ in hooks:
        h_.remove()
    return {L: {p: torch.cat(v).numpy().astype(np.float64) for p, v in d.items()} for L, d in out.items()}


Xs = [s['x'] for s in SIM]; Ps = [s['pts'] for s in SIM]; gs = [s['grp'] for s in SIM]
Xr = []
for r in REAL:
    im = np.asarray(crops[r['i']], np.float32).transpose(2, 0, 1) / 255.0
    Xr.append(((im - REAL_MEAN) / REAL_STD).astype(np.float16))
gr = [r['grp'] for r in REAL]
JS = O + '/partgap_clean%s.json' % ('_smoke' if SMOKE else '')
res = json.load(open(JS)) if os.path.exists(JS) else {}
for seed in ((0,) if SMOKE else (0, 1, 2)):
    name = 'KPSs%d' % seed
    fz = O + '/feats_%s%s.npz' % (name, '_smoke' if SMOKE else '')
    if name in res and os.path.exists(fz):
        continue
    t0 = time.time()
    m = KE.E.build(KE.E.CK % seed, torch, dev)
    fs = run(m, Xs, Ps)
    fr = run(m, Xr, None)
    np.savez_compressed(fz, domain=np.array(['sim'] * len(Xs) + ['real'] * len(Xr)), group=np.array(list(gs) + list(gr)),
                        real_crop_index=ridx, **{'%s__%s' % (L, p): np.concatenate([fs[L][p], fr[L][p]]).astype(np.float32) for L in LAYERS for p in POOLS})
    r = {L: {p: DG.metrics(fs[L][p], fr[L][p], gs, gr) for p in POOLS} for L in LAYERS}
    r['n_sim'] = len(Xs); r['n_real'] = len(Xr); r['real_source'] = CR + '/oc_tr1.npy (clean in2mav det1 train crops, linspace 2000)'
    res[name] = r
    json.dump(res, open(JS, 'w'), indent=1)
    print(name, '%.0f s' % (time.time() - t0), ' '.join('%s/%s:eta2 %.3f auc %.3f' % (L, p, r[L][p]['eta2'], r[L][p]['auc']) for L in LAYERS for p in ('win', 'ctr', 'motor', 'body8')), flush=True)
    del m
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
seeds = [s for s in (0, 1, 2) if 'KPSs%d' % s in res]
lines = ['# cleanfix B2: part-level domain gap on CLEAN real crops (srcx/in2mav det1 train crops); sim = partgap sim_samples (885)', '',
         'seeds %s; n_sim %d, n_real %d' % (seeds, len(Xs), len(Xr)), '',
         '| layer | pool | AUC mean | eta2 | Frechet |', '|---|---|---|---|---|']
for L in LAYERS:
    for p in POOLS:
        a = [res['KPSs%d' % s][L][p] for s in seeds]
        lines.append('| %s | %s | %.3f | %.3f | %.3f |' % (L, p, np.mean([x['auc'] for x in a]), np.mean([x['eta2'] for x in a]), np.mean([x['fd'] for x in a])))
open(O + '/PARTGAP_CLEAN_SUMMARY%s.md' % ('_smoke' if SMOKE else ''), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
