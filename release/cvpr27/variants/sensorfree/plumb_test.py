# -*- coding: utf-8 -*-
"""[review1113 sensorfree] CPU plumbing test of the matte swaps (no labels of any real split, no GPU, writes only sensorfree/plumb/).
  1) refiner sim_batch (E1, E2, E2a, E2b, E4 lib copies): composite rate and which array is the alpha / which is the silhouette target.
     Synthetic check: rgb = 0, background bank = 255, appearance aug off, prep = identity -> a composited crop is 255 * (1 - alpha), so the
     alpha used can be read back exactly and compared with the box hull and with the IR silhouette (same flip as the returned target).
  2) orientation-student source dataset (E1 lib's ou_train_x patch executed verbatim on a real KP.CacheDS): composite alpha read back the same
     way (constant background, augmentation off) and compared with hull_<src>.npy and with the cache's a_v (IR x LiDAR)."""
import os, sys, re, importlib.util
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
PL = HERE + '/plumb'; os.makedirs(PL, exist_ok=True)
which = sys.argv[1] if len(sys.argv) > 1 else 'all'


def load_mod(lib, name, env):
    os.environ.update(env)
    for k in [k for k in sys.modules if k in ('sx', name)]:
        del sys.modules[k]
    sys.path.insert(0, lib)
    spec = importlib.util.spec_from_file_location(name + '_' + str(abs(hash(lib))), lib + '/' + name + '.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.path.remove(lib)
    return m


def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 1.0


if which in ('all', 'refiner'):
    hull_m = np.load(GP + '/review1113/boxhull/in2mav/ref/sim_sil.npy', mmap_mode='r')
    ir_m = np.load(GP + '/srcx/in2mav/ref/sim_sil.npy', mmap_mode='r')
    N = 400; pick = np.random.default_rng(0).choice(len(hull_m), N, replace=False); pick.sort()
    hull = np.asarray(hull_m[pick]); ir = np.asarray(ir_m[pick])
    for arm, teach, target, matte in (('E1', 'none', hull, None), ('E2', 'full', hull, None), ('E2a', 'full', hull, ir), ('E2b', 'full', ir, hull), ('E4', 'none', ir, None)):
        tgt = 'mav'; lib = '%s/%s/%s_s0/lib' % (HERE, arm, tgt)
        if not os.path.isdir(lib):
            continue
        R = load_mod(lib, 'refiner_x', dict(SX_TEACH=teach, SX_SRC='in', SX_TGT=tgt, SX_SEEDSHIFT='0'))
        d = PL + '/ref_' + arm; os.makedirs(d, exist_ok=True)
        if matte is not None:
            np.save(d + '/sim_matte.npy', matte)
        elif os.path.exists(d + '/sim_matte.npy'):
            os.remove(d + '/sim_matte.npy')
        R.RO = d; R._SF_MT.clear()
        R.dt.prep = lambda x, rng, strong: x                       # identity (inspection only)
        rgb = np.zeros((N, 128, 128, 3), np.uint8); bank = np.full((4, 128, 128, 3), 255, np.uint8)
        S = np.zeros((N, 5), np.float32); dl = np.zeros(N, np.float32)
        rng = np.random.default_rng(0); idx = np.arange(N)
        X, _, _, mm = R.sim_batch(rgb, target, S, dl, idx, bank, rng, strong=False)
        X = np.asarray(X); mm = mm.numpy() > 0.5
        a_used = 1.0 - X[..., 0].astype(np.float32) / 255.0          # alpha actually used (0 everywhere if no composite)
        comp = (X[..., 0] > 127).reshape(N, -1).any(1)                # a composite leaves background (255) somewhere
        # which array was the alpha?  compare the used alpha with hull / IR in the same (flipped) frame as the returned target mm
        flip = np.array([not np.array_equal(mm[n], target[n] > 127) for n in range(N)])
        def fr(A, n):
            return (A[n][:, ::-1] if flip[n] else A[n]) > 127
        ih = [iou(a_used[n] > 0.5, fr(hull, n)) for n in np.where(comp)[0]]
        ii = [iou(a_used[n] > 0.5, fr(ir, n)) for n in np.where(comp)[0]]
        th = np.mean([np.array_equal(mm[n], fr(hull, n)) for n in range(N)]); ti = np.mean([np.array_equal(mm[n], fr(ir, n)) for n in range(N)])
        print('[refiner %-3s] SILW %.0f SF_COMPOSITE %s NOIR %s | composites %d/%d (%.0f%%) | alpha IoU vs hull %.3f, vs IR %.3f | target == hull %.2f, == IR %.2f'
              % (arm, R.SILW, R.SF_COMPOSITE, R.sx.NOIR, comp.sum(), N, 100 * comp.mean(), np.median(ih) if ih else np.nan, np.median(ii) if ii else np.nan, th, ti), flush=True)

if which in ('all', 'student'):
    for src, tgt in (('in', 'mav'), ('out', 'antifs')):
        lib = '%s/E1/%s_s0/lib' % (HERE, tgt)
        os.environ.update(SX_TEACH='none', SX_SRC=src, SX_TGT=tgt, SX_SEEDSHIFT='0', SX_NOMKDIR='1')
        for k in [k for k in sys.modules if k in ('sx',)]:
            del sys.modules[k]
        sys.path.insert(0, lib); import sx  # noqa
        assert sx.O.startswith(HERE + '/E1/'), sx.O
        sys.path.insert(0, GP + '/wf20/design'); sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
        import kzpaste_train as KP  # noqa
        from easydict import EasyDict
        from uavdet3d.config import cfg_from_yaml_file
        from uavdet3d.utils import common_utils
        logger = common_utils.create_logger()
        cfg = EasyDict(); cfg_from_yaml_file(sx.S['kp_cfg'], cfg)
        xbg = {'PROB': 1.0, 'BANK': '', 'MATTE': 'v'}
        src_ds = KP.CacheDS(cfg.DATA_CONFIG, sx.S['sim_crops'], training=True, logger=logger, xbg=xbg, seed=0)
        # execute the E1 patch lines verbatim from the lib copy
        txt = open(lib + '/ou_train_x.py', encoding='utf-8').read()
        blk = re.search(r"\n(    _HM = .*?\n    logger\.info\('\[sensorfree E1\].*?\n)", txt, re.S).group(1)
        class A: paste = 1.0
        ns = dict(np=np, sx=sx, src=src_ds, logger=logger, a=A)
        exec('\n'.join(l[4:] for l in blk.split('\n')), ns)
        assert src_ds._c['a_v'] is ns['_H']
        src_ds._bank = np.full((1, 288, 512, 3), 255, np.uint8)
        src_ds._augment = lambda image, boxes, names, K, with_rot_ok=False: (image, boxes, names, K) if not with_rot_ok else (image, boxes, names, K, None)
        src_ds.norm_mean = None
        src_ds._xrng = np.random.default_rng(0)
        H = ns['_H']; AV = np.load(sx.S['sim_crops'] + '/a_v.npy', mmap_mode='r')
        cand = [n for n in range(len(src_ds.cmeta)) if src_ds.cmeta[n]['has_matte']][:60]
        res = []
        for n in cand:
            rgb0 = np.asarray(src_ds._c['rgb'][n]).astype(np.float32)
            d = src_ds.__getitem__(n)
            img = d['image'][0][:3].transpose(1, 2, 0) * 255.0 if d['image'].ndim == 4 else d['image'][:3].transpose(1, 2, 0) * 255.0
            # composite view = a*rgb + (1-a)*255 -> a = (255 - view) / (255 - rgb) where rgb < 245
            ok = rgb0.mean(2) < 200
            if ok.sum() < 100:
                continue
            a = np.clip((255.0 - img.mean(2)) / np.maximum(255.0 - rgb0.mean(2), 1e-3), 0, 1)
            au = (a > 0.5) & ok
            res.append((iou(au, (np.asarray(H[n]) > 127) & ok), iou(au, (np.asarray(AV[n]) > 127) & ok)))
        res = np.array(res)
        print('[student E1 %s] composites %d tried / %d done | used alpha IoU vs box hull median %.3f (min %.3f) | vs IR x LiDAR a_v median %.3f'
              % (src, src_ds.n_try, src_ds.n_paste, np.median(res[:, 0]), res[:, 0].min(), np.median(res[:, 1])), flush=True)
        sys.path.remove(lib)
