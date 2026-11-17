# -*- coding: utf-8 -*-
"""[corrupt_1117] test-time image corruptions (inference only). Deterministic function of (corruption, severity, seq, frame, image width).
Applied identically to (a) the 512x288 test cache frames read by the detector and (b) the raw frames returned by sx.imread_raw for
the test crops (orientation crops te1 and the depth-refiner crops). Pixel-scale quantities are defined at 512 px width and scaled by W/512.
  lowlight s1/s2 : x <- (k*x)^1.4 with k = 0.35 / 0.20 (x in [0,1]), then additive Gaussian noise sigma 4/255 (fixed, per pixel/channel)
  noise    s1/s2 : additive Gaussian noise sigma 12/255 / 25/255 (per pixel and channel)
  blur     s1/s2 : Gaussian blur sigma 1.5 / 3.0 px at 512-px scale (x W/512, i.e. 5.625 / 11.25 px on 1920-px raw frames)
  down     s1/s2 : downscale by 2 / 3 (INTER_AREA) and upscale back to the same size (INTER_LINEAR)
  haze     s1/s2 : x <- (1-a) x + a * 128/255 with a = 0.4 / 0.6
All outputs are rounded and clipped to uint8.
  python corrupt.py cache <tgt antifs|mav> <corr> <sev>   -> D:/S2MUAV3D/corrupt_1117/<tgt>/<corr>_s<sev>/test/{rgb.npy,index.pkl}"""
import os, sys, zlib, shutil, pickle, time
import numpy as np, cv2

CORR = ('lowlight', 'noise', 'blur', 'down', 'haze')
SRC_CACHE = {'antifs': 'D:/mmcache/outdoor_real_fs', 'mav': 'E:/mmcache/mav6d_cn_obj'}
DST = 'D:/S2MUAV3D/corrupt_1117'


def _rng(corr, sev, seq, frame, w):
    return np.random.default_rng([zlib.crc32(corr.encode()), int(sev), zlib.crc32(str(seq).encode()), zlib.crc32(str(frame).encode()), int(w)])


def apply(img, corr, sev, seq, frame):
    """img: HxWx3 uint8 (BGR). Returns a corrupted uint8 copy."""
    sev = int(sev); H, W = img.shape[:2]; sc = W / 512.0
    if corr == 'lowlight':
        k = {1: 0.35, 2: 0.20}[sev]
        x = (k * img.astype(np.float32) / 255.0) ** 1.4
        x += _rng(corr, sev, seq, frame, W).standard_normal(x.shape, dtype=np.float32) * (4.0 / 255.0)
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    if corr == 'noise':
        s = {1: 12.0, 2: 25.0}[sev]
        x = img.astype(np.float32) + _rng(corr, sev, seq, frame, W).standard_normal(img.shape, dtype=np.float32) * s
        return np.clip(np.rint(x), 0, 255).astype(np.uint8)
    if corr == 'blur':
        s = {1: 1.5, 2: 3.0}[sev] * sc
        return cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT101)
    if corr == 'down':
        d = {1: 2, 2: 3}[sev]
        small = cv2.resize(img, (int(round(W / d)), int(round(H / d))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if corr == 'haze':
        a = {1: 0.4, 2: 0.6}[sev]
        x = (1 - a) * img.astype(np.float32) + a * 128.0
        return np.clip(np.rint(x), 0, 255).astype(np.uint8)
    raise ValueError(corr)


def build_cache(tgt, corr, sev):
    src = SRC_CACHE[tgt] + '/test'; dst = '%s/%s/%s_s%d/test' % (DST, tgt, corr, int(sev))
    if os.path.exists(dst + '/DONE'):
        print('cache exists', dst); return dst
    os.makedirs(dst, exist_ok=True)
    idx = pickle.load(open(src + '/index.pkl', 'rb'))
    X = np.load(src + '/rgb.npy', mmap_mode='r')
    Y = np.lib.format.open_memmap(dst + '/rgb.npy', 'w+', np.uint8, X.shape)
    t0 = time.time()
    for i in range(len(X)):
        m = idx['metas'][i]
        Y[i] = apply(np.ascontiguousarray(X[i]), corr, sev, m['seq'], m['frame']) if m else X[i]
    Y.flush(); del Y
    shutil.copy2(src + '/index.pkl', dst + '/index.pkl')
    open(dst + '/DONE', 'w').write('%s %s s%d from %s, %d frames, %.0fs\n' % (tgt, corr, int(sev), src, len(X), time.time() - t0))
    print('cache', dst, len(X), '%.0fs' % (time.time() - t0)); return dst


if __name__ == '__main__':
    if sys.argv[1] == 'cache':
        build_cache(sys.argv[2], sys.argv[3], int(sys.argv[4]))
