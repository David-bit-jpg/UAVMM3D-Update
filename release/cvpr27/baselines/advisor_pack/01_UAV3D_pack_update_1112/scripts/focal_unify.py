# -*- coding: utf-8 -*-
"""AntiUAV3D focal unification (11-12, the advisor's suggestion): rescale every image to one canonical focal length so a
monocular method sees the same "pixel size <-> distance" relation as in the simulated source. This is a preprocessing
option for the baseline methods only.

AntiUAV3D uses a zoom camera, so fx varies frame to frame (about 1,500-30,000 px at 1920 width). The CARLA outdoor
source is fixed at fx = 640 at 1280 width, i.e. 960 px at 1920 width.
Rule: s = min(1, f_canon / fx). The image is scaled by s about the principal point and centred on a black canvas of
the original 1920x1080 size; K' = [s*fx, s*fy, cx', cy']. Frames with fx <= f_canon keep their pixels: they are never
enlarged, because enlarging would crop away part of the field of view.
The 3D camera frame is unchanged (only pixels and K change), so the 3D boxes predicted on these images are scored with
the original test ground truth, unmodified; the scoring script reads the GT from data/gt/ and needs no change.
No ground truth is read here (labels.pkl only carries file / K / W / H; the boxes field is empty).

Output:
  data/cmpfull/outdoor_train_fu/, data/cmpfull/outdoor_test_fu/   (images + labels.pkl, same format)
  data/cmpfull/pcmd_teacher_pseudo_outdoor_train_fu.pkl            (= copy of the PCMD teacher pseudo-labels; 3D is unchanged)
Usage: python scripts/focal_unify.py [--f-canon 960] [--workers 8]"""
import argparse, os, pickle, shutil, sys
from multiprocessing import Pool
import numpy as np
import cv2

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PKG, 'data', 'cmpfull')


def one(job):
    src, dst, s, W, H, cx, cy = job
    if os.path.exists(dst):
        return
    if s >= 1.0:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    small = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    ox, oy = int(round(cx - s * cx)), int(round(cy - s * cy))     # keep the principal point where it was
    can = np.zeros_like(im)
    x0, y0 = max(ox, 0), max(oy, 0); x1, y1 = min(ox + w, W), min(oy + h, H)
    can[y0:y1, x0:x1] = small[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    cv2.imwrite(dst, can, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f-canon', type=float, default=960.0, help='canonical fx in px at 1920 width (default = CARLA outdoor source)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    for sp in ('outdoor_train', 'outdoor_test'):
        L = pickle.load(open(os.path.join(DATA, sp, 'labels.pkl'), 'rb'))
        od = os.path.join(DATA, sp + '_fu'); os.makedirs(os.path.join(od, 'images'), exist_ok=True)
        jobs, recs, ss = [], [], []
        for r in L:
            K = np.asarray(r['K'], np.float64).reshape(3, 3); W, H = int(r['W']), int(r['H'])
            fcan = a.f_canon * W / 1920.0
            s = min(1.0, fcan / K[0, 0])
            cx, cy = K[0, 2], K[1, 2]
            ox, oy = (int(round(cx - s * cx)), int(round(cy - s * cy))) if s < 1 else (0, 0)
            K2 = K.copy()
            if s < 1:
                K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] = s * cx + ox; K2[1, 2] = s * cy + oy
            jobs.append((os.path.join(DATA, sp, r['file']), os.path.join(od, r['file']), s, W, H, cx, cy))
            d = dict(r); d['K'] = K2; d['fu_scale'] = s; recs.append(d); ss.append(s)
        with Pool(a.workers) as p:
            for i, _ in enumerate(p.imap_unordered(one, jobs, chunksize=16)):
                if i % 500 == 0:
                    print('  %s %d/%d' % (sp, i, len(jobs)), flush=True)
        pickle.dump(recs, open(os.path.join(od, 'labels.pkl'), 'wb'))
        ss = np.array(ss)
        print('%s_fu: %d frames; scale s P10/50/90 = %.3f/%.3f/%.3f; unchanged (s=1) %.0f%%' % (
            sp, len(recs), *np.percentile(ss, [10, 50, 90]), 100 * np.mean(ss >= 1)))
    shutil.copy2(os.path.join(DATA, 'pcmd_teacher_pseudo_outdoor_train.pkl'), os.path.join(DATA, 'pcmd_teacher_pseudo_outdoor_train_fu.pkl'))
    print('done; f_canon = %.0f px at 1920 width' % a.f_canon)


if __name__ == '__main__':
    main()
