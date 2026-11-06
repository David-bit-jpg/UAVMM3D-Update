# -*- coding: utf-8 -*-
"""KZ-ST6 训练（起点 KZ-IRLiD + AdaBN）：kzst2_train 原样，只改 起点 = AdaBN KZ、数据/输出目录 = geo1004/kzst6、预登记 = prereg_kzst6.json。
    python kzst3_train.py --student k [--data-dir ... --out-root ... --epochs 1 --max-frames 16（冒烟）]"""
import sys
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst2', SCR + '/kzst1'):
    sys.path.insert(0, p)
import kzst2_common as S   # noqa: E402
OUT = '@@ROOT@@/output/camnorm/geo1004/kzst6'
S.OUT = OUT
S.DATA = OUT + '/data'
S.TRAIN_DIR = OUT + '/train'
S.PREREG = OUT + '/prereg_kzst6.json'
S.RELEASE_KZ = [SCR + '/kzst1/adabn_kzm/kz_s%d_adabn_kzm.pth' % k for k in range(5)]
import kzst2_train as T    # noqa: E402

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--data-dir' not in argv:
        argv += ['--data-dir', S.DATA]
    if '--out-root' not in argv:
        argv += ['--out-root', S.TRAIN_DIR]
    T.main(argv)
