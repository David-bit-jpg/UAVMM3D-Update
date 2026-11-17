# -*- coding: utf-8 -*-
"""KZ-ST7 训练（prereg_kzst7.json）：kzst2_train 原样（同 KZ-ST6），只换数据/输出目录到 kzst7/{data,train}_<臂>。
    python kzst7_train.py --arm q|w --student k"""
import sys
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst2', SCR + '/kzst1'):
    sys.path.insert(0, p)
import kzst2_common as S   # noqa: E402
argv = sys.argv[1:]
i = argv.index('--arm')
arm = argv[i + 1]
del argv[i:i + 2]
assert arm in ('q', 'w')
OUT = '@@ROOT@@/output/camnorm/geo1004/kzst7'
S.OUT = OUT
S.DATA = OUT + '/data_' + arm
S.TRAIN_DIR = OUT + '/train_' + arm
S.PREREG = OUT + '/prereg_kzst7.json'
S.RELEASE_KZ = [SCR + '/kzst1/adabn_kzm/kz_s%d_adabn_kzm.pth' % k for k in range(5)]
import kzst2_train as T    # noqa: E402

if __name__ == '__main__':
    if '--data-dir' not in argv:
        argv += ['--data-dir', S.DATA]
    if '--out-root' not in argv:
        argv += ['--out-root', S.TRAIN_DIR]
    T.main(argv)
