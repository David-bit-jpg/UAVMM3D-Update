# -*- coding: utf-8 -*-
"""11-10 无标签链："去真值帧表"。只从各划分 index.pkl 取每一行的 seq / frame / K_in（图像几何，非标签），
丢掉 boxes9d / names / qualified / center_inside / cls 等全部真值字段。下游脚本只读本文件取图像与内参，不再打开 index.pkl。
输出 nolabel/{train,val,test}_frames_nogt.pkl：dict(keys=[(seq, frame)], K_in=(N,3,3))，行号 = 该划分 rgb.npy 的行号。"""
import pickle, numpy as np
for sp in ('train', 'val', 'test'):
    d = pickle.load(open('@@MMCACHE@@/mav6d_cn_obj/%s/index.pkl' % sp, 'rb'))
    keys = [(str(m['seq']), str(m['frame'])) for m in d['metas']]
    K = np.stack([np.asarray(m['K_in'], np.float64) for m in d['metas']])
    del d
    pickle.dump(dict(keys=keys, K_in=K), open('@@ROOT@@/output/camnorm/geo1004/gproto/nolabel/%s_frames_nogt.pkl' % sp, 'wb'))
    print('%s_frames_nogt：%d 行' % (sp, len(keys)))
