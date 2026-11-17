# -*- coding: utf-8 -*-
"""11-10：把室外小真实集（D:/S2MUAV3D/outdoor_antiuav3d，取自 AntiUAV3D）导出成对手方法统一读的格式，与 export_data.py 的 real_* 相同：
  D:/cmpfull/outdoor_<split>/images/<seq>__<frame>.jpg（原图 1920×1080，硬链接，不重新编码）
  D:/cmpfull/outdoor_<split>/labels.pkl = list of dict(file, seq, frame, W, H, K(本帧内参，变焦相机每帧不同), boxes=[])
  outdoor_test/gt_sealed.pkl：测试真值，只给我们自己的评测脚本读，不进给导师的包、不进任何训练。
train 的标签在源数据里已封存，这里也只用它的图像信息（文件名、内参）。
python export_outdoor.py"""
import os, pickle, shutil
import numpy as np
from scipy.spatial.transform import Rotation as Rot
SRC = 'D:/S2MUAV3D/real_outdoor_antiuav3d'   # 11-10：50 m 以内版
OUT = 'D:/cmpfull'


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


for sp in ('train', 'val', 'test'):
    inp = pickle.load(open('%s/labels/%s.pkl' % (SRC, sp), 'rb'))          # train / test：只有图像信息；val：带标签，这里只取图像信息
    od = '%s/outdoor_%s' % (OUT, sp)
    os.makedirs(od + '/images', exist_ok=True)
    recs = []
    for r in inp:
        fn = 'images/%s__%s' % (r['seq'], r['frame'])
        link('%s/%s' % (SRC, r['img']), '%s/%s' % (od, fn))
        recs.append(dict(file=fn, seq=r['seq'], frame=r['frame'], W=int(r['W']), H=int(r['H']),
                         K=np.asarray(r['K'], np.float64).reshape(3, 3), boxes=[]))
    pickle.dump(recs, open(od + '/labels.pkl', 'wb'))
    if sp == 'test':
        gt = pickle.load(open('%s/labels/sealed/test_gt.pkl' % SRC, 'rb'))
        assert [(g['seq'], g['frame']) for g in gt] == [(r['seq'], r['frame']) for r in recs]
        sealed = []
        for g in gt:
            b = np.asarray(g['box9d_ours'], np.float64)
            sealed.append(dict(seq=g['seq'], frame=g['frame'], aircraft=g['aircraft'], cat=g['cat'], dist_m=g['dist_m'],
                               boxes=[dict(center=b[:3], lwh=b[3:6], euler_xyz=b[6:9], R=Rot.from_euler('xyz', b[6:9]).as_matrix(), cls=g['aircraft'])]))
        pickle.dump(sealed, open(od + '/gt_sealed.pkl', 'wb'))
    print('outdoor_%s：%d 帧%s' % (sp, len(recs), '（测试真值另存 gt_sealed.pkl）' if sp == 'test' else ''))
