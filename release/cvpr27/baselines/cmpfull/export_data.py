# -*- coding: utf-8 -*-
"""整图对比实验（AntiUAV3D 表 2 的跑法：对比方法用官方代码、整幅图输入、自己检测）的统一数据导出。
所有方法读同一份：D:/cmpfull/<split>/images/*.jpg + D:/cmpfull/<split>/labels.pkl
  sim_train  仿真训练（E:/mmcache/full_v2/train，1280×720，带标签）
  sim_val    仿真验证（E:/mmcache/full_v2/test，带标签）
  real_train 真实训练集图像（MAV6D train，去畸变 512×288，**不给标签**，只给自训练用）
  real_test  真实测试集图像（MAV6D test，去畸变 512×288）；标签单独放 real_test/gt_sealed.pkl，只评测脚本读
labels.pkl = list of dict(file, seq, frame, W, H, K(3×3, 对应存下来的这张图), boxes=[dict(center(3, 相机系 x右 y下 z前，框中心), lwh(3), euler_xyz(3), R(3×3), cls)])
框定义：角点 = (PROTO*lwh) @ R.T + center，PROTO = {±0.5}^3，R = Rot.from_euler('xyz', euler_xyz)（物体 → 相机）。
python export_data.py"""
import os, pickle, concurrent.futures as cf
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
OUT = 'D:/cmpfull'


def box(b9, cls):
    b9 = np.asarray(b9, np.float64)
    return dict(center=b9[:3], lwh=b9[3:6], euler_xyz=b9[6:9], R=Rot.from_euler('xyz', b9[6:9]).as_matrix(), cls=cls)


def sim(split_in, split_out):
    root = 'E:/mmcache/full_v2/' + split_in
    d = pickle.load(open(root + '/index.pkl', 'rb'))
    idx = np.load(root + '/rgb_jpg_index.npy')
    os.makedirs('%s/%s/images' % (OUT, split_out), exist_ok=True)
    blob = np.memmap(root + '/rgb_jpg.bin', dtype=np.uint8, mode='r')
    recs = []
    for i, m in enumerate(d['metas']):
        q = np.asarray(m['qualified']).reshape(-1)
        bs = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)
        boxes = [box(bs[j], m['names'][j]) for j in range(len(bs)) if q[j]]
        recs.append(dict(file='images/%06d.jpg' % i, seq=m['seq'], frame=m['frame'], W=int(m['raw_wh'][0]), H=int(m['raw_wh'][1]),
                         K=np.asarray(m['K_raw'], np.float64).reshape(3, 3), boxes=boxes))

    def w(i):
        o, n = idx[i]
        p = '%s/%s/%s' % (OUT, split_out, recs[i]['file'])
        if not os.path.exists(p):
            open(p, 'wb').write(bytes(blob[o:o + n]))
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(w, range(len(recs))))
    pickle.dump(recs, open('%s/%s/labels.pkl' % (OUT, split_out), 'wb'))
    print(split_out, len(recs), '帧，有框', sum(len(r['boxes']) > 0 for r in recs), flush=True)


def real(split_in, split_out, with_labels):
    root = 'E:/mmcache/mav6d_cn_obj/' + split_in
    d = pickle.load(open(root + '/index.pkl', 'rb'))
    arr = np.load(root + '/rgb.npy', mmap_mode='r')
    os.makedirs('%s/%s/images' % (OUT, split_out), exist_ok=True)
    recs, gt = [], []
    for i, m in enumerate(d['metas']):
        r = dict(file='images/%06d.jpg' % i, seq=m['seq'], frame=m['frame'], W=int(d['W']), H=int(d['H']),
                 K=np.asarray(m['K_in'], np.float64).reshape(3, 3), boxes=[])
        bs = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)
        g = [box(b, m['cls']) for b in bs]
        if with_labels:
            r['boxes'] = g
        recs.append(r); gt.append(dict(seq=m['seq'], frame=m['frame'], boxes=g))

    def w(i):
        p = '%s/%s/%s' % (OUT, split_out, recs[i]['file'])
        if not os.path.exists(p):
            cv2.imwrite(p, np.ascontiguousarray(arr[i]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(w, range(len(recs))))
    pickle.dump(recs, open('%s/%s/labels.pkl' % (OUT, split_out), 'wb'))
    if split_out == 'real_test':
        pickle.dump(gt, open('%s/%s/gt_sealed.pkl' % (OUT, split_out), 'wb'))
    print(split_out, len(recs), '帧', flush=True)


if __name__ == '__main__':
    sim('train', 'sim_train')
    sim('test', 'sim_val')
    real('train', 'real_train', with_labels=False)
    real('test', 'real_test', with_labels=False)
