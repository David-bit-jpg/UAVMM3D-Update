# -*- coding: utf-8 -*-
"""无标签版位置伪标签（11-10，导师：任何超参数都不许看真实标签）。
与 make_strict_pseudo.py（11-07，SEL=lb）的区别：
  - 全程不读任何真值（不读 dump 里的 gt、不做真值诊断）；
  - 校准：c = LAM × 全训练集修正量中位（LAM 缺省 1，即不缩；0.7/0.75/0.8 只做补充材料的敏感性分析）；
  - 筛帧：只按检测置信度 ≥ 0.5（不再用当时对照训练集真值挑出的"自估不确定度筛选"）；
  - 写出的索引只保留图像信息 + 伪框，不从真值索引复制 cls / center_inside / object_frame。
输入与 11-07 全严格版相同：ST2 五集成无时序背景检测（real_train_preds_st2nb.pkl）+ 第一代第二段修正量（st2m/delta_IR_nb.pkl）。
用法：LAM=1.0 TAG=nl100 python make_nolabel_pseudo.py"""
import os, sys, pickle, json, time
import numpy as np
ROOT = '@@ROOT@@'; G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT + '/tools'); sys.path.insert(0, G)
from anchor_eval import deploy, GUARD

LAM = float(os.environ.get('LAM', '1.0'))
TAG = os.environ['TAG']
CONF = 0.5
D = pickle.load(open(ROOT + '/output/camnorm/st/real_train_preds_st2nb.pkl', 'rb'))
DI = pickle.load(open(G + '/st2m/delta_IR_nb.pkl', 'rb'))['train']
frames = D['frames']; assert len(frames) == len(DI)
# 只取预测与置信度，不碰 frames 里的 gt / matched / cls
recs = [dict(seq_id=f['seq'], frame_id=f['frame'], pred=np.asarray(f['pred'], np.float64).reshape(-1, 9),
             conf=np.asarray(f['conf'], np.float64).reshape(-1)) for f in frames]
prI = [(np.asarray(e['d'], np.float64).reshape(-1), np.asarray(e['lb'], np.float64).reshape(-1)) for e in DI]

d0 = np.array([prI[i][0][0] if len(prI[i][0]) else np.nan for i in range(len(recs))])
okd = np.isfinite(d0) & (np.abs(d0) <= GUARD)
cg = float(np.median(d0[okd]))
nr = deploy(recs, prI, [np.full(len(r['pred']), LAM * cg) for r in recs])
keep = np.array([len(r['conf']) > 0 and float(r['conf'].max()) >= CONF for r in recs])

idx = pickle.load(open('@@MMCACHE@@/mav6d_cn_obj/train/index.pkl', 'rb'))
lwh = np.asarray(D['info']['calib_lwh'], np.float64)          # 常数尺寸：检测器在无标签验证集图像上自标定
key2i = {(m['seq'], m['frame']): i for i, m in enumerate(idx['metas']) if m}
IMG_KEYS = ('seq', 'frame', 'K_in', 'K_raw', 'raw_wh', 'undistorted')
new_metas = [None] * len(idx['metas']); valid = []
for i in np.where(keep)[0]:
    r = recs[i]; ii = key2i.get((r['seq_id'], r['frame_id']))
    if ii is None:
        continue
    p = nr[i]['pred'][int(np.argmax(r['conf']))]
    src = idx['metas'][ii]
    m = {k: src[k] for k in IMG_KEYS}
    m['boxes9d'] = np.concatenate([p[:3], lwh, p[6:9]]).astype(np.float32)[None]
    m['names'] = ['drone']; m['qualified'] = np.array([True]); m['pseudo'] = True; m['rot_ok'] = True
    new_metas[ii] = m; valid.append(ii)
out = {k: v for k, v in idx.items() if k != 'metas'}
out['metas'] = new_metas; out['valid_idx'] = np.asarray(sorted(valid), dtype=idx['valid_idx'].dtype)
out['pseudo'] = {'pipeline': '无标签版：ST2 五集成（无时序背景）+ 第一代第二段修正，c = %.2f × 训练集修正量中位；只按置信度 >= %.1f 筛帧；不读真值' % (LAM, CONF),
                 'calib_lwh': lwh.tolist(), 'lam': LAM, 'c_global': cg, 'conf': CONF}
op = '@@MMCACHE@@/mav6d_cn_obj/train/index_pseudo_%s.pkl' % TAG
pickle.dump(out, open(op, 'wb'))
msg = '%s 写出 %s：%d 帧；c_global=%.4f，LAM=%.2f，置信 >= %.1f' % (time.strftime('%m-%d %H:%M:%S'), op, len(valid), cg, LAM, CONF)
print(msg)
open(G + '/gproto/nolabel/make_nolabel_pseudo.log', 'a', encoding='utf-8').write(msg + '\n')
