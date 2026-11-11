# -*- coding: utf-8 -*-
"""【11-10 无标签链副本 ztrain_nl.py：常数 / 亮度 / 工具取自 chain_eval_m_nl（臂 = NL_ARM，FE_UNC=pp2 = 第二段第 2 步在新检测器训练集检测上的输出），
标签目录 = nolabel/<arm>/tags；输出 nolabel/<arm>/ztrain_<重裁标签>.pkl。用法：NL_ARM=<arm> FE_UNC=pp2 ZT_NMEM=4 python ztrain_nl.py <重裁标签> <源标签>】
wf23 第 2 条 · 第一步：在真实训练帧上算 LiDAR 教的角点测距 z_kp（与部署时 chain_eval_m.py 的 FE_ZKP 逐行同法；CPU，不读标签）。
  朝向网络 = 部署同一组（wf21/final/adabn_all/PSP ×5 + 各自的镜像视图 = 10 个成员），SA-TM 融合角点 → PnP 平移深度；
  PnP 用的机身尺寸与部署同规则（长宽来自检测框，高度 = 该帧颜色原型的高度比常数 × 平均(长, 宽)，常数取自无标签训练集）。
用法：python ztrain.py <重裁标签> <源标签目录> <源标签>
  例：python ztrain.py wsfnb_train_ouPSPallM E:/.../wf21/strict/tags wsfnb_train_kz
输出 ztrain_<重裁标签>.pkl：{(seq, frame): dict(z_kp, kp_sd, det=检测框 9 维)}"""
import os, sys, pickle, time
os.environ.setdefault('FE_CONST', 'train')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import chain_eval_m_nl as CE   # noqa: E402  （借用常数、亮度与 SA-TM/PnP 工具；CPU）
C, KFO, RC = CE.C, CE.KFO, CE.RC

tag, stag = sys.argv[1:3]
sdir = CE.NLA + '/tags'
OUTF = '%s/ztrain_%s.pkl' % (CE.NLA, tag)
if os.path.exists(OUTF):
    print('exists', OUTF); sys.exit(0)
NM = int(os.environ.get('ZT_NMEM', '10'))
D, info = C.load_npz_info('%s/%s/kz_members_%s.npz' % (sdir, tag, tag))
frames, _ = RC.load_frames('%s/%s/real_ens5flip_%s.pkl' % (sdir, stag, stag), with_gt=False)
K = CE.constants('train')
lum = CE.load_unc('train')['lum']
print('常数（无标签训练集）：颜色原型中心', np.round(K['col_c'], 1), '高度比', np.round(K['ht_col'], 3), flush=True)
for k, f in frames.items():
    p = np.asarray(f['pred'], np.float64).copy()
    lu = lum.get((str(k[0]), str(k[1])))
    if len(p) and lu is not None:
        p[:, 5] = K['ht_col'][int(np.argmin(np.abs(K['col_c'] - lu)))] * p[:, 3:5].mean(1)
    f['pred'] = p
keys = [(s, f) for s, f in zip(D['seq'], D['frame'])]
keys = [k if k in frames else (str(k[0]), str(k[1])) for k in keys]
assert all(k in frames for k in keys)
mem = list(range(NM))
search_ = int(info.get('search', 2)); ks_ = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
out, t0 = {}, time.time()
for i_, k_ in enumerate(keys):
    a_ = C.frame_arms(D, i_, frames[k_], mem, search_, ks_)
    if a_ is None or a_['plan'] is None:
        continue
    xs_ = [a_['ctx'].x(m_, a_['cell']) for m_ in mem]
    rt_ = KFO.pnp(KFO.f32(KFO.satm_vec(xs_, a_['plan'])), a_['ctx'].g, return_rt=True)
    if rt_ is None:
        continue
    zm_ = []
    for x_ in xs_:
        r1_ = KFO.pnp(KFO.f32(x_), a_['ctx'].g, return_rt=True)
        if r1_ is not None and r1_[2][2, 0] > 0:
            zm_.append(np.log(float(r1_[2][2, 0])))
    f = frames[k_]
    p = np.asarray(f['pred'], np.float64).reshape(-1, 9)
    out[(str(k_[0]), str(k_[1]))] = dict(z_kp=float(rt_[2][2, 0]), kp_sd=float(np.std(zm_)) if len(zm_) >= 3 else np.nan,
                                         det=p[int(np.argmax(np.asarray(f['conf']).reshape(-1)))].copy() if len(p) else None)
    if (i_ + 1) % 2000 == 0:
        print('%d/%d  %.0fs' % (i_ + 1, len(keys), time.time() - t0), flush=True)
pickle.dump(out, open(OUTF, 'wb'))
print('完成：%d 帧中 %d 帧有 z_kp，%.0fs' % (len(keys), len(out), time.time() - t0))
