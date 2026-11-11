# -*- coding: utf-8 -*-
"""wf23/fusetrain/build_targets.py 的无标签链副本（11-10）：第二段第 3 步的合并伪目标。
  δ_新 = δ_伪 + w·(log z_kp − log z_伪 − b_k)，即 log z_合 = (1−w)·log z_伪 + w·(log z_kp − b_k)；
  z_伪 = 位置伪标签 index_pseudo_<arm>（只按置信度筛帧）的距离；z_kp = 训练帧上的角点距离（ztrain_nl.py，部署同法，新检测器窗口）；
  只在检测与伪标签相距 < 0.5 m 的帧上合并；b_k = 颜色原型 k 上 median(log z_kp / z_伪)（无标签训练集）；没有 z_kp 的裁窗保持原伪目标。
  w = nolabel/<arm>/w_sim.json（仿真验证集逆方差，与部署同一个数）。
改动：伪标签换 index_pseudo_<arm>；裁窗行与帧的对应直接读 extract_nbm_nl.py 存的 keys；删去全部真值诊断（不打开 index.pkl）；不再写一致度权重 / 分歧文件（最终版不用）。
输出 selftrain/real_s{j}_nbm_<arm>_dpfz.npy（j = 0、1 → 种子 5、6）+ nolabel/<arm>/build_targets.json。
用法：NL_ARM=<arm> FE_UNC=pp2 python build_targets_nl.py"""
import os, sys, pickle, json
os.environ.setdefault('FE_CONST', 'train')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import chain_eval_m_nl as CE   # noqa: E402
G = CE.G; arm = CE.POSTAG; NLA = CE.NLA
w = float(json.load(open(NLA + '/w_sim.json', encoding='utf-8'))['w'])
ix = pickle.load(open('@@MMCACHE@@/mav6d_cn_obj/train/index_pseudo_%s.pkl' % arm, 'rb'))
assert ix.get('pseudo', {}).get('pipeline', '').startswith('无标签版')
ps = {(str(ix['metas'][i]['seq']), str(ix['metas'][i]['frame'])): np.asarray(ix['metas'][i]['boxes9d'], np.float64)[0] for i in ix['valid_idx']}
Z = pickle.load(open('%s/ztrain_%s_train1_ouS6I2M.pkl' % (NLA, arm), 'rb'))
K = CE.constants('train'); lum = CE.load_unc('train')['lum']
cc = np.asarray(K['col_c'])
colk = lambda k: int(np.argmin(np.abs(cc - lum[k]))) if k in lum else -1
dd = {}
for k, z in Z.items():
    if k in ps and z['det'] is not None and z['z_kp'] > 0 and np.linalg.norm(z['det'][:3] - ps[k][:3]) < 0.5:
        dd[k] = float(np.log(z['z_kp'] / ps[k][2]))
B = {}
for c in (0, 1):
    v = np.array([x for k, x in dd.items() if colk(k) == c])
    B[c] = float(np.median(v))
print('%s：w = %.4f；有 z_kp 且与伪标签同一架的训练帧 %d / 伪标签帧 %d / z_kp 帧 %d；b_k %s' % (arm, w, len(dd), len(ps), len(Z), {c: round(v, 4) for c, v in B.items()}), flush=True)
info = dict(arm=arm, w=w, B=B, n=len(dd), col_c=cc.tolist(), per_j={})
for j in (0, 1):
    pre = '%s/selftrain/real_s%d_nbm_%s_' % (G, j, arm)
    kk = pickle.load(open(pre + 'keys.pkl', 'rb'))
    dp = np.load(pre + 'dp.npy')
    assert len(kk) == len(dp)
    r = np.array([dd[k] - B[colk(k)] if (k in dd and colk(k) in B) else np.nan for k in kk])
    has = np.isfinite(r)
    dn = (dp + w * np.nan_to_num(r)).astype(np.float32)
    np.save(pre + 'dpfz.npy', dn)
    info['per_j'][j] = dict(n=len(kk), n_kp=int(has.sum()), mean_shift=float(np.mean(dn - dp)))
    print('s%d：裁窗 %d，有 z_kp %d（%.0f%%），目标平均变化 %+.4f' % (j, len(kk), int(has.sum()), 100 * has.mean(), float(np.mean(dn - dp))), flush=True)
json.dump(info, open(NLA + '/build_targets.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
