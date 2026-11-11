# 【11-10 无标签链副本 chain_eval_m_nl.py（wf23/mirror/chain_eval_m.py）：位置检测器 = 环境变量 NL_ARM（nl100/nl075/nl050）；
#   第二段输出 / 高度读数 / 朝向重裁标签都从 nolabel/<arm>/ 读；置信度不重排（τ = None，PREREG 第 5 条）；α = 0.5 不变；
#   整链组装时不载入封存真值（recs 的 gt 置空），真值只在 fuse_eval_nl.py 最后一次性打分时由 cmpeval 元数据读入。main() 删去。】
# 副本：wf21/orient_uda/strict_chain_eval.py（只改：标签多找 orient_uda/tags，结果写 wf21/final/），用于最终表复算
# -*- coding: utf-8 -*-
"""wf21/strict：最终"完全单帧"整链的组装与评测（只用 CPU；真值只用于打分）。
整链（每一步都只用本帧图像 + 若干在无标签真实训练集上求一次的全局常数）：
  ① 检测：全严格位置检测器 wsfnb（3 集成 + 翻转，推理不开热图时序背景）→ 每帧候选框（位置、尺寸、朝向粗值、置信度）
  ② 第二段距离修正：5 个裁窗测距网络对每个候选框读 δ̂（log 距离修正量），距离 × exp(δ̂ − c)；c = 按颜色原型分组的常数
  ③ 颜色原型：第二段网络自己画的剪影圈出机身 → 机身平均亮度 → 归到 2 个颜色原型（暗/亮，≈ 两种机型）
  ④ 高度：关键点高度比网络（KZ-KPS）逐帧读高度比；最终用该帧颜色原型的高度比常数 × 平均(长, 宽)
  ⑤ 朝向：5 个关键点网络（KZ 成员）在放大裁窗上出 8 角点 → 每个成员 PnP → 180° 商空间对齐 + 截尾平均（SA-TM）→ 每帧取票最多的机头象限（不看相邻帧）
常数来源（FE_CONST）：train = 在无标签真实训练集上统计（颜色原型中心、各原型的 c 与高度比）；val = 固定用无标签 val 图像统计（与检测器尺寸自标定同源）；other = 旧口径（取自另一评测划分，val↔test 互为）。
配置用环境变量：
  FE_UNC   第二段输出：old = finalads/unc_<split>_wsfnb.pkl（部署 V2Rp，训练素材跨帧）；或 unc/ 下的标签（如 st）
  FE_ORI   朝向标签：kzt7q（现部署，训练含跨帧）或 tags/ 下的 recut 标签后缀（如 kzMSst）
  FE_CONST train | other；FE_COLOR new（本链单帧亮度）| old（gproto/color_type_*.pkl，ST2 带时序背景检测）
  FE_KPS   strict（kps/shape_<split>_wsfnb.pkl，单帧 AdaBN）| old（lidar/kpsize/shape_<split>_wsfnb.pkl）
  FE_HT    color（颜色原型常数）| frame（逐帧高度比）
  FE_NAME  这一行的名字；FE_OUT 结果文件后缀
用法：FE_UNC=st FE_ORI=kzMSst FE_CONST=train FE_COLOR=new FE_KPS=strict FE_HT=color FE_NAME=最终 python final_eval.py"""
import os, sys, json, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__))
STRICT = '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/strict'   # 本副本只读 strict 分支的输入（unc / kps / tags），结果写本目录
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/ablation'); sys.path.insert(0, G + '/cmpeval')
import ablation as AB   # noqa: E402,F401  （把 finalads / lidar / kpsize 加进 path）
import paper_table as PT   # noqa: E402
import temporal_filter as T   # noqa: E402
RC, C, KFO = T.RC, T.C, T.KFO
GUARD = float(np.log(1.4))
POSTAG = os.environ['NL_ARM']
NLA = G + '/gproto/nolabel/' + POSTAG
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
CFG = dict(unc=os.environ.get('FE_UNC', 'st'), ori=os.environ.get('FE_ORI', 'kzMSst'), const=os.environ.get('FE_CONST', 'train'),
           color=os.environ.get('FE_COLOR', 'new'), kps=os.environ.get('FE_KPS', 'strict'), ht=os.environ.get('FE_HT', 'color'))
NAME = os.environ.get('FE_NAME', '|'.join(CFG[k] for k in ('unc', 'ori', 'const', 'color', 'kps', 'ht')))
_tp0 = RC.tag_paths


def tag_paths(tag, out_root=RC.OUT_ROOT):
    for base in (NLA + '/tags',):
        j = '%s/%s/kz_members_%s.json' % (base, tag, tag)
        if os.path.exists(j):
            P = dict(_tp0(json.load(open(j, encoding='utf-8'))['src_tag']))
            P.update(dir=os.path.dirname(j), npz=j[:-5] + '.npz', json=j)
            return P
    return _tp0(tag, out_root)


RC.tag_paths = tag_paths


def MEMS():
    """FE_MEMS='0,1,5,6' 显式指定成员下标（11-09 精简实验：成员子集）；缺省 = 前 FE_NMEM 个"""
    s = os.environ.get('FE_MEMS', '')
    return [int(x) for x in s.split(',')] if s else list(range(int(os.environ.get('FE_NMEM', '5'))))


def km2(v, it=50):
    c = np.quantile(v, [0.25, 0.75])
    for _ in range(it):
        a = np.abs(v[:, None] - c[None]).argmin(1); c = np.array([np.median(v[a == j]) for j in range(2)])
    return np.sort(c)


def load_unc(sp):
    return pickle.load(open('%s/unc/unc_%s_%s.pkl' % (NLA, sp, CFG['unc']), 'rb'))


def lum_of(sp, U):
    assert CFG['color'] == 'new'
    return U['lum']


def shape_of(sp):
    assert CFG['kps'] == 'strict'
    return pickle.load(open('%s/kps/shape_%s_%s%s.pkl' % (NLA, sp, POSTAG, os.environ.get('FE_KPSSFX', '_k0')), 'rb'))   # 1 个高度网络（KPS 种子 0）


def sig_of(u):
    return np.sqrt(2 * np.exp(np.asarray(u['lb'], np.float64)) ** 2 + np.asarray(u['sd'], np.float64) ** 2)


def constants(sp):
    """颜色原型中心 col_c、各原型校正常数 cal_col、各原型高度比 ht_col、全局常数 c_glob、σ 缺省值——都来自"常数来源集合"（不用被评测集合）"""
    src = CFG['const'] if CFG['const'] in ('train', 'val') else ('test' if sp == 'val' else 'val')   # train / val（固定用无标签 val 图像，与检测器尺寸自标定同源）/ other（val↔test 互为）
    if CFG['const'] == 'train':
        assert CFG['unc'] != 'old' and CFG['color'] == 'new', '训练集常数需要本链在 train 上的第二段输出与亮度'
    U = load_unc(src)
    lo = lum_of(src, U)
    c = km2(np.array(list(lo.values())))
    ro = shape_of(src)
    grp = {0: [], 1: []}
    for kk, vv in ro.items():
        lu = lo.get((str(kk[0]), str(kk[1])))
        if lu is not None:
            grp[int(np.argmin(np.abs(c - lu)))].append(float(vv))
    gd, ga, sg = {0: [], 1: []}, [], []
    for kk, u in zip(U['keys'], U['unc']):
        d = np.asarray(u['d'], np.float64).reshape(-1)
        if not len(d):
            continue
        ok = [float(x) for x in d if np.isfinite(x) and abs(x) <= GUARD]
        ga += ok
        s = sig_of(u)
        sg += [float(x) for x in s if np.isfinite(x)]
        lu = lo.get((str(kk[0]), str(kk[1])))
        if lu is not None:
            gd[int(np.argmin(np.abs(c - lu)))] += ok
    return dict(src=src, col_c=c, ht_col=[float(np.median(grp[0])), float(np.median(grp[1]))], n_ht=[len(grp[0]), len(grp[1])],
                cal_col=[float(np.median(gd[0])), float(np.median(gd[1]))], n_cal=[len(gd[0]), len(gd[1])],
                c_glob=float(np.median(ga)), sig_med=float(np.median(sg)))


def factors(sp, U, lum, K):
    recs = pickle.load(open('%s/%s_ens/ens_%s.pkl' % (G, POSTAG, sp), 'rb'))['records']
    assert [(r['seq_id'], r['frame_id']) for r in recs] == [tuple(k) for k in U['keys']]
    out, ncol = {}, 0
    for r, u in zip(recs, U['unc']):
        if not len(r['pred']):
            continue
        lu = lum.get((str(r['seq_id']), str(r['frame_id'])))
        if lu is not None:
            cc = K['cal_col'][int(np.argmin(np.abs(K['col_c'] - lu)))]
            ncol += 1
        else:
            cc = K['c_glob']
        d = np.asarray(u['d'], np.float64).reshape(-1)
        ok = np.isfinite(d) & (np.abs(d) <= GUARD)
        f = np.exp(np.where(ok, np.nan_to_num(d) - cc, 0.0))
        s = sig_of(u)
        out[(r['seq_id'], r['frame_id'])] = (r['pred'][:, :3].copy(), f, r['pred'][:, 2] * f, np.where(np.isfinite(s), s, K['sig_med']))
    return out, ncol


def build(sp):
    K = constants(sp)
    U = load_unc(sp)
    lum = lum_of(sp, U)
    rk = shape_of(sp)

    def height(k):
        if os.environ.get('FE_HTR'):                    # 11-09 精简实验：不用高度网络，所有帧同一个高宽比（= 常数尺寸 s̄ 的高 / 长宽均值）
            return float(os.environ['FE_HTR'])
        if CFG['ht'] == 'color':
            lu = lum.get((str(k[0]), str(k[1])))
            if lu is not None:
                return K['ht_col'][int(np.argmin(np.abs(K['col_c'] - lu)))]
        return float(rk[k]) if k in rk else None
    P = RC.tag_paths('%s_%s_%s' % (POSTAG, sp, CFG['ori']))
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)
    for k, f in frames.items():
        p = np.asarray(f['pred'], np.float64).copy()
        h = height(k)
        if len(p) and h is not None:
            p[:, 5] = h * p[:, 3:5].mean(1)
        f['pred'] = p
    keys, B = RC.compute_arms(D, info, frames, members=MEMS())
    ZKP = {}
    if os.environ.get('FE_ZKP') == '1':                 # wf23 方案二：朝向网络 SA-TM 融合角点 + PnP 的平移深度（第二种独立测距）
        mem = MEMS()
        search_ = int(info.get('search', 2)); ks_ = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
        for i_, k_ in enumerate(keys):
            a_ = C.frame_arms(D, i_, frames[k_], mem, search_, ks_)
            if a_ is None or a_['plan'] is None:
                continue
            xs_ = [a_['ctx'].x(m_, a_['cell']) for m_ in mem]
            rt_ = KFO.pnp(KFO.f32(KFO.satm_vec(xs_, a_['plan'])), a_['ctx'].g, return_rt=True)
            if rt_ is not None:
                zm_ = []
                for x_ in xs_:                                # 每个成员单独 PnP 的深度 → 成员分歧（LiDAR 教的角点测距的逐帧可靠程度）
                    r1_ = KFO.pnp(KFO.f32(x_), a_['ctx'].g, return_rt=True)
                    if r1_ is not None and r1_[2][2, 0] > 0:
                        zm_.append(np.log(float(r1_[2][2, 0])))
                ZKP[k_] = (float(rt_[2][2, 0]), float(np.std(zm_)) if len(zm_) >= 3 else np.nan)
    Fr = KFO.build_arm_frames(frames, keys, B['SATM'], B['top'])          # 无标签：不载入封存真值
    roles = [str(m.get('role', '')) for m in info.get('members', [])] if isinstance(info.get('members'), list) else []
    vmem = [m for m in MEMS() if not (m < len(roles) and roles[m] == 'recut-mirror')]   # 机头象限投票只用原图读数（10 成员时 = 前 5 个，与原来相同）
    votes = T.member_votes(D, info, frames, keys, B['SATM'], members=vmem)
    recs = [{'gt': np.zeros((0, 9)), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
             'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
    fac, ncol = factors(sp, U, lum, K)
    assert T.ALPHA == 0.5
    recs = T.F.apply(recs, fac, None, alpha=T.ALPHA)          # PREREG 第 5 条：不做置信度重排（τ = None）；尺寸联动 α = 0.5
    for r, v in zip(recs, votes):
        r['votes'] = v
    recs = T.do(recs, 'hmm_causal_g0_w1')        # γ=0、窗口 1：每帧取 5 成员票数最多的机头象限，不看相邻帧
    for r in recs:
        k = (str(r['seq_id']), str(r['frame_id']))
        h = height(k)
        if len(r['pred']) and h is not None:
            r['pred'] = r['pred'].copy()
            r['pred'][0, 5] = h * r['pred'][0, 3:5].mean()
    for r in recs:
        k = (str(r['seq_id']), str(r['frame_id']))
        zk_ = ZKP.get(k)
        r['z_kp'] = zk_[0] if zk_ else None
        r['kp_sd'] = zk_[1] if zk_ else np.nan
        r['lum'] = lum.get(k)
        r['sig_fin'] = np.nan                                 # 红外教的测距网络的逐帧可靠程度：σ = √(2·自估不确定度² + 5 网分歧²)，取与所选检测位置最近的那个
        fk_ = fac.get(k)
        if fk_ is not None and len(r['pred']):
            j_ = int(np.argmax(r['conf']))
            pc_ = fk_[0] * np.asarray(fk_[1]).reshape(-1, 1)
            jj_ = int(np.argmin(np.linalg.norm(pc_ - np.asarray(r['pred']).reshape(-1, 9)[j_][:3], axis=1)))
            r['sig_fin'] = float(np.asarray(fk_[3]).reshape(-1)[jj_])
    K['n_frames_with_color'] = ncol
    K['col_c'] = K['col_c'].tolist()
    return recs, K


def yawerr(Rp, Rg):
    E = Rg.T @ Rp
    return abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))


def ang_metrics(P, M, sel):
    g = M['gt']; Km = M['K']
    out = {k: np.full(len(g), np.nan) for k in ('hs', 'tilt', 'geo')}
    for i in sel:
        if not (np.isfinite(P['pt'][i]).all() and np.isfinite(P['conf'][i]) and np.isfinite(P['peul'][i]).all()):
            continue
        if PT.iou(PT.box2d(P['pt'][i], P['plwh'][i], P['peul'][i], Km[i]), PT.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], Km[i])) < 0.1:
            continue
        Rp = Rot.from_euler('xyz', P['peul'][i]).as_matrix(); Rg = Rot.from_euler('xyz', g[i, 6:9]).as_matrix()
        out['hs'][i] = min(yawerr(Rp @ RZ[q], Rg) for q in range(4))
        out['tilt'][i] = float(np.degrees(np.arccos(np.clip(Rp[:, 2] @ Rg[:, 2], -1, 1))))
        out['geo'][i] = PT.sym(Rp, Rg)
    return out


