# -*- coding: utf-8 -*-
"""KZ-ST3 评测（prereg_kzst3.json）：5 个学生在 val / test 上推理（位置 ST2P，同发布版命令），
最终管线只换朝向成员，对比现最终版（AdaBN 成员），同一在线 / 离线相邻帧一致。
    python kzst3_eval.py dump <val|test>     → night1003/repl/st2p_<split>_kzt3
    python kzst3_eval.py eval <val|test>     → kzst3/eval_<split>.json + 表"""
import json, os, pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/finalads')
import temporal_filter as T    # noqa: E402
F, RC, C, KFO = T.F, T.RC, T.C, T.KFO
O = G + '/kzst6'
CK = [O + '/train/LOO_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(5)]
ONLINE, OFFLINE = 'hmm_causal_g3_w9', 'hmm_offline_g30_w15'


def dump(split):
    sys.path.insert(0, '@@ROOT@@/output/camnorm/night1003/scratch/repl')
    import dump_members_repl as M
    M.POS_SETS['st2p'] = ['@@ROOT@@/output/models/uavdet_3d/camnorm/st2plus_mav6d_r34_geo_full_tr/ST2P_s%d/ckpt/checkpoint_epoch_3.pth' % i for i in range(5)]
    for c in CK:
        assert os.path.exists(c), c
    return M.main(['--split', split, '--pos', 'st2p', '--ckpts'] + CK + ['--tag', 'st2p_%s_kzt6' % split])


def build_m(split, mtag):
    P = RC.tag_paths('st2p_%s_%s' % (split, mtag))
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)
    keys, B = RC.compute_arms(D, info, frames, members=list(range(5)))
    frames_gt, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
    Fr = KFO.build_arm_frames(frames_gt, keys, B['SATM'], B['top'])
    votes = T.member_votes(D, info, frames, keys, B['SATM'])
    recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
             'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
    recs = F.apply(recs, F.factors(split), T.TAU, alpha=T.ALPHA)
    for r, v in zip(recs, votes):
        r['votes'] = v
    return recs


def desc(recs):
    _, a, _ = T.errs(recs)
    cls = np.array([r['seq_id'].split('/')[0] for r in recs])
    V = np.array([r['votes'] for r in recs])
    return {'ang_med_all': float(np.nanmedian(a)), 'ang_med_mavic2': float(np.nanmedian(a[cls == 'mavic2'])),
            'ang_med_phantom4': float(np.nanmedian(a[cls == 'phantom4'])), 'flip_gt135': float(np.nanmean(a > 135)),
            'lt30': float(np.nanmean(a < 30)), 'members_all_agree': float(np.mean(V.max(1) == 5))}


def evaluate(split):
    base = pickle.load(open(G + '/finalads/final_%s_recs_v.pkl' % split, 'rb'))
    new = build_m(split, 'kzt6')
    pickle.dump(new, open(O + '/kzt6_%s_recs_v.pkl' % split, 'wb'))
    rows, D = {}, {}
    cur = pickle.load(open(G + '/kzst4/kzt4_%s_recs_v.pkl' % split, 'rb'))
    kzla = build_m(split, 'kzma')
    kzab = build_m(split, 'adabn')
    pickle.dump(kzla, open(O + '/kzma_%s_recs_v.pkl' % split, 'wb'))
    for nm, recs in (('KZ + AdaBN（自训练前，对照）', kzab), ('KZ-IRLiD + AdaBN（自训练前）', kzla), ('现最终版（KZ + AdaBN + 高置信自训练）', cur), ('KZ-IRLiD 全链（+ AdaBN + 高置信自训练）', new)):
        for fn, fl in (('逐帧', None), ('在线一致', ONLINE), ('离线一致', OFFLINE)):
            r = recs if fl is None else T.do(recs, fl)
            k = '%s · %s' % (nm, fn)
            rows[k] = F.summ(r, 'kzst6_%s_%d_%s' % (split, len(rows), fn), ('indoor', 'laa'))
            D[k] = desc(r)
            print(k, round(rows[k]['ads_indoor'], 2), round(rows[k]['ads_laa'], 2), round(rows[k]['ang'], 1), json.dumps({x: round(y, 3) for x, y in D[k].items()}), flush=True)
    json.dump(dict(rows=rows, desc=D), open(O + '/eval_%s.json' % split, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L = ['', '## 朝向 IR + LiDAR 辅助答案 KZ-IRLiD 全链（PREREG_irlid B，fb43fbab）· 真实%s' % {'val': '验证', 'test': '测试'}[split], '',
         '| 配置 | 检出 | 位置中位 | <20cm | <50cm | 角度中位 | score | ADS indoor | ADS laa | 机头反向帧 | 角度 mavic2 / phantom4 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, x in rows.items():
        d = D[k]
        L.append('| %s | %.1f%% | %.3f m | %.1f%% | %.1f%% | %.1f° | %.4f | %.2f | %.2f | %.1f%% | %.1f° / %.1f° |' % (
            k, x['det'], x['pos'], x['lt20'], x['lt50'], x['ang'], x['score'], x['ads_indoor'], x['ads_laa'], 100 * d['flip_gt135'], d['ang_med_mavic2'], d['ang_med_phantom4']))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'dump': dump, 'eval': evaluate}[sys.argv[1]](sys.argv[2])
