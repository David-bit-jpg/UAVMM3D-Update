# -*- coding: utf-8 -*-
"""KZ-ST7 评测（prereg_kzst7.json）：两臂各 5 个学生在 val / test 上推理（位置 ST2P，与 KZ-ST6 同一命令），
最终管线只换朝向成员，对比现最终版 KZ-ST6，同一在线 / 离线相邻帧一致。
    python kzst7_eval.py dump <val|test> <q|w>
    python kzst7_eval.py eval <val|test>"""
import json, os, pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/kzst6')
sys.path.insert(0, G + '/finalads')
import kzst6_eval as E6        # noqa: E402
import temporal_filter as T    # noqa: E402
F = T.F
O = G + '/kzst7'
ONLINE, OFFLINE = E6.ONLINE, E6.OFFLINE
NAME = {'q': '臂 Q 前后不教（全部帧前后对称损失）', 'w': '臂 W 不确定度过滤（去掉老师分歧最大 30%）'}


def dump(split, arm):
    sys.path.insert(0, '@@ROOT@@/output/camnorm/night1003/scratch/repl')
    import dump_members_repl as M
    M.POS_SETS['st2p'] = ['@@ROOT@@/output/models/uavdet_3d/camnorm/st2plus_mav6d_r34_geo_full_tr/ST2P_s%d/ckpt/checkpoint_epoch_3.pth' % i for i in range(5)]
    CK = [O + '/train_%s/LOO_s%d/ckpt/checkpoint_epoch_3.pth' % (arm, k) for k in range(5)]
    for c in CK:
        assert os.path.exists(c), c
    return M.main(['--split', split, '--pos', 'st2p', '--ckpts'] + CK + ['--tag', 'st2p_%s_kzt7%s' % (split, arm)])


def flips(recs):
    _, a, _ = T.errs(recs)
    sc = np.array(['/'.join(r['seq_id'].split('/')[:2]) for r in recs])
    return {s: float(np.nanmean(a[sc == s] > 90)) for s in sorted(set(sc))}, float(np.nanmean(a > 90))


def evaluate(split):
    arms = {'cur': pickle.load(open(G + '/kzst6/kzt6_%s_recs_v.pkl' % split, 'rb'))}
    for arm in ('q', 'w'):
        p = O + '/kzt7%s_%s_recs_v.pkl' % (arm, split)
        if not os.path.exists(p):
            pickle.dump(E6.build_m(split, 'kzt7' + arm), open(p, 'wb'))
        arms[arm] = pickle.load(open(p, 'rb'))
    rows, D = {}, {}
    for a_, recs in arms.items():
        nm = '现最终版 KZ-ST6' if a_ == 'cur' else NAME[a_]
        for fn, fl in (('在线一致', ONLINE), ('离线一致', OFFLINE)):
            r = T.do(recs, fl)
            k = '%s · %s' % (nm, fn)
            rows[k] = F.summ(r, 'kzst7_%s_%d_%s' % (split, len(rows), fn), ('indoor', 'laa'))
            fs, fa = flips(r)
            D[k] = dict(E6.desc(r), flip90=fa, flip90_scene=fs, arm=a_, mode=fn)
            print(k, round(rows[k]['ads_indoor'], 2), round(rows[k]['ads_laa'], 2), round(rows[k]['ang'], 1), json.dumps({x: round(y, 3) for x, y in fs.items()}), flush=True)
    json.dump(dict(rows=rows, desc=D), open(O + '/eval_%s.json' % split, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    sc = sorted(D[next(iter(D))]['flip90_scene'])
    L = ['', '## 朝向自训练改法 KZ-ST7（prereg_kzst7.json）· 真实%s' % {'val': '验证', 'test': '测试'}[split], '',
         '| 配置 | 检出 | 位置中位 | 角度中位 | score | ADS indoor | ADS laa | 机头反向(>90°) | %s |' % ' | '.join(sc), '|---|---|---|---|---|---|---|---|' + '---|' * len(sc)]
    for k, x in rows.items():
        d = D[k]
        L.append('| %s | %.1f%% | %.3f m | %.1f° | %.4f | %.2f | %.2f | %.1f%% | %s |' % (
            k, x['det'], x['pos'], x['ang'], x['score'], x['ads_indoor'], x['ads_laa'], 100 * d['flip90'], ' | '.join('%.0f%%' % (100 * d['flip90_scene'][s]) for s in sc)))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'dump':
        dump(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2])
