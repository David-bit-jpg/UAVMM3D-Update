# -*- coding: utf-8 -*-
"""[review1113 hparam_1117] collect every finished sensitivity cell into results.json + results.md (reads only existing score JSONs).
Default value of every hyperparameter = the run's reported prediction (gamma 0.5 reproduces it byte-for-byte; gate 10 / mirror 12 / lambda 0.75
are the run itself). Per value: number of runs n, mean of the six test metrics, and the paired difference to the default on the same runs."""
import json, os
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
HP = GP + '/review1113/hparam_1117'
BASE = {'mav_s0': 'srcx/in2mav', 'mav_s10': 'review1113/sup3/run', 'mav_s20': 'review1113/seeds/mav_full_s20',
        'anti_s0': 'srcx/out2antifs', 'anti_s10': 'review1113/seeds/antifs_full_s10', 'anti_s20': 'review1113/seeds/antifs_full_s20'}
SPL = {'mav': 'real_test', 'anti': 'antifs_test'}
MET = ('AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS')
UP = {'AP': 1, 'ATE': -1, 'ASE': -1, 'AOE': -1, 'mDR': 1, 'UDS': 1}


def ld(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p))['all']
    return {k: float(d[k]) for k in MET}


def base(rk, split='test'):
    t = rk.split('_')[0]
    return ld('%s/%s/preds/pcmd_fused_%s.pkl.score_%s.json' % (GP, BASE[rk], split, SPL[t] if split == 'test' else 'val'))


def cell(kind, val, rk, split='test'):
    t = rk.split('_')[0]; sp = SPL[t] if split == 'test' else 'val'; ss = 'test' if split == 'test' else 'val'
    if kind == 'gamma':
        if val == '0.5':
            return base(rk, split)
        tg = {'0': 'g000', '1.0': 'g100'}[val]
        return ld('%s/gamma/%s/preds/%s_fused_%s.pkl.score_%s.json' % (HP, rk, tg, ss, sp))
    if kind in ('gate', 'mir'):
        if (kind, val) in (('gate', '10'), ('mir', '12')):
            return base(rk, split)
        if kind == 'gate' and val == 'none' and rk == 'mav_s0':   # review1113/sup2/nogate (same protocol, reused)
            return ld('%s/review1113/sup2/nogate/preds/nogate_fused_%s.pkl.score_%s.json' % (GP, ss, sp))
        arm = {('gate', '5'): 'g05', ('gate', '20'): 'g20', ('gate', 'none'): 'gall', ('mir', '6'): 'm06', ('mir', '24'): 'm24'}[(kind, val)]
        return ld('%s/stu/%s/%s/preds/hp_fused_%s.pkl.score_%s.json' % (HP, arm, rk, ss, sp))
    if kind == 'rep':
        return ld('%s/stu/rep/%s/preds/hp_fused_%s.pkl.score_%s.json' % (HP, rk, ss, sp))
    if kind == 'lam':
        if val == '0.75':
            return base(rk, split)
        tgt, sh = rk.split('_'); tgt = {'mav': 'mav', 'anti': 'antifs'}[tgt]
        return ld('%s/lam/%s_L%03d_%s/preds/hp_fused_%s.pkl.score_%s.json' % (HP, tgt, int(round(float(val) * 100)), sh, ss, sp))


HPS = [('lam', 'λ（伪框深度去除的偏差比例）', ['0.5', '0.75', '1.0'], '0.75'),
       ('gate', '朝向一致门限（度）', ['5', '10', '20', 'none'], '10'),
       ('mir', 'λ_mir（镜像一致损失权重）', ['6', '12', '24'], '12'),
       ('gamma', 'γ（长宽随深度修正的指数）', ['0', '0.5', '1.0'], '0.5')]
out = {}; md = []
for t, tname in (('mav', 'MAV6D'), ('anti', 'RealDrone3D')):
    runs = ['%s_s%d' % (t, s) for s in (0, 10, 20)]
    md.append('### %s（测试集）\n' % tname)
    md.append('| 超参数 | 取值 | 次数 | AP ↑ | ATE ↓ | ASE ↓ | AOE ↓ | mDR ↑ | UDS ↑ | ΔUDS vs 默认（同批运行） |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    for kind, name, vals, dflt in HPS:
        got = {v: {r: cell(kind, v, r) for r in runs} for v in vals}
        got = {v: {r: x for r, x in d.items() if x is not None} for v, d in got.items()}
        nd = [v for v in vals if v != dflt and got[v]]
        if not nd:
            continue
        used = sorted(set(r for v in nd for r in got[v]))      # default averaged over the runs that have a non-default cell
        for v in vals:
            rs = used if v == dflt else sorted(got[v])
            xs = [got[v][r] for r in rs if r in got[v]]
            if not xs:
                md.append('| %s | %s | 0 | 待跑 | | | | | | |' % (name, v)); continue
            m = {k: sum(x[k] for x in xs) / len(xs) for k in MET}
            dd = ''
            if v != dflt:
                pr = [r for r in rs if r in got[dflt]]
                dd = '%+.2f（%d 次配对）' % (sum(got[v][r]['UDS'] - got[dflt][r]['UDS'] for r in pr) / len(pr), len(pr)) if pr else ''
            lab = ('**%s（默认）**' % v) if v == dflt else v
            if kind == 'gate' and v == 'none':
                lab = '不设门限'
            md.append('| %s | %s | %d | %.2f | %.3f | %.3f | %.2f | %.2f | %.2f | %s |' % (name, lab, len(xs), m['AP'], m['ATE'], m['ASE'], m['AOE'], m['mDR'], m['UDS'], dd))
            out.setdefault(t, {}).setdefault(kind, {})[v] = dict(n=len(xs), runs=[r for r in rs if r in got[v]], mean=m, per_run={r: got[v][r] for r in rs if r in got[v]})
    rp = {r: cell('rep', None, r) for r in runs}; rp = {r: x for r, x in rp.items() if x is not None}
    if rp:
        out.setdefault(t, {})['rep'] = rp
        md.append('\n重训噪声参照（默认取值把两个朝向学生按同协议重训一次，对比报告值）：' + '；'.join(
            '%s UDS %.2f（报告 %.2f）AOE %.2f（报告 %.2f）' % (r, x['UDS'], base(r)['UDS'], x['AOE'], base(r)['AOE']) for r, x in rp.items()))
    md.append('')
json.dump(out, open(HP + '/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(HP + '/results.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))
