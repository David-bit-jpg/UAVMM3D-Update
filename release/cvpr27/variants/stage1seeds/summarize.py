# -*- coding: utf-8 -*-
"""[review1113 stage1seeds] pooled-by-run summary: PST vs sensor-free E1 vs pure RGB self-training E4, 3 runs per group and target
(run 0 = the reported run with the original stage-1 networks; runs sd30 / sd40 = full reseed incl. stage 1). Run k of the three groups shares
the same stage-1 detectors (and, E1/E4, the same RGB nets) and the same downstream seed shift -> paired by run.
usage: python summarize.py [--boot] [--partial] [--force] [--dry]
  default : reads ONLY the existing score files (<pred>.score_<split>.json written by score_x.py once per run); prints / writes per-run metrics,
            per-run differences vs PST (and E4 - E1), mean +- sd (ddof 1) over the available runs -> summary/SUMMARY.md + summary/summary.json
  --boot  : also runs the existing heads/heads_boot.py (paired sequence / block bootstrap, pooled over the run pairs) for E1-PST, E4-PST, E4-E1;
            only when all 3 runs of both groups exist (--partial: with >= 2 pairs). Output summary/boot_<tgt>_<a>_vs_<b>.json, skipped if present
            (--force recomputes). heads_boot reads TEST labels (existing scorer); nothing else here reads labels.
  --dry   : list the expected prediction / score files and whether they exist; no bootstrap."""
import os, sys, json, subprocess
import numpy as np
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; R = GP + '/review1113'; HERE = R + '/stage1seeds'; OUT = HERE + '/summary'
PY = 'D:/Miniconda3/envs/city/python.exe'
SPLIT = {'mav': 'real_test', 'antifs': 'antifs_test'}
NAME = {'mav': 'MAV6D', 'antifs': 'OutdoorUAV3D'}
GN = {'full': 'PST', 'E1': 'E1 无传感器对照', 'E4': 'E4 纯 RGB 自训练'}
RUNS = ['rep', 'sd30', 'sd40']
KEYS = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')


def pred(tgt, g, k):
    if k == 'rep':
        return {'full': GP + '/srcx/%s/preds/pcmd_fused_test.pkl' % ('in2mav' if tgt == 'mav' else 'out2antifs'),
                'E1': R + '/sensorfree/E1/%s_s0/preds/e1_fused_test.pkl' % tgt,
                'E4': R + '/sensorfree/E4/%s_s0/preds/e4_fused_test.pkl' % tgt}[g]
    return HERE + '/runs/%s_%s_s1%s/preds/pcmd_fused_test.pkl' % (tgt, g, k)


def srconly(tgt, k):
    return (GP + '/srcx/%s/preds/srconly_test.pkl' % ('in2mav' if tgt == 'mav' else 'out2antifs')) if k == 'rep' else HERE + '/runs/%s_full_s1%s/preds/srconly_test.pkl' % (tgt, k)


def done(tgt, g, k):
    return True if k == 'rep' else os.path.exists(HERE + '/runs/%s_%s_s1%s/.done' % (tgt, g, k))


def metrics(p, tgt):
    f = p + '.score_%s.json' % SPLIT[tgt]
    return json.load(open(f))['all'] if os.path.exists(f) else None


def ms(v, f='%.2f'):
    v = [x for x in v if x is not None]
    if not v:
        return '—'
    return ((f + ' ± ' + f + ' (n=%d)') % (np.mean(v), np.std(v, ddof=1), len(v))) if len(v) > 1 else (f + ' (n=1)') % v[0]


def main():
    a = sys.argv[1:]; dry = '--dry' in a
    os.makedirs(OUT, exist_ok=True)
    md = ['# 第一阶段也换种子：按轮配对汇总（只读已有打分文件）', '']
    js = {}
    for tgt in ('mav', 'antifs'):
        md += ['## %s（测试集）' % NAME[tgt], '']
        if dry:
            for g in ('full', 'E1', 'E4'):
                for k in RUNS:
                    p = pred(tgt, g, k); print('  %-6s %-4s %-5s pred %-5s score %-5s %s' % (tgt, g, k, os.path.exists(p), os.path.exists(p + '.score_%s.json' % SPLIT[tgt]), p))
            continue
        T = {g: {k: (metrics(pred(tgt, g, k), tgt) if done(tgt, g, k) else None) for k in RUNS} for g in ('full', 'E1', 'E4')}
        S = {k: metrics(srconly(tgt, k), tgt) for k in RUNS}
        js[tgt] = dict(metrics=T, srconly=S)
        md += ['| 轮 | 组 | ' + ' | '.join(KEYS) + ' |', '|' + '---|' * (len(KEYS) + 2)]
        for k in RUNS:
            for g in ('full', 'E1', 'E4'):
                m = T[g][k]
                md.append('| %s | %s | %s |' % (k, GN[g], ' | '.join(('%.3f' % m[x] if x == 'ATE' else '%.2f' % m[x]) if m else '—' for x in KEYS)))
            m = S[k]; md.append('| %s | 仅仿真检测器（第一阶段） | %s |' % (k, ' | '.join(('%.3f' % m[x] if x == 'ATE' else '%.2f' % m[x]) if m else '—' for x in KEYS)))
        md += ['', '每轮差值（同一轮内配对）与跨轮 均值 ± 标准差：', '', '| 量 | ' + ' | '.join(KEYS) + ' |', '|' + '---|' * (len(KEYS) + 1)]
        for g in ('full', 'E1', 'E4'):
            md.append('| %s 均值 | %s |' % (GN[g], ' | '.join(ms([T[g][k][x] if T[g][k] else None for k in RUNS], '%.3f' if x == 'ATE' else '%.2f') for x in KEYS)))
        for g, h in (('E1', 'full'), ('E4', 'full'), ('E4', 'E1')):
            d = {x: [(T[g][k][x] - T[h][k][x]) if (T[g][k] and T[h][k]) else None for k in RUNS] for x in KEYS}
            js[tgt]['diff_%s_minus_%s' % (g, h)] = d
            md.append('| %s − %s | %s |' % (GN[g], GN[h], ' | '.join(ms(d[x], '%+.3f' if x == 'ATE' else '%+.2f') for x in KEYS)))
        md.append('')
        if '--boot' in a:
            for g, h in (('E1', 'full'), ('E4', 'full'), ('E4', 'E1')):
                pairs = [k for k in RUNS if T[g][k] and T[h][k]]
                if len(pairs) < 3 and not ('--partial' in a and len(pairs) >= 2):
                    md.append('- 自助法 %s vs %s：只有 %d 对，未算' % (g, h, len(pairs))); continue
                oj = '%s/boot_%s_%s_vs_%s.json' % (OUT, tgt, g, h)
                if os.path.exists(oj) and '--force' not in a and json.load(open(oj)).get('_pairs') == pairs:
                    pass
                else:
                    args = [PY, R + '/heads/heads_boot.py', tgt, oj, '%s %s vs %s pooled over runs %s' % (tgt, g, h, ','.join(pairs))]
                    for k in pairs:
                        args += [pred(tgt, g, k), pred(tgt, h, k)]
                    r = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8='1', CUDA_VISIBLE_DEVICES=''))
                    open(oj[:-5] + '.log', 'w', encoding='utf-8').write(r.stdout + r.stderr)
                    if r.returncode:
                        md.append('- 自助法 %s vs %s 失败，见 %s' % (g, h, oj[:-5] + '.log')); continue
                    z = json.load(open(oj)); z['_pairs'] = pairs; json.dump(z, open(oj, 'w'), indent=1)
                last = [l for l in open(oj[:-5] + '.log', encoding='utf-8').read().split('\n') if l.strip()]
                md.append('- 合并自助法（%s − %s，%d 对：%s）：%s' % (GN[g], GN[h], len(pairs), ','.join(pairs), last[-1] if last else oj))
            md.append('')
    if dry:
        return
    open(OUT + '/SUMMARY.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    json.dump(js, open(OUT + '/summary.json', 'w'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
