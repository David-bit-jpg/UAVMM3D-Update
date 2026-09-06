# -*- coding: utf-8 -*-
"""把 output/bench_json 里的评测结果汇成 Markdown 表（位置中位 / 角度中位 / acc@0.2 m），按臂 x 档位。

    D:/Miniconda3/envs/city/python.exe tools/summarize_bench.py --arms C,B,M0,MT,S1,S1MT,RM0 --fracs p01,p05,p10 \
        --zero ZM0_mix,ZMT_mix,ZS1_mix,ZS1MT_mix --out output/bench_json/summary_night.md
"""
import argparse
import json
import os


def load(d, tag):
    p = os.path.join(d, tag + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def cell(r):
    if not r:
        return '—'
    return '%.3f / %.1f / %.3f' % (r['pos_median'], r['ang_median'], r.get('acc_0.2', float('nan')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', default='output/bench_json')
    ap.add_argument('--arms', required=True)
    ap.add_argument('--fracs', default='p01,p05,p10')
    ap.add_argument('--zero', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arms = a.arms.split(',')
    fracs = a.fracs.split(',')
    lines = ['| 预算 | ' + ' | '.join(arms) + ' |', '|---|' + '---|' * len(arms)]
    for f in fracs:
        rows = [load(a.json_dir, '%s_%s' % (arm, f)) for arm in arms]
        best = None
        vals = [r['pos_median'] for r in rows if r]
        if vals:
            best = min(vals)
        cells = []
        for r in rows:
            c = cell(r)
            if r and best is not None and r['pos_median'] == best:
                c = '**' + c + '**'
            cells.append(c)
        lines.append('| %s | ' % f + ' | '.join(cells) + ' |')
    if a.zero:
        zs = []
        for z in a.zero.split(','):
            r = load(a.json_dir, z)
            zs.append('%s: %s' % (z, ('%.2f m / %.0f° / acc@0.5m %.3f' % (r['pos_median'], r['ang_median'], r.get('acc_0.5', float('nan')))) if r else '—'))
        lines.append('')
        lines.append('零样本（不微调，decode MAX_DIS 40）：' + '；'.join(zs))
    txt = '\n'.join(lines)
    print('（位置中位 m / 角度中位 ° / acc@0.2 m）')
    print(txt)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write('（位置中位 m / 角度中位 ° / acc@0.2 m）\n\n' + txt + '\n')


if __name__ == '__main__':
    main()
