# Bold check of tab_main: within each detector block, the best value of each column must carry \best{}.
import re
BS = chr(92)
tex = open('D:/CVPR27_UAV/tables/tab_main.tex', encoding='utf-8').read()
body = tex[tex.index('Method & AP'):tex.index(BS + 'bottomrule')]
blocks = body.split(BS + 'midrule')[1:]
cols = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS'] * 2; blk = ['MAV6D'] * 6 + ['RealDrone3D'] * 6
up = {'AP': 1, 'mDR': 1, 'UDS': 1, 'ATE': -1, 'ASE': -1, 'AOE': -1}
pat = re.compile(re.escape(BS) + r'best\{([^}]*)\}')
n = 0
for b in blocks:
    rows = [l for l in b.strip().splitlines() if '&' in l]
    P = []
    for l in rows:
        c = [x.strip() for x in l.replace(BS + BS, '').split('&')]
        cells = []
        for x in c[1:13]:
            v = pat.sub(r'\1', x)
            cells.append((BS + 'best' in x, None if v == '--' else float(v)))
        P.append((c[0], cells))
    for j in range(12):
        vals = [(r[0], r[1][j][1], r[1][j][0]) for r in P if r[1][j][1] is not None]
        if len(vals) < 2: continue
        best = max(v[1] * up[cols[j]] for v in vals)
        for name, v, isbold in vals:
            should = (v * up[cols[j]] == best); n += 1
            if should != isbold:
                print('BOLD MISMATCH |', name, '|', blk[j], cols[j], v, '| bold' if isbold else '| not bold', '-> should be bold' if should else '-> should not be bold')
print('checked', n, 'cells')
