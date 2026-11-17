# Build INVENTORY.csv: every metric cell of the paper tables -> source -> recomputed value (unified protocol) -> status.
import re, json, csv, numpy as np
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'; R = G + 'review1113/'
W = 'C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs/'
SC = {}
for f in ('scan_baselines.json', 'scan_pst.json'):
    SC.update(json.load(open(f, encoding='utf-8')))
D = json.load(open('derived.json', encoding='utf-8'))
RA = json.load(open('raised.json', encoding='utf-8'))
D['MAV raised to mDR 69.8']=RA['MAV keep 69.8% (n=3350)']; D['MAV raised to mDR 85.0']=RA['MAV keep 85.0% (n=4080)']; D['RD raised to mDR 78.5']=RA['RD keep 79.4% (n=2083)']; D['RD raised to mDR 85.0']=RA['RD keep 87.5% (n=2296)']
K6 = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS']
SCORER = 'D:/UAV3D_baselines_pack/scripts/score.py (11-16, stable sort), top-1 box, sealed TEST meta'

def src_single(p):
    return dict(kind='single', src=p, vals={k: SC[p][k] for k in K6})
def src_mean(name, paths):
    return dict(kind='mean of %d runs' % len(paths), src=' + '.join(paths), vals={k: float(np.mean([SC[p][k] for p in paths])) for k in K6})
def src_derived(name, label=None):
    v = D[name]; return dict(kind='derived', src=label or name, vals={k: v[k] for k in K6 if k in v})

def wb(m, s, tgt, step): return W + '%s_%s_%s_%s/pred_%s_test.pkl' % (m, s, step, tgt, 'real' if tgt == 'mav' else 'outdoor')
def ws(m, s, tgt): return W + '%s_%s_sim_%s/pred_%s_test.pkl' % (m, 'in' if tgt == 'mav' else 'out', s, 'real' if tgt == 'mav' else 'outdoor')
def kd3(m, tgt):
    step = 'kdc' if tgt == 'mav' else 'kd'; tg = 'mav' if tgt == 'mav' else 'anti'; s = 'in' if tgt == 'mav' else 'out'
    return [W + '%s_%s_%s_%s_s%d/pred_%s_test.pkl' % (m, s, step, tg, i, 'real' if tgt == 'mav' else 'outdoor') for i in range(3)]
P0 = G + 'srcx/in2mav/preds/pcmd_fused_test.pkl'; A0 = G + 'srcx/out2antifs/preds/pcmd_fused_test.pkl'
S0 = G + 'srcx/in2mav/preds/srconly_test.pkl'; AS0 = G + 'srcx/out2antifs/preds/srconly_test.pkl'

def strip(c):
    c = c.strip(); c = re.sub(r'\\best\{([^}]*)\}', r'\1', c); c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c); return c.replace('$-$', '-').strip()
def row_cells(tex, label):
    for line in tex.splitlines():
        cells = [strip(c) for c in line.replace('\\\\', '').split('&')]
        if cells and cells[0].startswith(label) and len(cells) >= 3:
            return cells, line
    raise KeyError(label)
def fmt_like(v, s):
    d = len(s.split('.')[1]) if '.' in s else 0
    return ('%.' + str(d) + 'f') % v

rows = []
def add(table, row, colnames, paper_cells, source, intended=None, note=''):
    for cn0, pv in zip(colnames, paper_cells):
        if cn0 is None: continue
        cn = cn0.split(' ')[0]
        if pv in ('--', ''):
            avail = source and cn in source['vals']
            rows.append([table, row, cn0, pv, source['src'] if source else '', source['kind'] if source else '', SCORER if source else '',
                         fmt_like(source['vals'][cn], '0.00') if avail else '', 'not reported (dash)' + ('; value exists' if avail else ''), note]); continue
        if source is None:
            rows.append([table, row, cn0, pv, '', '', '', '', 'cannot recompute (no local prediction)', note]); continue
        rv = fmt_like(source['vals'][cn], pv)
        st = 'same' if rv == pv else 'changed'
        if source['kind'] != 'single' and st == 'same': st = 'same (derived: %s)' % source['kind']
        if intended is not None:
            iv = fmt_like(intended['vals'][cn], pv)
            rows.append([table, row, cn0, pv, intended['src'], intended['kind'] + ' [INTENDED pending edit]', SCORER, iv,
                         ('same' if iv == pv else 'changed (pending edit: single run -> 3-run mean)'), 'single run s0 = %s; %s' % (rv, note)]); continue
        rows.append([table, row, cn0, pv, source['src'], source['kind'], SCORER, rv, st, note])

# ---------- tab_main ----------
tex = open('D:/CVPR27_UAV/tables/tab_main.tex', encoding='utf-8').read()
def B(m, s, step, tg, seed): return W + '%s_%s_%s_%s_s%d/pred_%s_test.pkl' % (m, s, step, tg, seed, 'real' if tg == 'mav' else 'outdoor')
def SIM(m, s, seed, tg): return W + '%s_%s_sim_s%d/pred_%s_test.pkl' % (m, s, seed, 'real' if tg == 'mav' else 'outdoor')
spec = [
 ('MonoDETR~', src_single(SIM('monodetr', 'in', 0, 'mav')), src_single(SIM('monodetr', 'out', 0, 'anti')), None, None),
 ('MonoDETR + ST', src_single(B('monodetr', 'in', 'st', 'mav', 0)), src_single(B('monodetr', 'out', 'st', 'anti', 0)), None, None),
 ('MonoDETR + PST', src_single(B('monodetr', 'in', 'kdc', 'mav', 0)), src_single(B('monodetr', 'out', 'kd', 'anti', 0)), None, src_mean('RD MonoDETR+PST', kd3('monodetr', 'anti'))),
 ('MonoCD~', src_mean('MAV MonoCD sim (s1,s2)', [SIM('monocd', 'in', 1, 'mav'), SIM('monocd', 'in', 2, 'mav')]), src_single(SIM('monocd', 'out', 0, 'anti')), None, None),
 ('MonoCD + ST', src_single(B('monocd', 'in', 'st', 'mav', 1)), src_single(B('monocd', 'out', 'st', 'anti', 0)), None, None),
 ('MonoCD + PST', src_single(B('monocd', 'in', 'kdc', 'mav', 1)), src_single(B('monocd', 'out', 'kd', 'anti', 0)), src_mean('MAV MonoCD+PST', kd3('monocd', 'mav')), src_mean('RD MonoCD+PST', kd3('monocd', 'anti'))),
 ('MonoDGP~', src_single('D:/cmpfull_runs/monodgp_s0/pred_real_test.pkl'), src_single(SIM('monodgp', 'out', 0, 'anti')), None, None),
 ('MonoDGP + ST', src_single(B('monodgp', 'in', 'st', 'mav', 0)), src_single(B('monodgp', 'out', 'st', 'anti', 0)), None, None),
 ('MonoDGP + PST', src_single(B('monodgp', 'in', 'kdc', 'mav', 0)), src_single(B('monodgp', 'out', 'kd', 'anti', 0)), src_mean('MAV MonoDGP+PST', kd3('monodgp', 'mav')), src_mean('RD MonoDGP+PST', kd3('monodgp', 'anti'))),
 ('MonoCoP~', src_single(SIM('monocop', 'in', 0, 'mav')), src_single(SIM('monocop', 'out', 0, 'anti')), None, None),
 ('MonoCoP + ST', src_single(B('monocop', 'in', 'st', 'mav', 0)), src_single(B('monocop', 'out', 'st', 'anti', 0)), None, None),
 ('MonoCoP + PST', src_single(B('monocop', 'in', 'kdc', 'mav', 0)), src_single(B('monocop', 'out', 'kd', 'anti', 0)), None, src_mean('RD MonoCoP+PST', kd3('monocop', 'anti'))),
 ('GUPNet~', src_single('D:/cmpfull_runs/gupnet_s0/pred_real_test.pkl'), src_single(SIM('gupnet', 'out', 0, 'anti')), None, None),
 ('GUPNet + DANN', src_single('D:/cmpfull_runs/gupnet_uda_dann_s0/pred_real_test.pkl'), None, None, None),
 ('GUPNet + ST', src_single(B('gupnet', 'in', 'st', 'mav', 0)), src_single(B('gupnet', 'out', 'st', 'anti', 0)), None, None),
 ('GUPNet + PST', src_mean('MAV GUPNet+PST', kd3('gupnet', 'mav')), src_single(B('gupnet', 'out', 'kd', 'anti', 0)), None, src_mean('RD GUPNet+PST', kd3('gupnet', 'anti'))),
 ('CenterNet~', src_single(S0), src_single(AS0), None, None),
 ('CenterNet + DANN', None, None, None, None),
 ('CenterNet + ST', src_single(R + 'B4/run/preds/naive_det1_test.pkl'), None, None, None),
 ('CenterNet + PST', src_single(P0), src_single(A0), None, None),
 ('Upper bound', None, None, None, None),
]
for lab, smav, srd, imav, ird in spec:
    cells, _ = row_cells(tex, lab); name = lab.rstrip('~')
    add('tab_main', name, [c + ' (MAV6D)' for c in K6], cells[1:7], smav, None, '')
    if imav is not None:
        rows[-6:] = []; add('tab_main', name, [c + ' (MAV6D)' for c in K6], cells[1:7], smav, imav)
    add('tab_main', name, [c + ' (RealDrone3D)' for c in K6], cells[7:13], srd, None, '')
    if ird is not None:
        rows[-6:] = []; add('tab_main', name, [c + ' (RealDrone3D)' for c in K6], cells[7:13], srd, ird)
# fix column keys (names carry block suffix): remap vals
for r in rows:
    pass

# ---------- tab_ablation ----------
tex = open('D:/CVPR27_UAV/tables/tab_ablation.tex', encoding='utf-8').read()
cols = ['AP', 'mDR', 'ATE', 'ASE', 'AOE', 'UDS']
for lab, s in [('Simulation only', src_single(S0)), ('+ Step 1', src_single(R + 'clean2/buildup/b_det1_test.pkl')),
               ('+ Step 2', src_single(R + 'clean2/buildup/c_students_test.pkl')), ('+ Step 3', src_single(G + 'srcx/in2mav/preds/pcmd_nofuse_test.pkl')),
               ('+ Fusion', src_single(P0)), ('PST, no refiner', src_single(R + 'heads/H3/mav_s0/preds/h3_fused_test.pkl')),
               ('PST, mDR of row 1', src_derived('MAV raised to mDR 69.8', 'srcx/in2mav pcmd_fused_test.pkl, top frames by own confidence until mDR=69.8'))]:
    cells, _ = row_cells(tex, lab); add('tab_ablation', lab, cols, cells[2:8], s)

# ---------- tab_source ----------
tex = open('D:/CVPR27_UAV/tables/tab_source.tex', encoding='utf-8').read()
lines = [l for l in tex.splitlines() if ('& PST' in l or 'Sim.\\ only' in l)]
spec = [('outdoor part / PST', None, src_single(A0)), ('outdoor part / Sim only', None, src_single(AS0)),
        ('indoor part / PST', src_single(P0), src_single(G + 'srcx/in2antifs/preds/pcmd_fused_test.pkl')),
        ('indoor part / Sim only', src_single(S0), src_single(G + 'srcx/in2antifs/preds/srconly_test.pkl')),
        ('mixed / PST', None, src_single(G + 'srcx/mix2antifs/preds/pcmd_fused_test.pkl')),
        ('mixed / Sim only', None, src_single(G + 'srcx/mix2antifs/preds/srconly_test.pkl'))]
for l, (lab, sm, sr) in zip(lines, spec):
    cells = [strip(c) for c in l.replace('\\\\', '').split('&')]
    add('tab_source', lab, [c + ' (MAV6D)' for c in K6], cells[2:8], sm); add('tab_source', lab, [c + ' (RealDrone3D)' for c in K6], cells[8:14], sr)

# ---------- tab_orient ----------
tex = open('D:/CVPR27_UAV/tables/tab_orient.tex', encoding='utf-8').read()
for lab, p in [('PST', P0), ('Without the agreement', R + 'sup2/nogate/preds/nogate_fused_test.pkl'), ('Without the mirror', R + 'clean3/nomir/preds/nomir_fused_test.pkl'),
               ('Without sim-on-real', R + 'cleanfix/B4_orient/nocomp/preds/nocomp_fused_test.pkl'), ('Without the minimum', R + 'cleanfix/B4_orient/nosym/preds/nosym_fused_test.pkl'),
               ('Teacher features', R + 'B3/feat/preds/feat_fused_test.pkl'), ('Pseudo labels plus', R + 'B3/both/preds/both_fused_test.pkl'), ('RGB-only teachers', G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl')]:
    cells, _ = row_cells(tex, lab); add('tab_orient', lab, ['AOE', 'UDS'], cells[1:3], src_single(p))

# ---------- tab_sensor ----------
tex = open('D:/CVPR27_UAV/tables/tab_sensor.tex', encoding='utf-8').read()
def uds_src(paths, meanname=None):
    out = []
    for p in paths: out.append(dict(kind='single', src=p, vals={'UDS': SC[p]['UDS']}))
    if meanname: out.append(dict(kind='mean of 3 runs', src=meanname, vals={'UDS': float(np.mean([SC[p]['UDS'] for p in paths]))}))
    return out
SFm = [R + 'sensorfree/E1/mav_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)]
SFa = [R + 'sensorfree/E1/antifs_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)]
PSm = [P0, R + 'sup3/run/preds/pcmd_fused_test.pkl', R + 'seeds/mav_full_s20/preds/pcmd_fused_test.pkl']
PSa = [A0, R + 'seeds/antifs_full_s10/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_full_s20/preds/pcmd_fused_test.pkl']
for lab, mv, rv in [('RGB self-training', uds_src([R + 'sensorfree/E4/mav_s0/preds/e4_fused_test.pkl']), uds_src([R + 'sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl'])),
                    ('Sensor-free variant', uds_src(SFm, 'MAV SF 3 runs'), uds_src(SFa, 'RD SF 3 runs')),
                    ('PST', uds_src(PSm, 'MAV PST 3 runs'), uds_src(PSa, 'RD PST 3 runs'))]:
    cells, _ = row_cells(tex, lab)
    for blk, vals, cc in (('MAV6D', mv, cells[1:5]), ('RealDrone3D', rv, cells[5:9])):
        for i, (cn, pv) in enumerate(zip(['Run 1', 'Run 2', 'Run 3', 'Mean'], cc)):
            s = vals[i] if i < len(vals) else None
            if pv == '--': rows.append(['tab_sensor', lab, cn + ' UDS (' + blk + ')', pv, '', '', '', '', 'not run (dash)', '']); continue
            rvv = fmt_like(s['vals']['UDS'], pv); rows.append(['tab_sensor', lab, cn + ' UDS (' + blk + ')', pv, s['src'], s['kind'], SCORER, rvv, 'same' if rvv == pv else 'changed', ''])

# ---------- supplement tables ----------
sup = open('D:/CVPR27_UAV/sec/X_suppl.tex', encoding='utf-8').read()
nop = sup[sup.index('\\begin{tabular}{l cccccc cccccc}'):sup.index('\\label{tab:nopriv}')]
for lab, sm, sr in [('PST', src_single(P0), src_single(A0)),
                    ('Sensor-free variant', src_single(SFm[0]), src_single(SFa[0])),
                    ('RGB-only teachers', src_single(G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl'), src_single(G + 'srcx/out2antifs_none/preds/pcmd_fused_test.pkl')),
                    ('RGB self-training', src_single(R + 'sensorfree/E4/mav_s0/preds/e4_fused_test.pkl'), src_single(R + 'sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl')),
                    ('Simulation-only detector', src_single(S0), src_single(AS0))]:
    cells, _ = row_cells(nop, lab)
    add('tab:nopriv (suppl)', lab, [c + ' (MAV6D)' for c in K6], cells[1:7], sm); add('tab:nopriv (suppl)', lab, [c + ' (RealDrone3D)' for c in K6], cells[7:13], sr)
mt = sup[sup.index('\\label{tab:steps_ci}'):sup.index('\\label{tab:matched}')]
mt = mt[mt.index('\\begin{tabular}'):]
mav_part = mt[:mt.index('emph{\\antiuav}')]; rd_part = mt[mt.index('emph{\\antiuav}'):]
for part, blk, spec in [(mav_part, 'MAV6D', [('Simulation-only detector', src_single(S0)), ('PST, top 69.8', src_derived('MAV raised to mDR 69.8', 'in2mav pcmd_fused, raised confidence threshold')),
                                            ('PST, top 85.0', src_derived('MAV raised to mDR 85.0', 'in2mav pcmd_fused, raised confidence threshold')), ('PST &', src_single(P0))]),
                        (rd_part, 'RealDrone3D', [('Simulation-only detector', src_single(AS0)), ('PST, top 79.4', src_derived('RD raised to mDR 78.5', 'out2antifs pcmd_fused, raised confidence threshold')),
                                                 ('PST, top 87.5', src_derived('RD raised to mDR 85.0', 'out2antifs pcmd_fused, raised confidence threshold')), ('PST &', src_single(A0))])]:
    for lab, s in spec:
        if lab == 'PST &':
            line = [l for l in part.splitlines() if l.strip().startswith('PST &')][0]; cells = [strip(c) for c in line.replace('\\\\', '').split('&')]; lab = 'PST'
        else:
            cells, _ = row_cells(part, lab)
        add('tab:matched (suppl)', lab + ' [' + blk + ']', K6, cells[1:7], s)

# remap column names with block suffix to metric keys
fixed = []
for r in rows:
    fixed.append(r)
with open('INVENTORY_tables_raw.json', 'w', encoding='utf-8') as f: json.dump(rows, f, ensure_ascii=False)
print(len(rows))
with open('INVENTORY.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['table','row','column','paper_value','source_predictions','source_kind','scorer','recomputed_unified','status','note'])
    w.writerows(rows)
