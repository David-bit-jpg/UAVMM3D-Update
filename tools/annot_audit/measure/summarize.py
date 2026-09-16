# -*- coding: utf-8 -*-
"""汇总 measure_bg_offset.py 的逐帧结果：按域/机型/阈值取中位数、排除率，并最小二乘拟合「视觉中心相对标签中心」的机体系偏移。
  python summarize.py
"""
import os, pickle, json, csv
import numpy as np

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/measure'
MAIN_THR = 30
MAV_GEOM_OFF = [-0.01, 0.01, -0.055]


def med(rows, k):
    v = np.array([r[k] for r in rows if k in r and r[k] is not None], dtype=np.float64)
    return float(np.median(v)) if len(v) else float('nan')


def q(rows, k, p):
    v = np.array([r[k] for r in rows if k in r and r[k] is not None], dtype=np.float64)
    return float(np.percentile(v, p)) if len(v) else float('nan')


def fit_body_offset(rows):
    """dx,dy ≈ J @ o，o 为机体系(米)偏移；返回 o、拟合前后残差 RMS(px)。"""
    A = np.concatenate([np.asarray(r['J'], np.float64) for r in rows], 0)
    b = np.concatenate([[r['dx_px'], r['dy_px']] for r in rows]).astype(np.float64)
    o, *_ = np.linalg.lstsq(A, b, rcond=None)
    rms0 = float(np.sqrt(np.mean(b ** 2)))
    rms1 = float(np.sqrt(np.mean((b - A @ o) ** 2)))
    return o.tolist(), rms0, rms1


def summarize_group(rows_all, label):
    """rows_all: 同一 (domain, cls, thr) 的所有帧。"""
    n = len(rows_all)
    found = [r for r in rows_all if r.get('found')]
    ok = [r for r in rows_all if r.get('found') and not r.get('excluded')]
    s = dict(group=label, n_frames=n, n_seq=len(set(r['seq'] for r in rows_all)),
             n_found=len(found), n_kept=len(ok), excl_rate=round(1 - len(ok) / max(n, 1), 4),
             Z_median=med(ok, 'Z'), body_px_median=med(ok, 'body_px'),
             dx_px_median=med(ok, 'dx_px'), dy_px_median=med(ok, 'dy_px'),
             dx_px_q25=q(ok, 'dx_px', 25), dx_px_q75=q(ok, 'dx_px', 75), dy_px_q25=q(ok, 'dy_px', 25), dy_px_q75=q(ok, 'dy_px', 75),
             dist_px_median=med(ok, 'dist_px'), dx_m_median=med(ok, 'dx_m'), dy_m_median=med(ok, 'dy_m'),
             vis_w_over_lab_w_median=med(ok, 'vis_w_over_lab_w'), vis_h_over_lab_h_median=med(ok, 'vis_h_over_lab_h'),
             vis_w_over_lab_w_q25=q(ok, 'vis_w_over_lab_w', 25), vis_w_over_lab_w_q75=q(ok, 'vis_w_over_lab_w', 75),
             vis_h_over_lab_h_q25=q(ok, 'vis_h_over_lab_h', 25), vis_h_over_lab_h_q75=q(ok, 'vis_h_over_lab_h', 75),
             un_dx_px_median=med(ok, 'un_dx_px'), un_dy_px_median=med(ok, 'un_dy_px'),
             un_w_over_lab_w_median=med(ok, 'un_w_over_lab_w'), un_h_over_lab_h_median=med(ok, 'un_h_over_lab_h'),
             un_nblob_median=med(ok, 'un_nblob'), blob_area_median=med(ok, 'blob_area'))
    if ok and 'dx_geo_px' in ok[0]:
        d0 = np.array([np.hypot(r['dx_px'], r['dy_px']) for r in ok])
        d1 = np.array([np.hypot(r['dx_geo_px'], r['dy_geo_px']) for r in ok])
        s.update(dx_geo_px_median=med(ok, 'dx_geo_px'), dy_geo_px_median=med(ok, 'dy_geo_px'),
                 dist_median_vs_label=float(np.median(d0)), dist_median_vs_geomcenter=float(np.median(d1)),
                 frac_geomcenter_closer=float(np.mean(d1 < d0)),
                 vis_w_over_off_w_median=med(ok, 'vis_w_over_off_w'), vis_h_over_off_h_median=med(ok, 'vis_h_over_off_h'))
    if len(ok) >= 6:
        o, r0, r1 = fit_body_offset(ok)
        s.update(body_offset_fit_m=[round(x, 4) for x in o], fit_rms_px_before=round(r0, 2), fit_rms_px_after=round(r1, 2))
    return s


def main():
    summaries, per_seq = [], []
    for dom in ('sim', 'mav6d'):
        p = os.path.join(OUT, 'frames_%s.pkl' % dom)
        if not os.path.exists(p):
            print('missing', p); continue
        rows = pickle.load(open(p, 'rb'))
        thrs = sorted(set(r['thr'] for r in rows))
        clss = sorted(set(r['cls'] for r in rows))
        for thr in thrs:
            for cls in clss:
                g = [r for r in rows if r['thr'] == thr and r['cls'] == cls]
                s = summarize_group(g, '%s/%s' % (dom, cls)); s.update(domain=dom, cls=cls, thr=thr); summaries.append(s)
            if dom == 'sim':   # 机型合并 _up
                for base in sorted(set(c.replace('_up', '') for c in clss)):
                    g = [r for r in rows if r['thr'] == thr and r['cls'].replace('_up', '') == base]
                    s = summarize_group(g, '%s/%s(+_up)' % (dom, base)); s.update(domain=dom, cls=base + '(+_up)', thr=thr); summaries.append(s)
            g = [r for r in rows if r['thr'] == thr]
            s = summarize_group(g, '%s/ALL' % dom); s.update(domain=dom, cls='ALL', thr=thr); summaries.append(s)
        for seq in sorted(set(r['seq'] for r in rows)):
            g = [r for r in rows if r['thr'] == MAIN_THR and r['seq'] == seq]
            s = summarize_group(g, seq); s.update(domain=dom, cls=g[0]['cls'], thr=MAIN_THR, seq=seq)
            # 无人机在画面里动了多少（投影中心的散布），判断中值背景是否可靠
            u = np.array([r['u_lab'] for r in g]); v = np.array([r['v_lab'] for r in g])
            s.update(center_spread_px=float(np.hypot(u.std(), v.std())), center_range_px=float(max(u.ptp(), v.ptp())))
            per_seq.append(s)

    json.dump(dict(main_thr=MAIN_THR, mav_geom_center_offset_body=MAV_GEOM_OFF, groups=summaries, per_seq=per_seq),
              open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    keys = []
    for s in summaries + per_seq:
        for k in s:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, 'summary_groups.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(summaries)
    with open(os.path.join(OUT, 'summary_per_seq.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(per_seq)

    fmt = '%-42s %4s %5s %5s %6s | %7s %7s %7s %7s | %6s %6s | %6s %6s | %s'
    print(fmt % ('group', 'thr', 'nfrm', 'kept', 'excl', 'dx_px', 'dy_px', 'dx_m', 'dy_m', 'w/lab', 'h/lab', 'un_w', 'un_h', 'geo: dxg dyg fracCloser | fit o(m) rms0->rms1'))
    for s in summaries:
        extra = ''
        if 'dx_geo_px_median' in s:
            extra = '%6.1f %6.1f %5.2f' % (s['dx_geo_px_median'], s['dy_geo_px_median'], s['frac_geomcenter_closer'])
        if 'body_offset_fit_m' in s:
            extra += ' | o=%s %.1f->%.1f' % (s['body_offset_fit_m'], s['fit_rms_px_before'], s['fit_rms_px_after'])
        print(fmt % (s['group'], s['thr'], s['n_frames'], s['n_kept'], '%.2f' % s['excl_rate'],
                     '%.1f' % s['dx_px_median'], '%.1f' % s['dy_px_median'], '%.3f' % s['dx_m_median'], '%.3f' % s['dy_m_median'],
                     '%.2f' % s['vis_w_over_lab_w_median'], '%.2f' % s['vis_h_over_lab_h_median'],
                     '%.2f' % s['un_w_over_lab_w_median'], '%.2f' % s['un_h_over_lab_h_median'], extra))
    print('\nper-seq (thr=%d):' % MAIN_THR)
    for s in per_seq:
        print('%-55s n=%3d kept=%3d excl=%.2f Z=%.2f body_px=%5.1f spread=%6.1f dx=%6.1f dy=%6.1f w/lab=%.2f h/lab=%.2f un_w=%.2f un_h=%.2f' % (
            s['seq'], s['n_frames'], s['n_kept'], s['excl_rate'], s['Z_median'], s['body_px_median'], s['center_spread_px'],
            s['dx_px_median'], s['dy_px_median'], s['vis_w_over_lab_w_median'], s['vis_h_over_lab_h_median'],
            s['un_w_over_lab_w_median'], s['un_h_over_lab_h_median']))


if __name__ == '__main__':
    main()
