# -*- coding: utf-8 -*-
"""nose_det 公用：带符号偏航差的分布统计（口径与 tools/diag_heading_convention.py 一致，只是多了正负拆分与环形均值）。"""
import numpy as np

OUT_ROOT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/nose_det'
BINS = np.arange(-180, 181, 30)
LABELS = ['%d~%d' % (a, a + 30) for a in range(-180, 180, 30)]


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {'n': 0}
    rad = np.radians(v)
    mid = (np.abs(v) > 60) & (np.abs(v) < 120)
    return {
        'n': int(n),
        'median': float(np.median(v)),
        'circ_mean': float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))),
        'circ_R': float(np.hypot(np.sin(rad).mean(), np.cos(rad).mean())),
        'within30': float((np.abs(v) < 30).mean()),
        'pm60_120': float(mid.mean()),
        'pos60_120': float((mid & (v > 0)).mean()),
        'neg60_120': float((mid & (v < 0)).mean()),
        'over150': float((np.abs(v) > 150).mean()),
        'hist30': (np.histogram(v, bins=BINS)[0] / n * 100).round(1).tolist(),
    }


def table(per, title):
    """per: {机型: [yaw...]} -> 文本表（含 全部 行）。"""
    lines = [title,
             '%-18s %5s  %s   | %s' % ('机型', 'n', '  '.join('%8s' % l for l in LABELS),
                                       '中位 | 环形均值(R) | <30° | ±(60~120)° [+ / -] | >150°')]
    allv = []
    for k in sorted(per):
        v = np.asarray(per[k], dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        allv.append(v)
        s = stats(v)
        lines.append('%-18s %5d  %s   | %+6.1f° | %+6.1f°(%.2f) | %4.0f%% | %4.0f%% [%3.0f%% / %3.0f%%] | %4.0f%%'
                     % (k, s['n'], '  '.join('%8.1f' % x for x in s['hist30']), s['median'], s['circ_mean'],
                        s['circ_R'], 100 * s['within30'], 100 * s['pm60_120'], 100 * s['pos60_120'],
                        100 * s['neg60_120'], 100 * s['over150']))
    if allv:
        v = np.concatenate(allv)
        s = stats(v)
        lines.append('%-18s %5d  %s   | %+6.1f° | %+6.1f°(%.2f) | %4.0f%% | %4.0f%% [%3.0f%% / %3.0f%%] | %4.0f%%'
                     % ('全部', s['n'], '  '.join('%8.1f' % x for x in s['hist30']), s['median'], s['circ_mean'],
                        s['circ_R'], 100 * s['within30'], 100 * s['pm60_120'], 100 * s['pos60_120'],
                        100 * s['neg60_120'], 100 * s['over150']))
    return '\n'.join(lines)
