# -*- coding: utf-8 -*-
"""类别无关的 6-DoF 位姿评测核心，训练时的验证集选轮和最终测试共用这一份，保证口径一致。

口径（审查 P19 之后的约定）：
- 所有准确率 / 选轮分数的分母都是【全部 GT 目标】，没检出 = 失败（误差按上限计）。
- 角度误差 = 测地距离（度），不做 180° 折算；折算版单独给一列 rotfold。
- 匹配：
    top1   每帧恰好 1 个 GT（MAV6D）：取该帧置信度最高的检测
    greedy 多目标（仿真）：检测按置信度降序，各自配最近的未匹配 GT，3D 距离 < dist_thresh 才算检出
- 选轮分数 score = 0.5 * mean(1 - min(pos, P)/P) + 0.5 * mean(1 - min(ang, A)/A)，P=0.5 m、A=30°，
  与 LAA3D_ADS 里 posi/orin accuracy 同形（ADS 的 indoor 口径 P=1 m），只在验证集上用。
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_inference(model, loader):
    """-> list[dict(gt (G,9), pred (P,9), conf (P,), K (3,3), seq_id, frame_id)]，model 需已在 GPU。"""
    from uavdet3d.model import load_data_to_gpu
    was_training = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            out = model(batch)
            for b in range(out['batch_size']):
                gt = _np(batch['gt_box9d'][b]).astype(np.float64).reshape(-1, 9)
                gt = gt[np.abs(gt).sum(1) > 0]                  # 去掉 collate 补的零行
                pred = out['pred_boxes9d'][b]
                conf = out['confidence'][b]
                pred = _np(pred).astype(np.float64) if pred is not None else np.zeros((0, 10))
                pred = pred.reshape(len(pred), -1)[:, :9] if len(pred) else np.zeros((0, 9))
                conf = _np(conf).astype(np.float64).reshape(-1) if conf is not None and len(conf) else np.zeros(0)
                records.append({'gt': gt, 'pred': pred, 'conf': conf,
                                'K': _np(batch['intrinsic'][b]).astype(np.float64).reshape(-1, 3, 3)[0],
                                'seq_id': str(batch['seq_id'][b]), 'frame_id': str(batch['frame_id'][b])})
    if was_training:
        model.train()
    return records


def _pair_errors(p, g, K, seq):
    rp = R.from_euler(seq, p[6:9])
    rg = R.from_euler(seq, g[6:9])
    ang = float(np.degrees((rp.inv() * rg).magnitude()))
    mp = PROTO8 * g[3:6]
    cp = mp @ rp.as_matrix().T + p[:3]
    cg_ = mp @ rg.as_matrix().T + g[:3]
    add = float(np.linalg.norm(cp - cg_, axis=1).mean())
    uv_p = K @ p[:3]
    uv_g = K @ g[:3]
    uv = float(np.linalg.norm(uv_p[:2] / uv_p[2] - uv_g[:2] / uv_g[2])) if uv_p[2] > 1e-6 and uv_g[2] > 1e-6 else np.nan
    return {'pos': float(np.linalg.norm(p[:3] - g[:3])), 'z': float(abs(p[2] - g[2])), 'ang': ang,
            'ang_fold': min(ang, 180.0 - ang), 'add': add, 'uv': uv,
            'diam': float(np.linalg.norm(g[3:6])), 'gt_z': float(g[2])}


def match_records(records, seq='xyz', match='top1', dist_thresh=1.0):
    """-> (pairs: list[dict]，每个 GT 一条，未检出的 matched=False), n_frames"""
    pairs = []
    for r in records:
        gt, pred, conf, K = r['gt'], r['pred'], r['conf'], r['K']
        if len(gt) == 0:
            continue
        if match == 'top1':
            g = gt[0]
            if len(pred):
                e = _pair_errors(pred[int(np.argmax(conf))], g, K, seq)
                e['matched'] = True
            else:
                e = {'matched': False, 'gt_z': float(g[2]), 'diam': float(np.linalg.norm(g[3:6]))}
            pairs.append(e)
            continue
        used = np.zeros(len(gt), bool)
        order = np.argsort(-conf) if len(pred) else []
        got = {}
        for j in order:
            d = np.linalg.norm(gt[:, :3] - pred[j, :3], axis=1)
            d[used] = np.inf
            k = int(np.argmin(d))
            if d[k] < dist_thresh:
                used[k] = True
                got[k] = j
        for k in range(len(gt)):
            if k in got:
                e = _pair_errors(pred[got[k]], gt[k], K, seq)
                e['matched'] = True
            else:
                e = {'matched': False, 'gt_z': float(gt[k, 2]), 'diam': float(np.linalg.norm(gt[k, 3:6]))}
            pairs.append(e)
    return pairs


def summarize(pairs, n_frames=None, pos_cap=0.5, ang_cap=30.0):
    n = len(pairs)
    m = [p for p in pairs if p['matched']]
    out = {'n_gt': n, 'n_matched': len(m), 'det_rate': len(m) / max(n, 1)}
    if n_frames is not None:
        out['n_frames'] = int(n_frames)
    if not m:
        out.update({'score': 0.0})
        return out
    arr = {k: np.array([p[k] for p in m], dtype=np.float64) for k in ('pos', 'z', 'ang', 'ang_fold', 'add', 'uv')}
    for k in ('pos', 'z', 'ang', 'ang_fold', 'add'):
        out[k + '_median'] = float(np.median(arr[k]))
        out[k + '_mean'] = float(arr[k].mean())
        out[k + '_p90'] = float(np.percentile(arr[k], 90))
    u = arr['uv'][np.isfinite(arr['uv'])]
    if len(u):
        out['uv_median'] = float(np.median(u))
    N = float(max(n, 1))
    for t in (0.05, 0.1, 0.2, 0.5):
        out['ACC_pos_%g' % t] = float((arr['pos'] < t).sum() / N)
    for t in (5, 10, 20, 30):
        out['ACC_rot_%d' % t] = float((arr['ang'] < t).sum() / N)
        out['ACC_rotfold_%d' % t] = float((arr['ang_fold'] < t).sum() / N)
    for pt, at in ((0.05, 5), (0.1, 10), (0.2, 20)):
        out['ACC_%gm%ddeg' % (pt, at)] = float(((arr['pos'] < pt) & (arr['ang'] < at)).sum() / N)
    diam = np.array([p['diam'] for p in m])
    out['ACC_add_10'] = float((arr['add'] < 0.1 * diam).sum() / N)
    out['ACC_add_20'] = float((arr['add'] < 0.2 * diam).sum() / N)
    # 选轮分数：未检出的 GT 贡献 0
    pos_acc = (1.0 - np.minimum(arr['pos'], pos_cap) / pos_cap).sum() / N
    ang_acc = (1.0 - np.minimum(arr['ang'], ang_cap) / ang_cap).sum() / N
    out['pos_acc'] = float(pos_acc)
    out['ang_acc'] = float(ang_acc)
    out['score'] = float(0.5 * pos_acc + 0.5 * ang_acc)
    return out


def evaluate(model, loader, seq='xyz', match='top1', dist_thresh=1.0):
    recs = run_inference(model, loader)
    pairs = match_records(recs, seq=seq, match=match, dist_thresh=dist_thresh)
    return summarize(pairs, n_frames=len(recs)), recs


def format_summary(tag, s):
    if 'pos_median' not in s:
        return '%s: 无有效检测（GT %d）' % (tag, s.get('n_gt', 0))
    return ('%s: GT %d 检出 %.1f%% | 位置 中位 %.3f m 均值 %.3f p90 %.3f | 深度 中位 %.3f | 角度 中位 %.1f° (折算 %.1f°) | '
            'pos<0.1 %.1f%% rot<10 %.1f%% 双<0.1m&10° %.1f%% | score %.4f'
            % (tag, s['n_gt'], 100 * s['det_rate'], s['pos_median'], s['pos_mean'], s['pos_p90'], s['z_median'],
               s['ang_median'], s['ang_fold_median'], 100 * s['ACC_pos_0.1'], 100 * s['ACC_rot_10'],
               100 * s['ACC_0.1m10deg'], s['score']))
