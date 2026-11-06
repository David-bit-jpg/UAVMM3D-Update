# -*- coding: utf-8 -*-
"""把纯仿真模型在 MAV6D 真实 test 上的逐帧预测存成一个 pkl，给 view_real_preds.py 翻页看（2026-09-22）。

推理与 viz_best.py / eval_camnorm 同一套钩子（集成 → 翻转 → 无标签自标定尺寸 → 在线热图时序背景），
不给尺寸；匹配 greedy（1 m 内算检出）。一次跑完（5 模型 + 翻转约 20~30 分钟），之后看图不需要 GPU。

    python annot_audit/dump_real_preds.py                       # 缺省 = 发布版 5 种子集成 + 翻转
    python annot_audit/dump_real_preds.py --ensemble --no-flip --out ../output/camnorm/viewer/real_preds_s0.pkl   # 单模型
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)
import eval_camnorm as EC   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

REAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
REL = os.path.join(TOOLS, '..', 'output', 'release_20260920')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=REAL, help='真实评测配置（带 kp2d 头的权重用 mav6d_r34_geo_full_tr_kp.yaml）')
    ap.add_argument('--ckpt', default=os.path.join(REL, 'ramp_s0.pth'))
    ap.add_argument('--ensemble', nargs='*', default=[os.path.join(REL, 'ramp_s%d.pth' % i) for i in range(1, 5)],
                    help='其余种子；给空（只写 --ensemble）= 单模型')
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--kp2d-from', default=None, nargs='+', help='旋转来源分离：带 kp2d 头的权重，只贡献 PnP 旋转；位置仍全部来自 --ckpt/--ensemble')
    ap.add_argument('--kp2d-cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr_kp.yaml')
    ap.add_argument('--kp2d-zoom', type=float, default=None, help='放大裁窗版旋转来源：目标 2D 宽 px（KZ 臂用 95）；不给 = 整幅喂')
    ap.add_argument('--kp2d-raw-root', default='@@MAV6D@@')
    ap.add_argument('--kp2d-oracle-crop', action='store_true', help='归因用：裁窗按真值中心/真值 2D 宽（不是部署方式）')
    ap.add_argument('--bg', default='causal100')
    ap.add_argument('--calib-conf', type=float, default=0.7)
    ap.add_argument('--calib-interval', type=int, default=2)
    ap.add_argument('--size-source', default='calib', choices=['calib', 'gt', 'known'],
                    help='calib = 不给尺寸（val 上无标签自标定，缺省）；gt = 给定尺寸，逐帧用该机型真实尺寸；'
                         'known = 给定尺寸，两机型共用配置里的 KNOWN_SIZE（旧做法）')
    ap.add_argument('--out', default=os.path.join(TOOLS, '..', 'output', 'camnorm', 'viewer', 'real_preds_ens5flip.pkl'))
    ap.add_argument('--split', default='test', help='跑哪个划分（test / train / val）；自训练伪标签用 train（不看它的标签）')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()

    cfg = EasyDict()
    cfg_from_yaml_file(a.cfg, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
    if a.split != 'test':
        cfg.DATA_CONFIG.DATA_SPLIT['test'] = a.split          # 数据集按 DATA_SPLIT[mode] 找子目录，评测模式 mode = test
    seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=logger, training=False)

    def load(ck):
        m = build_network(cfg.MODEL, ds)
        n, t = m.load_params_from_file(ck, to_cpu=False)
        assert n == t, ck
        return m.cuda().eval()

    model = load(a.ckpt)
    if a.ensemble:
        EC.install_ensemble_hook(model, [load(c) for c in a.ensemble])
    if a.kp2d_from:
        kcfg = EasyDict()
        cfg_from_yaml_file(a.kp2d_cfg, kcfg)
        kds, _, _ = build_dataloader(kcfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=logger, training=False)
        kms = []
        for ck in a.kp2d_from:
            km = build_network(kcfg.MODEL, kds)
            n, t = km.load_params_from_file(ck, to_cpu=False)
            assert n == t, ck
            kms.append(km.cuda().eval())
        if a.kp2d_zoom is None:
            EC.install_kp2d_source_hook(model, kms)
    if not a.no_flip:
        EC.install_flip_hook(model)
    if a.size_source == 'calib':
        ks, n_cal = EC.self_calibrate_size(cfg, model, 'val', a.calib_interval, a.calib_conf, 8, logger)
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
        model.model_cfg.POST_PROCESSING.KNOWN_SIZE = ks.tolist()
    elif a.size_source == 'gt':
        ks = np.zeros(3)
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'gt'
    else:
        ks = np.asarray(cfg.MODEL.POST_PROCESSING.KNOWN_SIZE, np.float64)
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
    if a.bg != 'none':      # 'none'：相机不固定的数据（AntiUAV3D 跟踪变焦相机）不做时序背景；缺省 causal100 行为不变
        EC.install_background_hook(model, EC.temporal_background(model, loader, a.bg))
    zoom_stats = None
    if a.kp2d_from and a.kp2d_zoom is not None:
        import kp_zoom
        zoom_stats = kp_zoom.install_zoom_kp2d_hook(model, kms, kp_zoom.RawSource(ds, a.kp2d_raw_root), target_w2d=a.kp2d_zoom,
                                                    norm_mean=cfg.DATA_CONFIG.get('NORM_MEAN', None), norm_std=cfg.DATA_CONFIG.get('NORM_STD', None),
                                                    oracle_gt=a.kp2d_oracle_crop)
    recs = pose_eval.run_inference(model, loader)
    if zoom_stats:
        print('放大裁窗：%d 帧，缩放倍数中位 %.2f [5%%, 95%%] = [%.2f, %.2f]' % (
            zoom_stats['n'], np.median(zoom_stats['z']), *np.percentile(zoom_stats['z'], [5, 95])))
        for cname in ('phantom4', 'mavic2'):
            zz = np.array([r[2] for r in zoom_stats['rows'] if r[0].startswith(cname)])
            ww = np.array([r[3] for r in zoom_stats['rows'] if r[0].startswith(cname)])
            if len(zz):
                print('  %-9s 缩放倍数中位 %.2f [%.2f, %.2f]，位置网络预测的 2D 宽中位 %.0f px [%.0f, %.0f]' % (
                    cname, np.median(zz), *np.percentile(zz, [5, 95]), np.median(ww), *np.percentile(ww, [5, 95])))
    pairs = pose_eval.match_records(recs, seq=seq, match='greedy')
    assert len(pairs) == len(recs)
    summ = pose_eval.summarize(pairs)
    print(pose_eval.format_summary('DUMP', summ))

    # 逐帧：真值、全部预测、匹配到的预测下标（与 pose_eval greedy 同规则）、去畸变全分辨率内参
    frames = []
    for i, r in enumerate(recs):
        meta = ds.metas[int(ds.valid_idx[i])]
        assert meta['seq'] == r['seq_id'] and meta['frame'] == r['frame_id'], (i, meta['seq'], r['seq_id'])
        g = r['gt'][0]
        jm = None
        for j in np.argsort(-r['conf']):
            if np.linalg.norm(r['pred'][j, :3] - g[:3]) < 1.0:
                jm = int(j)
                break
        frames.append({'seq': r['seq_id'], 'frame': r['frame_id'], 'cls': meta['cls'], 'cache_idx': int(ds.valid_idx[i]),
                       'gt': g.astype(np.float64), 'pred': r['pred'].astype(np.float64), 'conf': r['conf'].astype(np.float64),
                       'matched': jm, 'K_in': np.asarray(r['K'], np.float64), 'K_raw': np.asarray(meta['K_raw'], np.float64),
                       'raw_wh': tuple(meta['raw_wh'])})
    frames.sort(key=lambda f: (f['seq'], EC.frame_key(f['frame'])))
    hit = np.array([f['matched'] is not None for f in frames])
    size_txt = {'calib': '无标签自标定尺寸（val，conf≥%.1f）' % a.calib_conf, 'gt': '给定尺寸（逐帧该机型真实尺寸）',
                'known': '给定尺寸（两机型共用 KNOWN_SIZE %s）' % np.round(ks, 3).tolist()}[a.size_source]
    info = {'pipeline': '纯仿真训练；%d 模型集成%s + %s + 在线热图时序背景 %s%s；greedy 1 m 匹配'
                        % (1 + len(a.ensemble), '' if a.no_flip else ' + 翻转 TTA', size_txt, a.bg,
                           '；旋转 ← kp2d 网络 %s + PnP%s' % ('+'.join(os.path.basename(c) for c in a.kp2d_from),
                                                            '（放大裁窗 %.0f px）' % a.kp2d_zoom if a.kp2d_zoom else '') if a.kp2d_from else ''),
            'size_source': a.size_source, 'data_path_cfg': str(cfg.DATA_CONFIG.DATA_PATH), 'kp2d_from': a.kp2d_from, 'kp2d_zoom': a.kp2d_zoom, 'kp2d_oracle_crop': a.kp2d_oracle_crop,
            'ckpts': [a.ckpt] + list(a.ensemble), 'calib_lwh': np.asarray(ks, np.float64).tolist(), 'euler_seq': seq,
            'data_path': str(cfg.DATA_CONFIG.DATA_PATH), 'split': a.split, 'n': len(frames), 'det_rate': float(hit.mean())}
    with open(a.out + '.tmp', 'wb') as f:
        pickle.dump({'info': info, 'frames': frames}, f)
    os.replace(a.out + '.tmp', a.out)
    print('写出 %s：%d 帧，检出 %.1f%%，自标定 lwh %s，用时 %.0f s' % (
        os.path.abspath(a.out), len(frames), 100 * hit.mean(), np.round(ks, 3).tolist(), time.time() - t0))


if __name__ == '__main__':
    main()
