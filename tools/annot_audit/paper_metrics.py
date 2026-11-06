# -*- coding: utf-8 -*-
"""按论文 Table I 的口径算我们的指标（2026-09-20）。

论文（Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for MAVs, IEEE T-RO 2024, Table I）用三个指标：
    AP          2D 框的平均精度（我们这里报检出率代替，定义不同，只能当参考）
    10°10cm     旋转误差 < 10° 且平移误差 < 10 cm 的比例
    ADD(0.5d)   预测与真值关键点的平均 3D 距离 < 50% 模型直径的比例（这里用 3D 框 8 角点，直径 = 真值框对角）
论文按机型分别训练 / 微调并分别报，所以这里也按机型拆。

    python annot_audit/paper_metrics.py --ckpt <权重> [--ensemble ...] [--flip] [--bg causal100|none]
                                        [--self-calib | --known] [--tag 名字]
"""
import argparse
import os
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ensemble', nargs='*', default=[])
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--bg', default='causal100', help='causal100 / p50 / none')
    ap.add_argument('--self-calib', action='store_true', help='不给尺寸：val 上无标签自标定')
    ap.add_argument('--calib-conf', type=float, default=0.7)
    ap.add_argument('--calib-interval', type=int, default=2)
    ap.add_argument('--tag', default='本方法')
    ap.add_argument('--kp2d-from', default=None, nargs='+', help='旋转来源分离：带 kp2d 头的权重（如 K30），只贡献 PnP 旋转')
    ap.add_argument('--kp2d-cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml')
    ap.add_argument('--kp2d-zoom', type=float, default=None, help='放大裁窗版旋转来源（KZ 臂用 95）；不给 = 整幅喂')
    ap.add_argument('--kp2d-raw-root', default='@@MAV6D@@')
    a = ap.parse_args()

    cfg = EasyDict()
    cfg_from_yaml_file(REAL, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred' if a.self_calib else 'known',
                   'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
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
        # 旋转来源分离（2026-09-22）：kp2d 网络只贡献 PnP 旋转，位置仍全部来自 --ckpt / --ensemble
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
        print('旋转 ← kp2d 网络 %s + PnP（位置不变）%s' % (a.kp2d_from, '，放大裁窗 %.0f px' % a.kp2d_zoom if a.kp2d_zoom else ''))
    if a.flip:
        EC.install_flip_hook(model)
    if a.self_calib:
        ks, n_cal = EC.self_calibrate_size(cfg, model, 'val', a.calib_interval, a.calib_conf, 8, logger)
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
        model.model_cfg.POST_PROCESSING.KNOWN_SIZE = ks.tolist()
        print('无标签自标定 lwh %s（%d 个检测）' % (np.round(ks, 4).tolist(), n_cal))
    if a.bg != 'none':
        EC.install_background_hook(model, EC.temporal_background(model, loader, a.bg))
    if a.kp2d_from and a.kp2d_zoom is not None:
        import kp_zoom
        kp_zoom.install_zoom_kp2d_hook(model, kms, kp_zoom.RawSource(ds, a.kp2d_raw_root), target_w2d=a.kp2d_zoom,
                                       norm_mean=cfg.DATA_CONFIG.get('NORM_MEAN', None), norm_std=cfg.DATA_CONFIG.get('NORM_STD', None))

    recs = pose_eval.run_inference(model, loader)
    pairs = pose_eval.match_records(recs, seq=seq, match='greedy')
    assert len(pairs) == len(recs)
    cls = np.array([r['seq_id'].replace(chr(92), '/').split('/')[0] for r in recs])
    hit = np.array([p['matched'] for p in pairs])
    pos = np.array([p['pos'] if p['matched'] else np.inf for p in pairs])
    rot = np.array([p['ang'] if p['matched'] else np.inf for p in pairs])
    rotf = np.array([p['ang_fold'] if p['matched'] else np.inf for p in pairs])
    add = np.array([p['add'] if p['matched'] else np.inf for p in pairs])
    diam = np.array([p['diam'] if p['matched'] else np.nan for p in pairs])
    d_med = float(np.nanmedian(diam))

    print('%s｜背景 %s｜尺寸 %s｜模型数 %d%s' % (
        a.tag, a.bg, '无标签自标定' if a.self_calib else '官方真值框', 1 + len(a.ensemble), '，翻转 TTA' if a.flip else ''))
    print('真值框对角（模型直径）%.3f m，ADD(0.5d) 阈值 %.3f m，ADD(0.1d) 阈值 %.3f m' % (d_med, 0.5 * d_med, 0.1 * d_med))
    print('%-10s %6s %8s %9s %10s %11s %10s' % ('机型', '帧数', '检出率', '10°10cm', 'ADD(0.5d)', 'ADD(0.5d)|检出', 'ADD(0.1d)'))
    for name, sel in (('phantom4', cls == 'phantom4'), ('mavic2', cls == 'mavic2'), ('全部', np.ones(len(recs), bool))):
        n = int(sel.sum())
        nd = int(hit[sel].sum())
        j = ((pos[sel] < 0.1) & (rot[sel] < 10.0)).sum()
        jf = ((pos[sel] < 0.1) & (rotf[sel] < 10.0)).sum()
        a5 = (add[sel] < 0.5 * d_med).sum()
        a1 = (add[sel] < 0.1 * d_med).sum()
        print('%-10s %6d %7.2f%% %8.2f%% %9.2f%% %10.2f%% %9.2f%%' % (
            name, n, 100.0 * nd / n, 100.0 * j / n, 100.0 * a5 / n, 100.0 * a5 / max(nd, 1), 100.0 * a1 / n))
        print('%-10s %6s %7s %8.2f%% (折 180°)' % ('', '', '', 100.0 * jf / n))
    print('分母 = 全部真值帧（漏检算失败）；「ADD(0.5d)|检出」那一列分母只算检出的帧，用于和论文「先微调再测」的口径对齐参考。')


if __name__ == '__main__':
    main()
