# -*- coding: utf-8 -*-
"""零样本定位误差 4 m 是不是单位没统一？逐帧查数。

对每个匹配上的 (GT, 预测)：
  标注单位自洽  GT 8 角点投影 2D 框的长边像素 s_px；s_px * Z_gt / f_in 应 ≈ GT 物理尺寸（投影长边对应的米数），两域都查
  深度比        Z_pred / Z_gt
  尺寸比        diag(lwh_pred) / diag(lwh_gt)
  网络自洽      s_px * Z_pred / f_in vs 预测尺寸：网络是不是「把目标当成更大的无人机 -> 放得更远」
  虚拟深度      Zv = Z * 512 / f_in（网络真正输出的量）
判读：深度比 ≈ 10/100/1000 -> 单位；≈ 两域焦距比 1.92 -> 焦距处理；≈ 尺寸比 -> 尺度先验（仿真网格偏大），不是单位。

    python tools/annot_audit/diag_units.py
"""
import os
import shutil
import sys

import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(TOOLS))
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
M = '../output/models/uavdet_3d/camnorm'
TMP = 'C:/Users/jiang/AppData/Local/Temp/claude/D--/7e7725c9-419c-44d4-a011-2a87465a0f35/scratchpad'


def infer(cfg_file, interval, ckpt, sets=None):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, (n, t)
    model.cuda().eval()
    return pose_eval.run_inference(model, loader)


def proj_extent(b, K):
    c = (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])), c


def analyze(name, recs, show=8):
    rows = []
    for r in recs:
        K = r['K']
        f = float(np.sqrt(K[0, 0] * K[1, 1]))
        for g in r['gt']:
            if len(r['pred']) == 0:
                continue
            j = int(np.argmin(np.linalg.norm(r['pred'][:, :3] / r['pred'][:, 2:3] - g[:3] / g[2], axis=1)))
            p = r['pred'][j]
            s_px, cg = proj_extent(g, K)
            # 投影长边对应的物理长度：用 GT 8 角点在相机平面（垂直光轴）上的最大跨度
            phys = max(np.ptp(cg[:, 0]), np.ptp(cg[:, 1]))
            uvp = (K @ p[:3])[:2] / p[2]
            uvg = (K @ g[:3])[:2] / g[2]
            rows.append(dict(seq=r['seq_id'], frame=r['frame_id'], f=f, zg=g[2], zp=p[2], sg=np.linalg.norm(g[3:6]), sp=np.linalg.norm(p[3:6]),
                             s_px=s_px, phys=phys, unit_ratio=s_px * g[2] / f / max(phys, 1e-6), uv_err=np.linalg.norm(uvp - uvg),
                             lwh_g=g[3:6], lwh_p=p[3:6], xyz_g=g[:3], xyz_p=p[:3]))
    a = {k: np.array([x[k] for x in rows]) for k in ('f', 'zg', 'zp', 'sg', 'sp', 's_px', 'unit_ratio', 'uv_err')}
    ok = a['uv_err'] < 0.5 * a['s_px']
    print('\n' + '=' * 110)
    print('%s：%d 个匹配（2D 中心误差 < 半个目标的 %d 个，下面比值只在这些上统计）' % (name, len(rows), ok.sum()))
    print('  f_in（输入 512x288 下的焦距）: %s' % np.unique(np.round(a['f'], 1))[:6])
    print('  GT 深度 Z 中位 %.2f m（p10 %.2f / p90 %.2f）；GT 尺寸对角线中位 %.3f m；GT 投影长边中位 %.1f px' % (
        np.median(a['zg']), np.percentile(a['zg'], 10), np.percentile(a['zg'], 90), np.median(a['sg']), np.median(a['s_px'])))
    print('  【标注单位自洽】s_px*Z/f / 物理跨度：中位 %.4f（=1 表示 GT 位置、尺寸、内参单位一致）' % np.median(a['unit_ratio']))
    rz, rs = a['zp'][ok] / a['zg'][ok], a['sp'][ok] / a['sg'][ok]
    print('  深度比 Z_pred/Z_gt 中位 %.3f | 尺寸比 中位 %.3f | 深度比/尺寸比 中位 %.3f | 两者相关系数 %.2f' % (
        np.median(rz), np.median(rs), np.median(rz / rs), np.corrcoef(rz, rs)[0, 1] if len(rz) > 2 else np.nan))
    print('  虚拟深度 Zv=Z*512/f：GT 中位 %.2f，预测中位 %.2f' % (np.median(a['zg'][ok] * 512 / a['f'][ok]), np.median(a['zp'][ok] * 512 / a['f'][ok])))
    print('  逐帧样例（米）：')
    idx = np.where(ok)[0]
    for i in idx[:: max(1, len(idx) // show)][:show]:
        x = rows[i]
        print('    %-28s f=%.0f  GT xyz %s lwh %s | 预测 xyz %s lwh %s | 深度比 %.2f 尺寸比 %.2f 像素长边 %.0f' % (
            (x['seq'] + '/' + x['frame'])[-28:], x['f'], np.round(x['xyz_g'], 2), np.round(x['lwh_g'], 2), np.round(x['xyz_p'], 2),
            np.round(x['lwh_p'], 2), x['zp'] / x['zg'], x['sp'] / x['sg'], x['s_px']))


def main():
    s1 = M + '/sim_indoor8_mz/S1/ckpt/best.pth'
    s3_src = M + '/sim_indoor8_mz_nosescale/S3/ckpt/best.pth'
    s3 = os.path.join(TMP, 'S3_best_snapshot.pth')
    shutil.copyfile(s3_src, s3)
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    sim = 'cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz_nose.yaml'   # 深度/尺寸比与机头朝向无关
    for z in ('1.0', '1.92'):
        analyze('S1 · 仿真 test（裁窗 %s 倍，f_in=%.0f，域内参照）' % (z, 256 * float(z)), infer(sim, 20, s1, ['DATA_CONFIG.VAL_ZOOMS', '[%s]' % z]))
    analyze('S3（机头+尺度修正，训练中快照）· MAV6D 真实 test（零样本）', infer(mav, 10, s3))

if __name__ == '__main__':
    main()
