# -*- coding: utf-8 -*-
"""源域 (UAV-MM3D / LAAM6D) 与目标域 (MAV6D) 的对齐审计。

回答三个问题，全部用真实样本和真实模型给数字，不靠推断：

    1. 模态对齐   —— 两边 image 张量各是什么形状、K 是几、每个通道是什么
    2. 内外参对齐 —— 内参/畸变/外参、编码器用的是哪套缩放约定、
                     同一个目标在两边分别占多少像素、深度归一化差多少
    3. 坐标系对齐 —— GT 在世界系还是相机系、检测头到底学的是什么、
                     解码器输出到哪个系、欧拉角约定差异有多大

最后直接建两个模型逐张量比对，给出"到底有多少权重能真正迁过去"。

用法：
    python tools/align_audit.py \
        --src-cfg cfgs/models/uavdet_3d/laam6d/centerdet_rgb_resnet8x.yaml \
        --dst-cfg cfgs/models/uavdet_3d/mav6d/centerdet.yaml \
        --dst-data E:/dataset/MAV6D_fake

    # 只看配置和权重对齐，不加载样本（快）
    python tools/align_audit.py ... --no-samples

    # 有源域 ckpt 时，实测能载入多少
    python tools/align_audit.py ... --ckpt output/.../checkpoint_epoch_1.pth
"""
import argparse
import sys

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network
from uavdet3d.utils import common_utils

LINE = '=' * 78
SUB = '-' * 78


def load_cfg(path, data_path=None):
    c = EasyDict()
    cfg_from_yaml_file(path, c)
    if data_path:
        c.DATA_CONFIG.DATA_PATH = data_path
    return c


class _StubDataset:
    """只为建模型用：DetectorTemplate 只碰 dataset_cfg 和 im_num。"""

    def __init__(self, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        self.im_num = dataset_cfg.get('IM_NUM', 1)


def fmt_K(K):
    return np.array2string(np.asarray(K), precision=2, suppress_small=True)


def fov_deg(f, size_px):
    return 2 * np.degrees(np.arctan(size_px / (2.0 * f)))


# --------------------------------------------------------------------------- #
# 1. 模态
# --------------------------------------------------------------------------- #
MODAL_MEANING = {
    'rgb': 'RGB 图 (3ch)',
    'ir': '红外图 (单通道复制成 3ch)',
    'dvs': '事件相机图 (3ch)',
    'lidar': 'LiDAR 投影得到的世界坐标图 (x,y,z 当 3ch)',
    'radar': '毫米波雷达速度热图 (单通道复制成 3ch)',
}


def audit_modality(src_cfg, dst_cfg, src_sample, dst_sample):
    print('\n' + LINE)
    print('1. 模态对齐')
    print(LINE)

    src_modal = list(src_cfg.DATA_CONFIG.get('MODALITIES', ['rgb', 'ir', 'dvs', 'lidar', 'radar']))
    print('源域 MODALITIES : %s' % src_modal)
    for m in src_modal:
        print('    - %-6s %s' % (m, MODAL_MEANING.get(m, '?')))
    print('目标域           : MAV6D 只有单目 RGB，K 恒为 1')

    if src_sample is not None:
        print('\n源域   image 张量: %s   (K=%d)' % (src_sample['image'].shape, src_sample['image'].shape[0]))
    if dst_sample is not None:
        print('目标域 image 张量: %s   (K=%d)' % (dst_sample['image'].shape, dst_sample['image'].shape[0]))

    src_bb = src_cfg.MODEL.BACKBONE_2D
    dst_bb = dst_cfg.MODEL.BACKBONE_2D
    print('\n骨干网络:')
    print('  源域   %-18s NUM_FILTERS=%s  IN=%s OUT=%s' %
          (src_bb.NAME, src_bb.NUM_FILTERS, src_bb.INPUT_CHANNELS, src_bb.OUT_CHANNELS))
    print('  目标域 %-18s NUM_FILTERS=%s  IN=%s OUT=%s' %
          (dst_bb.NAME, dst_bb.NUM_FILTERS, dst_bb.INPUT_CHANNELS, dst_bb.OUT_CHANNELS))

    ok = (src_bb.NAME == dst_bb.NAME and
          list(src_bb.NUM_FILTERS) == list(dst_bb.NUM_FILTERS) and
          src_bb.INPUT_CHANNELS == dst_bb.INPUT_CHANNELS and
          src_bb.OUT_CHANNELS == dst_bb.OUT_CHANNELS)
    print('  => 骨干%s' % ('完全一致，权重可整体迁移' if ok else
                          '不一致：第一层通道/每层宽度对不上，strict=False 会把它们静默跳过'))
    if src_sample is not None and dst_sample is not None:
        k_ok = src_sample['image'].shape[0] == dst_sample['image'].shape[0]
        print('  => 模态数 K %s' % ('一致' if k_ok else '不一致，特征图 batch 维会对不上'))
    return ok


# --------------------------------------------------------------------------- #
# 2. 内外参
# --------------------------------------------------------------------------- #
def audit_intrinsics(src_cfg, dst_cfg, src_sample, dst_sample):
    print('\n' + LINE)
    print('2. 内外参对齐')
    print(LINE)

    for tag, c, s, conv in [
        ('源域 LAAM6D', src_cfg, src_sample,
         '编码器收到的是【已按 IM_RESIZE 缩放过】的内参，投影后只乘 1/stride'),
        ('目标域 MAV6D', dst_cfg, dst_sample,
         '编码器收到的是【原始分辨率】内参，投影后乘 new/raw/stride'),
    ]:
        dc = c.DATA_CONFIG
        print('\n' + SUB)
        print('%s' % tag)
        print('  IM_SIZE(raw)  : %s' % (list(dc.IM_SIZE),))
        print('  IM_RESIZE     : %s   stride=%s  -> 热图 %dx%d' %
              (list(dc.IM_RESIZE), dc.STRIDE,
               dc.IM_RESIZE[0] // dc.STRIDE, dc.IM_RESIZE[1] // dc.STRIDE))
        print('  内参约定      : %s' % conv)
        if s is None:
            continue
        K = np.asarray(s['intrinsic']).reshape(-1, 3, 3)[0]
        D = np.asarray(s['distortion']).reshape(-1)[:5]
        E = np.asarray(s['extrinsic']).reshape(-1, 4, 4)[0]
        print('  实测内参      : fx=%.2f fy=%.2f cx=%.2f cy=%.2f' %
              (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        # 折算回原始分辨率，两边才能公平比较
        if tag.startswith('源域'):
            sx = dc.IM_SIZE[0] / float(dc.IM_RESIZE[0])
            sy = dc.IM_SIZE[1] / float(dc.IM_RESIZE[1])
            fx_raw, fy_raw = K[0, 0] * sx, K[1, 1] * sy
            print('  折回原分辨率  : fx=%.2f fy=%.2f  (乘回 %.3f/%.3f)' % (fx_raw, fy_raw, sx, sy))
        else:
            fx_raw, fy_raw = K[0, 0], K[1, 1]
        print('  水平/垂直 FOV : %.1f° / %.1f°' %
              (fov_deg(fx_raw, dc.IM_SIZE[0]), fov_deg(fy_raw, dc.IM_SIZE[1])))
        print('  畸变          : %s%s' % (np.round(D, 4), '  (全零)' if np.allclose(D, 0) else ''))
        print('  外参          : %s' % ('单位阵' if np.allclose(E, np.eye(4)) else
                                       '非单位阵，平移=%s' % np.round(E[:3, 3], 2)))
        print('  MAX_DIS=%s  MAX_SIZE=%s  OB_SIZE=%s' %
              (dc.MAX_DIS, dc.get('MAX_SIZE', '(缺)'), dc.OB_SIZE[0]))

    # 同一目标在两边占多少像素：这是"表观大小 <-> 距离"关系的核心
    print('\n' + SUB)
    print('目标表观大小对比（网络就是靠这个回归距离的）')
    print('  %-10s %-12s %-10s %-14s %-14s' % ('域', '目标尺寸', '距离', '原图像素', '热图像素'))
    for tag, c, s in [('源域', src_cfg, src_sample), ('MAV6D', dst_cfg, dst_sample)]:
        if s is None:
            continue
        dc = c.DATA_CONFIG
        K = np.asarray(s['intrinsic']).reshape(-1, 3, 3)[0]
        if tag == '源域':
            fx_raw = K[0, 0] * dc.IM_SIZE[0] / float(dc.IM_RESIZE[0])
            dists = [10.0, 50.0, 150.0]
        else:
            fx_raw = K[0, 0]
            dists = [2.0, 5.0, 8.0]
        size_m = float(np.asarray(dc.OB_SIZE[0])[0])
        for d in dists:
            px_raw = size_m * fx_raw / d
            px_hm = px_raw * (dc.IM_RESIZE[0] / float(dc.IM_SIZE[0])) / dc.STRIDE
            print('  %-10s %-12s %-10s %-14s %-14s' %
                  (tag, '%.2f m' % size_m, '%.0f m' % d, '%.1f px' % px_raw, '%.2f px' % px_hm))

    print('\n深度归一化：center_dis 头学的是 Z / MAX_DIS')
    print('  源域   Z=%.0fm -> %.4f ；  Z=%.0fm -> %.4f' %
          (10, 10 / src_cfg.DATA_CONFIG.MAX_DIS, 150, 150 / src_cfg.DATA_CONFIG.MAX_DIS))
    print('  MAV6D  Z=%.0fm -> %.4f ；  Z=%.0fm -> %.4f' %
          (2, 2 / dst_cfg.DATA_CONFIG.MAX_DIS, 8, 8 / dst_cfg.DATA_CONFIG.MAX_DIS))
    print('  => 同一个归一化输出值，在两边代表的米数差 %.1f 倍。'
          % (src_cfg.DATA_CONFIG.MAX_DIS / float(dst_cfg.DATA_CONFIG.MAX_DIS)))


# --------------------------------------------------------------------------- #
# 3. 坐标系 / 旋转
# --------------------------------------------------------------------------- #
def audit_coordinate(src_cfg, dst_cfg):
    print('\n' + LINE)
    print('3. 坐标系与旋转约定对齐')
    print(LINE)

    print("""
                     源域 LAAM6D                     目标域 MAV6D
  GT 存储           世界系 (convert_box_opencv_to_world  相机系 (read_truth_Rt 已用
                    把 8 角点转到世界系)                 camera2vicon 外参转好)
  送进编码器前      pre_processor_laam6d 用             不转换，本来就是相机系
                    inv(extrinsic) 转回相机系
  检测头学到的      相机系深度 Z / MAX_DIS              相机系深度 Z / MAX_DIS
  解码器输出        再转回【世界系】                    留在【相机系】
  评测比较的        世界系坐标                          相机系坐标
""")
    print('  => 关键结论：两边**检测头学的是同一种东西**（相机系深度 + 图像平面中心），')
    print('     世界/相机的差异只发生在编码前和解码后。所以 head 权重是可迁移的，')
    print('     不需要为坐标系改网络，但**解码器必须各用各的**，不能混。')

    print('\n' + SUB)
    print('欧拉角约定')
    print('  源域   : object_encoder_laam6d / pre_processor_laam6d 全部用 zyx')
    print('  MAV6D  : mav6d_*_dataset.py 的 as_euler(xyz) + object_encoder(xyz)')
    print('  rot 头 : 6 通道 = [cos a1, sin a1, cos a2, sin a2, cos a3, sin a3]')
    print('           源域   a1=绕z  a2=绕y  a3=绕x')
    print('           MAV6D  a1=绕x  a2=绕y  a3=绕z')
    print('  => 通道 0/1 和 4/5 的物理含义在两边是【对调】的，')
    print('     直接迁 rot 头等于把第一个和第三个旋转轴接反。')

    rng = np.random.RandomState(0)
    ang = rng.uniform(-np.pi, np.pi, (2000, 3))
    r_xyz = R.from_euler('xyz', ang)
    r_zyx = R.from_euler('zyx', ang)
    rel = (r_xyz.inv() * r_zyx).magnitude()
    deg = np.degrees(rel)
    print('\n  把同一组角按 xyz / zyx 两种顺序解释，得到的旋转差异:')
    print('    中位 %.1f°   均值 %.1f°   90 分位 %.1f°   最大 %.1f°' %
          (np.median(deg), deg.mean(), np.percentile(deg, 90), deg.max()))
    print('  => 这就是修 eval.py 之前，角度指标的误差量级（不是小偏差）。')


# --------------------------------------------------------------------------- #
# 4. 权重可迁移性
# --------------------------------------------------------------------------- #
def audit_weights(src_cfg, dst_cfg, ckpt=None):
    print('\n' + LINE)
    print('4. 权重可迁移性（逐张量比对）')
    print(LINE)

    src_model = build_network(model_cfg=src_cfg.MODEL, dataset=_StubDataset(src_cfg.DATA_CONFIG))
    dst_model = build_network(model_cfg=dst_cfg.MODEL, dataset=_StubDataset(dst_cfg.DATA_CONFIG))

    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    match, shape_bad, only_src, only_dst = [], [], [], []
    for k, v in src_sd.items():
        if k not in dst_sd:
            only_src.append(k)
        elif tuple(v.shape) == tuple(dst_sd[k].shape):
            match.append((k, tuple(v.shape)))
        else:
            shape_bad.append((k, tuple(v.shape), tuple(dst_sd[k].shape)))
    for k in dst_sd:
        if k not in src_sd:
            only_dst.append(k)

    n_match = sum(int(np.prod(s)) for _, s in match)
    n_total = sum(int(v.numel()) for v in dst_sd.values())
    print('源域参数张量 %d 个 / 目标域 %d 个' % (len(src_sd), len(dst_sd)))
    print('  名字+形状都对上 : %d 个张量, %.2fM 参数 (占目标模型 %.1f%%)'
          % (len(match), n_match / 1e6, 100.0 * n_match / max(n_total, 1)))
    print('  名字对上但形状不同 : %d 个   <- 注意 strict=False 并不能放过形状不符，'
          '会抛 RuntimeError；必须先按形状过滤（load_params_from_file 已这么做）'
          % len(shape_bad))
    for k, a, b in shape_bad:
        print('      %-42s 源%s  目标%s' % (k, a, b))
    if only_src:
        print('  仅源域有 : %d 个   例: %s' % (len(only_src), only_src[:3]))
    if only_dst:
        print('  仅目标域有 : %d 个   例: %s' % (len(only_dst), only_dst[:3]))

    if ckpt:
        print('\n' + SUB)
        print('用真实 ckpt 实测: %s' % ckpt)
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('model_state', sd)
        loaded = [k for k, v in sd.items()
                  if k in dst_sd and tuple(v.shape) == tuple(dst_sd[k].shape)]
        skipped = [k for k, v in sd.items()
                   if k in dst_sd and tuple(v.shape) != tuple(dst_sd[k].shape)]
        missing = [k for k in sd if k not in dst_sd]
        n_loaded = sum(int(np.prod(sd[k].shape)) for k in loaded)
        print('  ckpt 张量 %d 个' % len(sd))
        print('  实际能载入 : %d 个, %.2fM 参数 (目标模型的 %.1f%%)'
              % (len(loaded), n_loaded / 1e6, 100.0 * n_loaded / max(n_total, 1)))
        print('  形状不符被跳过 : %d 个 %s' % (len(skipped), skipped[:5]))
        print('  目标模型里没有 : %d 个 %s' % (len(missing), missing[:5]))
        # 注意：不能直接 load_state_dict(sd, strict=False) —— strict=False 只放过
        # 多余/缺失的 key，形状不符照样抛 RuntimeError。必须先按形状过滤。
        # detector_template.load_params_from_file 已按同样逻辑修好。
        filt = {k: v for k, v in sd.items()
                if k in dst_sd and tuple(v.shape) == tuple(dst_sd[k].shape)}
        res = dst_model.load_state_dict(filt, strict=False)
        print('  过滤后实际 load: 载入 %d 个, 仍随机初始化 %d 个'
              % (len(filt), len(res.missing_keys)))

    return len(match), len(shape_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-cfg', default='cfgs/models/uavdet_3d/laam6d/centerdet_rgb_resnet8x.yaml')
    ap.add_argument('--dst-cfg', default='cfgs/models/uavdet_3d/mav6d/centerdet.yaml')
    ap.add_argument('--src-data', default=None)
    ap.add_argument('--dst-data', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--no-samples', action='store_true')
    args = ap.parse_args()

    src_cfg = load_cfg(args.src_cfg, args.src_data)
    dst_cfg = load_cfg(args.dst_cfg, args.dst_data)

    src_sample = dst_sample = None
    if not args.no_samples:
        logger = common_utils.create_logger()
        for name, c, slot in [('源域', src_cfg, 'src'), ('目标域', dst_cfg, 'dst')]:
            try:
                ds, _, _ = build_dataloader(dataset_cfg=c.DATA_CONFIG, batch_size=1, dist=False,
                                            workers=0, logger=logger, training=True)
                s = ds[0]
                if slot == 'src':
                    src_sample = s
                else:
                    dst_sample = s
                print('[%s] 载入样本成功, 数据集 %d 帧' % (name, len(ds)))
            except Exception as e:
                print('[%s] 载入样本失败(%s)，该部分将只用配置信息' % (name, type(e).__name__))

    audit_modality(src_cfg, dst_cfg, src_sample, dst_sample)
    audit_intrinsics(src_cfg, dst_cfg, src_sample, dst_sample)
    audit_coordinate(src_cfg, dst_cfg)
    audit_weights(src_cfg, dst_cfg, args.ckpt)

    print('\n' + LINE)
    print('迁移建议')
    print(LINE)
    print("""  1. 骨干  : 直接迁，这是主要收益来源。
  2. hm    : 类别数不同(7 vs 2)，必然重初始化，无需处理。
  3. rot   : 通道语义在两边对调过，建议重初始化；或把源域也改成 xyz 顺序重训。
  4. center_dis : MAX_DIS 差 %.0f 倍，迁过去初始预测会系统性偏大，
                  建议重初始化该头，或前几个 epoch 只训这一层。
  5. dim   : MAV6D 目标尺寸恒定，这个头几乎没有信息量，迁不迁都行。
  6. 分辨率: 源域 %s，MAV6D %s，全卷积不影响载入，
             但会改变目标的像素尺度，fine-tune 时注意学习率别太大。"""
          % (src_cfg.DATA_CONFIG.MAX_DIS / float(dst_cfg.DATA_CONFIG.MAX_DIS),
             list(src_cfg.DATA_CONFIG.IM_RESIZE), list(dst_cfg.DATA_CONFIG.IM_RESIZE)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
