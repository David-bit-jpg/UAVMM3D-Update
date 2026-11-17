# -*- coding: utf-8 -*-
"""camnorm 系列统一评测：类别无关位姿指标（uavdet3d/utils/pose_eval.py，分母 = 全部 GT）
+ 可选的仓库原版 LAA3D_ADS 打印（6dof error/accuracy、3D AP_R40、2D AP_R40、ADS）。

    python eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt <权重>.pth --tag T \
        --split test --json ../output/camnorm/json/T.json --ads

与旧 eval_on_mav6d.py / eval_ads_mav6d.py 的区别（都对应审查条目）：
- 两域几何约定统一（虚拟深度、MAX_DIS / MAX_SIZE 相同），零样本不需要任何 --decode-max-* 换算（P2）
- ADS 前显式 set_default_euler_seq(EULER_SEQ)，九点生成 / 角度误差都按存储约定 'xyz'（P16）
- ADS 尺寸误差按 l,w,h 参数算（P15，见 eval6dof_laam6d.py）
- 准确率一律以全部 GT 为分母（P19）
- 权重必须逐张量全部载入，否则直接报错（迁移最怕以为载上了其实没有）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402
from uavdet3d.utils import common_utils, frame_convention, pose_eval   # noqa: E402

CARLA_TO_OPENCV = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

ADS_PROFILES = {
    'laa': dict(DisMax=150, DisMin=0, MinPixel=16,
                AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                AP3D=dict(DisThresh=[1, 2, 4, 6], RecallNum=101),
                Dof6=dict(DisNormMax=8, OriNormMax=30, SizeNormMax=1)),
    'indoor': dict(DisMax=10, DisMin=0, MinPixel=16,
                   AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                   AP3D=dict(DisThresh=[0.05, 0.1, 0.2, 0.5], RecallNum=101),
                   Dof6=dict(DisNormMax=1, OriNormMax=30, SizeNormMax=0.2)),
}


def ads_eval(records, raw_whs, W, H, seq, profiles, out_root, tag):
    """仓库原版 LAA3D_ADS。框在 OpenCV 相机系，传 extrinsic=CARLA->OpenCV 让 box9d_to_2d 的两步变换抵消；
    内参换算回原图分辨率（MinPixel=16 是按原图像素定义的）。"""
    from uavdet3d.datasets.laam6d.ads_metric_laam6d import LAA3D_ADS_Metric
    from uavdet3d.datasets.laam6d.dataset_utils import convert_9params_to_9points
    frame_convention.set_default_euler_seq(seq)          # 审查 P16

    def pts(p):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 9)
        return convert_9params_to_9points(p) if len(p) else np.empty((0, 9, 3), np.float32)

    annos = []
    for r, (rw, rh) in zip(records, raw_whs):
        annos.append({
            'seq_id': r['seq_id'], 'frame_id': r['frame_id'],
            'gt_names': np.array(['drone'] * len(r['gt'])), 'gt_boxes': pts(r['gt']),
            'pred_names': np.array(['drone'] * len(r['pred'])), 'pred_boxes': pts(r['pred']),
            'confidence': r['conf'].astype(np.float32),
            'intrinsic': cg.scale_K(r['K'], rw / float(W), rh / float(H)),
            'extrinsic': CARLA_TO_OPENCV, 'distortion': np.zeros(5),
        })
    reports = {}
    for prof in profiles:
        out = os.path.join(out_root, tag, prof)
        os.makedirs(out, exist_ok=True)
        m = LAA3D_ADS_Metric(eval_config=EasyDict(ADS_PROFILES[prof]), classes=['drone'], metric_save_path=out)
        s = m.eval(annos)
        open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(s + '\n')
        reports[prof] = s
    return reports


def frame_key(f):
    """帧号排序键：纯数字的帧名按数值排。MAV6D phantom4 的帧名是不补零的 1.jpg … 395.jpg，
    按字符串排会变成 1, 10, 100, 101 …（2026-09-20 踩过：在线背景混进了未来帧、卡尔曼按乱序滤波）。"""
    stem = os.path.splitext(os.path.basename(str(f)))[0]
    return (0, int(stem), '') if stem.isdigit() else (1, 0, stem)


def temporal_background(model, loader, mode):
    """热图时序背景（固定相机）：先跑一遍原始热图，按序列算每帧的背景图。

    为什么（docs/audit/NIGHT_2026-09-19.md §3.11）：纯仿真模型在 MAV6D mavic2 上检出只有 8~20%，
    但目标并非没有响应 —— 58% 的帧真值在前 5 个候选里，被固定位置的背景假峰（网结 / 灯 / 货架）压住。
    固定相机下这些假峰每帧都在同一处、无人机在动；把每格在序列里的常态响应除掉，动的目标就浮出来。
      p50       离线：整段序列逐格中位数（含当前帧之后的帧）
      causalN   在线：只用当前帧之前的 N 帧的逐格中位数；之前不足 10 帧时不做背景（返回 None）
    返回 {(seq_id, frame_id): 背景图 (H, W) 或 None}。
    """
    import numpy as np
    from uavdet3d.model import load_data_to_gpu
    per_seq = {}
    with torch.no_grad():
        for batch in loader:
            seqs = [str(s) for s in batch['seq_id']]
            frames = [str(f) for f in batch['frame_id']]
            load_data_to_gpu(batch)
            model(batch)
            hm = torch.sigmoid(model.dense_head_2d.forward_loss_dict['pred_center_dict']['hm']).float().cpu().numpy()
            hm = hm.reshape(len(seqs), -1, *hm.shape[-2:])[:, 0]
            for b, (s, f) in enumerate(zip(seqs, frames)):
                per_seq.setdefault(s, []).append((f, hm[b]))
    out = {}
    for s, items in per_seq.items():
        items.sort(key=lambda x: frame_key(x[0]))
        stack = np.stack([h for _, h in items])
        if mode == 'p50':
            bg = np.median(stack, axis=0)
            for f, _ in items:
                out[(s, f)] = bg
        elif mode.startswith('causal'):
            n = int(mode[6:])
            for t, (f, _) in enumerate(items):
                out[(s, f)] = np.median(stack[max(0, t - n):t], axis=0) if t >= 10 else None
        else:
            raise ValueError('--temporal-bg 只能是 p50 / causalN，收到 %r' % mode)
    return out


def install_background_hook(model, bgmap, eps=0.02):
    """检测头前向之后，把热图换成「除以背景」的结果（再转回 logit，后面的峰值/回归/置信度流程不变）。"""
    head = model.dense_head_2d
    orig = head.forward

    def fwd(batch_dict):
        out = orig(batch_dict)
        pc = batch_dict['pred_center_dict']
        logit = pc['hm']
        p = torch.sigmoid(logit.float())
        seqs = [str(s) for s in batch_dict['seq_id']]
        frames = [str(f) for f in batch_dict['frame_id']]
        k = p.shape[0] // max(1, len(seqs))
        for b, key in enumerate(zip(seqs, frames)):
            bg = bgmap.get(key)
            if bg is None:
                continue
            bgt = torch.as_tensor(bg, device=p.device, dtype=p.dtype)
            r = p[b * k:(b + 1) * k] / (bgt + eps)
            p[b * k:(b + 1) * k] = r / (1.0 + r)
        p = p.clamp(1e-6, 1 - 1e-6)
        pc['hm'] = torch.log(p / (1 - p)).to(logit.dtype)
        return out
    head.forward = fwd


def euler6_to_mat(v):
    """euler6（三个角各 cos/sin，通道在最后一维）-> 旋转矩阵。约定与 rotation_repr.euler_to_vec + 外旋 'xyz' 一致：
    R = Rz(a3) Ry(a2) Rx(a1)。先把每对 (cos, sin) 归一化（网络回归出来的不一定在单位圆上）。"""
    def n(c, s):
        r = torch.sqrt(c * c + s * s + 1e-12)
        return c / r, s / r
    ca, sa = n(v[..., 0], v[..., 1])
    cb, sb = n(v[..., 2], v[..., 3])
    cc, sc = n(v[..., 4], v[..., 5])
    row0 = torch.stack([cb * cc, sa * sb * cc - ca * sc, ca * sb * cc + sa * sc], -1)
    row1 = torch.stack([cb * sc, sa * sb * sc + ca * cc, ca * sb * sc - sa * cc], -1)
    row2 = torch.stack([-sb, sa * cb, ca * cb], -1)
    return torch.stack([row0, row1, row2], -2)


def mat_to_euler6(m):
    b = torch.asin((-m[..., 2, 0]).clamp(-1, 1))
    a = torch.atan2(m[..., 2, 1], m[..., 2, 2])
    c = torch.atan2(m[..., 1, 0], m[..., 0, 0])
    return torch.stack([torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b), torch.cos(c), torch.sin(c)], -1)


def _chan_last(x):
    """(B, 6, H, W) -> (B, H, W, 6)"""
    return x.permute(0, 2, 3, 1)


def _chan_first(x):
    return x.permute(0, 3, 1, 2)


def align_rot_to(ref6, other6):
    """把 other 的旋转对齐到 ref：在 {R, R·Rz(180°)} 里取与 ref 夹角小的那个（配方里前后等价，不同权重可能差 180°）。
    ref6 / other6: (B, 6, H, W)。"""
    A = euler6_to_mat(_chan_last(ref6))
    B_ = euler6_to_mat(_chan_last(other6))
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=B_.device, dtype=B_.dtype))
    B2 = B_ @ flip
    tr = lambda X: (A * X).sum(dim=(-1, -2))                      # trace(A^T X)，越大夹角越小
    pick = (tr(B_) >= tr(B2)).unsqueeze(-1).unsqueeze(-1)
    return _chan_first(mat_to_euler6(torch.where(pick, B_, B2)))


def mean_rot(vecs):
    """一组已对齐的 euler6 逐通道平均（cos/sin 平均等价于角度的圆均值方向）。"""
    return torch.stack(vecs).mean(0)


def hflip_rot_back(v6):
    """把「翻转图」上预测的旋转映射回原图：R = M R' S，M = diag(-1,1,1)（相机镜像），S = diag(1,-1,1)（机体左右对称面）。
    与训练增广 camera_geometry.hflip_rotation 同一个映射（对合），allocentric 下同样成立（视线也一起镜像）。"""
    Rm = euler6_to_mat(_chan_last(v6))
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=Rm.device, dtype=Rm.dtype))
    S = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=Rm.device, dtype=Rm.dtype))
    return _chan_first(mat_to_euler6(M @ Rm @ S))


def install_ensemble_hook(model, extras, rot_mode='main'):
    """多模型（多种子）输出集成：主模型检测头之后，让 extras 在同一输入上各跑一遍，逐项平均后交给主模型的解码器。
    热图在概率空间平均；center_res / center_dis / dim / size2d 算术平均。
    rot：'main' = 只用主模型（缺省，与 2026-09-20 之前的结果一致）；'align' = 把成员的旋转在 {R, R·Rz(180°)}
    里对齐到主模型后再平均 —— 配方里前后等价，不对齐直接平均 euler6 会得到无意义的旋转。
    注意旋转会经水平跨度影响几何解深度，所以改 rot 也会动位置。"""
    head = model.dense_head_2d
    orig = head.forward

    def fwd(batch_dict):
        out = orig(batch_dict)
        pc = batch_dict['pred_center_dict']
        dicts = [pc]
        with torch.no_grad():
            for em in extras:
                tb = {k: v for k, v in batch_dict.items() if k != 'pred_center_dict'}
                for mod in em.module_list:
                    tb = mod(tb)
                dicts.append(tb['pred_center_dict'])
        if rot_mode == 'align' and 'rot' in pc:
            ref = pc['rot'].float()
            pc['rot'] = mean_rot([ref] + [align_rot_to(ref, d['rot'].float()) for d in dicts[1:]]).to(pc['rot'].dtype)
        for k in list(pc.keys()):
            # kp2d（2026-09-22）同 rot：只用主模型。有序角点在成员之间可能差一个前后对调，直接平均会得到退化的框
            if k in ('rot', 'kp2d'):
                continue
            vals = [d[k].float() for d in dicts if k in d]
            if k == 'hm':
                p = torch.stack([torch.sigmoid(v) for v in vals]).mean(0).clamp(1e-6, 1 - 1e-6)
                pc[k] = torch.log(p / (1 - p)).to(pc[k].dtype)
            else:
                pc[k] = torch.stack(vals).mean(0).to(pc[k].dtype)
        return out
    head.forward = fwd


def temporal_fuse(model, loader, alpha, dilate=3):
    """热图时序融合（在线）：f_t = (1-α)·p_t + α·maxpool(f_{t-1})，p_t 是当前（已装的钩子处理后）的热图概率。
    maxpool 给前一帧的峰留出目标移动的余量；只用过去帧。每段序列按帧号排序；返回 {(seq_id, frame_id): f_t}。"""
    import torch.nn.functional as F
    from uavdet3d.model import load_data_to_gpu
    per_seq = {}
    with torch.no_grad():
        for batch in loader:
            seqs = [str(s) for s in batch['seq_id']]
            frames = [str(f) for f in batch['frame_id']]
            load_data_to_gpu(batch)
            model(batch)
            hm = torch.sigmoid(model.dense_head_2d.forward_loss_dict['pred_center_dict']['hm'].float()).cpu()
            hm = hm.reshape(len(seqs), -1, *hm.shape[-2:])
            for b, (s, f) in enumerate(zip(seqs, frames)):
                per_seq.setdefault(s, []).append((f, hm[b]))
    out = {}
    for s, items in per_seq.items():
        items.sort(key=lambda x: frame_key(x[0]))
        prev = None
        for f, p in items:
            cur = p if prev is None else (1 - alpha) * p + alpha * F.max_pool2d(prev[None], dilate, 1, dilate // 2)[0]
            out[(s, f)] = cur.numpy()
            prev = cur
    return out


def install_replace_hm_hook(model, hmmap):
    """检测头前向之后，把热图整张换成预先算好的（概率图），其余输出不动。"""
    head = model.dense_head_2d
    orig = head.forward

    def fwd(batch_dict):
        out = orig(batch_dict)
        pc = batch_dict['pred_center_dict']
        seqs = [str(s) for s in batch_dict['seq_id']]
        frames = [str(f) for f in batch_dict['frame_id']]
        p = torch.stack([torch.as_tensor(hmmap[key], device=pc['hm'].device) for key in zip(seqs, frames)])
        p = p.reshape(pc['hm'].shape).float().clamp(1e-6, 1 - 1e-6)     # (B, K*cls, h, w) -> (B*K, cls, h, w)
        pc['hm'] = torch.log(p / (1 - p)).to(pc['hm'].dtype)
        return out
    head.forward = fwd


def install_kp2d_source_hook(model, kp_model):
    """kp_model 可以是一个网络，也可以是列表（多种子）：多个时 kp2d 图逐格算术平均后再进 PnP。"""
    kp_models = list(kp_model) if isinstance(kp_model, (list, tuple)) else [kp_model]
    return _install_kp2d_source_hook(model, kp_models)


def _install_kp2d_source_hook(model, kp_models):
    """旋转来源分离（2026-09-22）：位置网络 model 照常出检出 / 中心 / 深度 / 尺寸，另一个带 kp2d 头的网络 kp_model
    在同一输入上跑一遍，把它的 kp2d 图塞进 model 的输出，再让 model 的解码器只用 PnP 换旋转（KP2D_POSE_PARTS=rot）。
    位置路径一个数都不动：深度仍用 model 自己 rot 头的跨度（审计发现 size2d 预测与 rot 头输出耦合，换旋转进深度会伤位置）。
    kp2d 在 model 的峰值格上读取（model 的 2D 中心误差中位 ~2 px，与 kp_model 自己的峰基本同格）。
    装在集成钩子之后（集成只平均位置类输出）、翻转钩子之前（翻转钩子本来就跳过 kp2d）。"""
    head = model.dense_head_2d
    orig = head.forward
    pp = model.model_cfg.POST_PROCESSING
    pp.POSE_FROM_KP2D = True
    pp.KP2D_POSE_PARTS = 'rot'

    def fwd(batch_dict):
        out = orig(batch_dict)
        maps = []
        with torch.no_grad():
            for km in kp_models:
                tb = {k: v for k, v in batch_dict.items() if k != 'pred_center_dict'}
                for mod in km.module_list:
                    tb = mod(tb)
                maps.append(tb['pred_center_dict']['kp2d'].float())
        batch_dict['pred_center_dict']['kp2d'] = torch.stack(maps).mean(0).to(maps[0].dtype) if len(maps) > 1 else maps[0]
        return out
    head.forward = fwd


def install_flip_hook(model, rot=False):
    """水平翻转 TTA：检测头之后，把输入水平翻转再跑一遍同一个模型，输出翻回来与原输出平均。
    翻转后再右移 1 像素，使像素 u -> W - u（而不是 W-1-u）：stride 整除输入宽时热图格子正好镜像，
    格内偏移 res_x -> 1 - res_x。热图在概率空间平均；center_dis / dim / size2d 翻转不变，直接平均；
    rot：缺省只用原输出；rot=True 时把翻转那一遍的旋转按 R = M R' S 映射回来（同训练增广的映射），
    对齐 180° 后与原输出平均。"""
    head = model.dense_head_2d
    orig = head.forward

    def fwd(batch_dict):
        out = orig(batch_dict)
        pc = batch_dict['pred_center_dict']
        img = torch.flip(batch_dict['image'], dims=[-1])
        img = torch.cat([img[..., :1], img[..., :-1]], dim=-1)
        tb = {k: v for k, v in batch_dict.items() if k != 'pred_center_dict'}
        tb['image'] = img
        with torch.no_grad():
            for mod in model.module_list:
                if mod is head:
                    tb = orig(tb)
                    break
                tb = mod(tb)
        head.forward_loss_dict['pred_center_dict'] = pc        # 翻转那一遍会改写它；时序背景要读原来的
        fd = tb['pred_center_dict']
        if rot and 'rot' in fd:
            back = hflip_rot_back(torch.flip(fd['rot'].float(), dims=[-1]))
            ref = pc['rot'].float()
            pc['rot'] = mean_rot([ref, align_rot_to(ref, back)]).to(pc['rot'].dtype)
        for k in list(pc.keys()):
            if k in ('rot', 'kp2d') or k not in fd:
                continue
            v = torch.flip(fd[k].float(), dims=[-1])
            if k == 'hm':
                p = ((torch.sigmoid(pc[k].float()) + torch.sigmoid(v)) / 2).clamp(1e-6, 1 - 1e-6)
                pc[k] = torch.log(p / (1 - p)).to(pc[k].dtype)
                continue
            if k == 'center_res':
                v = v.clone()
                v[:, 0] = 1.0 - v[:, 0]
            pc[k] = ((pc[k].float() + v) / 2).to(pc[k].dtype)
        return out
    head.forward = fwd


def adapt_bn(cfg, model, split, interval, batch_size, max_batches, logger):
    """AdaBN：在目标域【无标注】图像上重估 BN 统计量（清零后累计平均，随机打乱顺序），权重不动。
    只有 BN 层切 train，模型其余部分（含 MixStyle）保持 eval。"""
    import copy
    from uavdet3d.model import load_data_to_gpu
    dc = copy.deepcopy(cfg.DATA_CONFIG)
    dc.DATA_SPLIT['test'] = split
    dc.SAMPLED_INTERVAL['test'] = interval
    ds_, _, _ = build_dataloader(dataset_cfg=dc, batch_size=batch_size, dist=False,
                                 workers=0, logger=logger, training=False)
    loader = torch.utils.data.DataLoader(ds_, batch_size=batch_size, shuffle=True, num_workers=0,
                                         collate_fn=ds_.collate_batch, generator=torch.Generator().manual_seed(0))
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= max_batches:
                break
            load_data_to_gpu(batch)
            model(batch)
            n += 1
    for bn in bns:
        bn.eval()
    return len(bns), n


def self_calibrate_size(cfg, model, split, interval, conf_thr, batch_size, logger, bg_mode=None):
    """在目标域的【无标注】图像上估一个固定机身尺寸：只按检测置信度挑框，取预测 lwh 的逐分量中位。
    bg_mode（如 causal100）：标定时也做热图时序背景减除（标定 split 自己算背景，用完把钩子撤掉），
    免得静态背景上的高置信假检测（尺寸是退化的）混进中位数。

    为什么这样做（docs/audit/NOSIZE_2026-09-19.md）：dim 头在真实域上【总体几乎无偏】（中位 x0.91~0.96，
    且换一个先验差 39% 的机队重训仍是 x0.99，不是在背先验），坏的只是逐帧散布（log 0.30），
    而同一批无人机的物理尺寸本就是常数 —— 取中位把散布压掉，就逼近了「知道真值尺寸」的上界。
    """
    import copy
    from uavdet3d.model import load_data_to_gpu
    dc = copy.deepcopy(cfg.DATA_CONFIG)
    dc.DATA_SPLIT['test'] = split
    dc.SAMPLED_INTERVAL['test'] = interval
    _, loader, _ = build_dataloader(dataset_cfg=dc, batch_size=batch_size, dist=False,
                                    workers=0, logger=logger, training=False)
    prev = model.model_cfg.POST_PROCESSING.get('SIZE_SOURCE', 'pred')
    model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'pred'          # 标定必须用模型自己预测的尺寸
    head_fwd = model.dense_head_2d.forward
    if bg_mode:
        install_background_hook(model, temporal_background(model, loader, bg_mode))
    got = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            res = model(batch)
            for b in range(res['batch_size']):
                p = np.asarray(res['pred_boxes9d'][b]).reshape(-1, 10)
                c = np.asarray(res['confidence'][b]).reshape(-1)
                if not len(c) or c[0] < conf_thr or p[0, 2] <= 0:
                    continue
                got.append(np.abs(p[0, 3:6]))
    model.model_cfg.POST_PROCESSING.SIZE_SOURCE = prev
    model.dense_head_2d.forward = head_fwd
    if not got:
        raise RuntimeError('自标定没有拿到任何检测（split=%s，conf>=%.2f）' % (split, conf_thr))
    return np.median(np.asarray(got), axis=0), len(got)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--interval', type=int, default=1)
    ap.add_argument('--match', default='top1', choices=['top1', 'greedy'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--json', default=None)
    ap.add_argument('--ads', action='store_true', help='同时跑仓库原版 LAA3D_ADS 并打印')
    ap.add_argument('--ads-profiles', default='laa,indoor')
    ap.add_argument('--ads-out', default='@@ROOT@@/output/camnorm/ads')
    ap.add_argument('--self-calib', default=None, metavar='SPLIT',
                    help='先在该 split 的【无标注】图像上跑一遍，把预测尺寸的逐分量中位冻成 KNOWN_SIZE 再评测。'
                         '不给定机型尺寸时用它：三个权重实测把位置中位从 0.455 做到 0.333 m，与给真值尺寸持平。'
                         '标定只用检测置信度挑框，不读任何真值；标定 split 应与评测 split 分开（如 val / test）。')
    ap.add_argument('--temporal-bg', default=None, metavar='MODE',
                    help='热图时序背景减除（前提：每段序列相机固定）。p50 = 整段逐格中位数（离线）；'
                         'causalN = 只用之前 N 帧（在线，如 causal100）。不用标签、不改模型。')
    ap.add_argument('--ensemble', nargs='+', default=None, metavar='CKPT',
                    help='额外的同结构权重（如其他种子），与 --ckpt 做输出集成（热图/回归平均，旋转用主模型）')
    ap.add_argument('--temporal-fuse', type=float, default=None, metavar='ALPHA',
                    help='在线热图时序融合：f_t = (1-α)p_t + α·maxpool3(f_{t-1})，装在时序背景之后')
    ap.add_argument('--flip-rot', action='store_true',
                    help='翻转 TTA 也平均旋转（把翻转那一遍的 R 按 M R S 映射回来）')
    ap.add_argument('--ensemble-rot', default='main', choices=['main', 'align'],
                    help='集成时旋转的处理：main = 只用主模型（缺省）；align = 对齐 180° 后平均')
    ap.add_argument('--flip-tta', action='store_true', help='水平翻转 TTA（翻转一遍、输出翻回来平均，旋转用原输出）')
    ap.add_argument('--kp2d-from', default=None, nargs='+', metavar='CKPT',
                    help='旋转来源分离：另一个带 kp2d 头的权重，只贡献 8 角点 → PnP 解旋转；位置全部仍来自 --ckpt（及其集成）')
    ap.add_argument('--kp2d-cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr_kp.yaml',
                    help='--kp2d-from 权重对应的评测配置（网络结构要带 kp2d 头）')
    ap.add_argument('--kp2d-zoom', type=float, default=None, metavar='W2D',
                    help='放大裁窗版旋转来源（KZ 臂）：按位置网络预测的 2D 宽把去畸变原图裁到目标宽≈W2D px 再喂 kp2d 网络；'
                         '装在集成/翻转/背景之后。不给 = 整幅喂（K30 方式）')
    ap.add_argument('--kp2d-raw-root', default='@@MAV6D@@', help='--kp2d-zoom 用的真实原图根目录')
    ap.add_argument('--adabn', default=None, metavar='SPLIT',
                    help='AdaBN：先在该 split 的【无标注】图像上重估 BN 统计量（不读真值）')
    ap.add_argument('--adabn-interval', type=int, default=4)
    ap.add_argument('--adabn-batches', type=int, default=300)
    ap.add_argument('--self-calib-bg', action='store_true',
                    help='自标定时也按 --temporal-bg 做时序背景减除（在标定 split 上自己算背景）')
    ap.add_argument('--self-calib-interval', type=int, default=2,
                    help='门限 0.7 下每个种子在 val 上仍有 150~220 个样本（间隔 2）')
    ap.add_argument('--self-calib-conf', type=float, default=0.7,
                    help='标定只收这个置信度以上的检测。低置信度的框尺寸是退化的（实测 h 塌到 0.07~0.12），'
                         '会把中位拉偏两三成。门限不需要真值就能选：标定值随门限变化，取它不再变化的平台'
                         '（annot_audit/calib_stability.py --thrs）。2026-09-20：0.5 在 5 个种子里有 2 个落在'
                         '退化簇上（对角 0.43~0.46 m，ADS 31~36），0.6~0.8 是平台，改为 0.7 后 5 个种子 ADS 44.6~45.5。')
    ap.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.split
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = args.interval
    seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                     workers=args.workers, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    n_loaded, n_total = model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    if n_loaded != n_total:
        raise RuntimeError('权重没有全部载入（%d/%d），结构与配置不一致，拒绝评测' % (n_loaded, n_total))
    model.cuda().eval()
    if args.adabn:
        nb, nbatch = adapt_bn(cfg, model, args.adabn, args.adabn_interval, args.batch_size, args.adabn_batches, logger)
        print('AdaBN（%s split）：%d 个 BN 层，%d 个 batch' % (args.adabn, nb, nbatch))

    if args.ensemble:
        extras = []
        for ck in args.ensemble:
            em = build_network(model_cfg=cfg.MODEL, dataset=ds)
            a_, b_ = em.load_params_from_file(filename=ck, to_cpu=False, logger=logger)
            if a_ != b_:
                raise RuntimeError('集成成员权重没有全部载入：%s' % ck)
            em.cuda().eval()
            if args.adabn:
                adapt_bn(cfg, em, args.adabn, args.adabn_interval, args.batch_size, args.adabn_batches, logger)
            extras.append(em)
        install_ensemble_hook(model, extras, rot_mode=args.ensemble_rot)   # 先装集成，自标定和时序背景都基于集成后的输出
        print('输出集成：主模型 + %d 个成员' % len(extras))

    if args.kp2d_from:
        kcfg = EasyDict()
        cfg_from_yaml_file(args.kp2d_cfg, kcfg)
        kds, _, _ = build_dataloader(dataset_cfg=kcfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                     workers=0, logger=logger, training=False)
        assert str(kcfg.DATA_CONFIG.DATA_PATH) == str(cfg.DATA_CONFIG.DATA_PATH), 'kp2d 网络与主网络必须读同一份数据'
        kms = []
        for ck in args.kp2d_from:
            km = build_network(model_cfg=kcfg.MODEL, dataset=kds)
            a_, b_ = km.load_params_from_file(filename=ck, to_cpu=False, logger=logger)
            if a_ != b_:
                raise RuntimeError('kp2d 网络权重没有全部载入：%s' % ck)
            kms.append(km.cuda().eval())
        if args.kp2d_zoom is None:
            install_kp2d_source_hook(model, kms)
            print('旋转来源分离：kp2d ← %d 个网络 %s，PnP 只换旋转，位置不变' % (len(kms), args.kp2d_from))

    if args.flip_tta:
        install_flip_hook(model, rot=args.flip_rot)   # 装在集成外层：翻转那一遍也走集成
        print('水平翻转 TTA')

    if args.self_calib:
        ks, n_det = self_calibrate_size(cfg, model, args.self_calib, args.self_calib_interval,
                                        args.self_calib_conf, args.batch_size, logger,
                                        bg_mode=args.temporal_bg if args.self_calib_bg else None)
        cfg.MODEL.POST_PROCESSING.SIZE_SOURCE = 'known'
        cfg.MODEL.POST_PROCESSING.KNOWN_SIZE = ks.tolist()
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
        model.model_cfg.POST_PROCESSING.KNOWN_SIZE = ks.tolist()
        print('无标签自标定（%s split，%d 个检测）：lwh = %s，对角 %.3f m' % (
            args.self_calib, n_det, np.round(ks, 4).tolist(), float(np.linalg.norm(ks))))

    if args.temporal_bg:
        bgmap = temporal_background(model, loader, args.temporal_bg)
        n_bg = sum(v is not None for v in bgmap.values())
        install_background_hook(model, bgmap)
        print('热图时序背景（%s）：%d / %d 帧有背景图' % (args.temporal_bg, n_bg, len(bgmap)))

    if args.kp2d_from and args.kp2d_zoom is not None:
        # 放大裁窗版旋转来源：装在最外层，读到的峰值格 = 解码器用的那个
        import kp_zoom
        src = kp_zoom.RawSource(ds, args.kp2d_raw_root)
        kp_zoom.install_zoom_kp2d_hook(model, kms, src, target_w2d=args.kp2d_zoom,
                                       norm_mean=cfg.DATA_CONFIG.get('NORM_MEAN', None), norm_std=cfg.DATA_CONFIG.get('NORM_STD', None))
        print('旋转来源分离（放大裁窗，目标 2D 宽 %.0f px）：kp2d ← %d 个网络 %s' % (args.kp2d_zoom, len(kms), args.kp2d_from))

    if args.temporal_fuse:
        install_replace_hm_hook(model, temporal_fuse(model, loader, args.temporal_fuse))
        print('热图时序融合（在线）：alpha = %.2f' % args.temporal_fuse)

    summary, recs = pose_eval.evaluate(model, loader, seq=seq, match=args.match)
    raw_whs = [tuple(ds.metas[int(i)].get('raw_wh', (ds.W, ds.H))) for i in ds.valid_idx]
    assert len(raw_whs) == len(recs)

    print('=' * 100)
    print(pose_eval.format_summary('%s [%s %d 帧]' % (args.tag, args.split, len(recs)), summary))
    if 'pos_median' in summary:
        print('  准确率（分母 = 全部 %d 个 GT，没检出算失败）' % summary['n_gt'])
        print('    位置 <5cm %.1f%%  <10cm %.1f%%  <20cm %.1f%%  <50cm %.1f%%'
              % tuple(100 * summary['ACC_pos_%g' % t] for t in (0.05, 0.1, 0.2, 0.5)))
        print('    朝向 <5° %.1f%%  <10° %.1f%%  <20° %.1f%%  <30° %.1f%%   (180° 折算: %.1f / %.1f / %.1f / %.1f%%)'
              % (tuple(100 * summary['ACC_rot_%d' % t] for t in (5, 10, 20, 30)) +
                 tuple(100 * summary['ACC_rotfold_%d' % t] for t in (5, 10, 20, 30))))
        print('    位置且朝向 <5cm&5° %.1f%%  <10cm&10° %.1f%%  <20cm&20° %.1f%%   ADD<10%%直径 %.1f%%'
              % (100 * summary['ACC_0.05m5deg'], 100 * summary['ACC_0.1m10deg'], 100 * summary['ACC_0.2m20deg'],
                 100 * summary['ACC_add_10']))
    print('=' * 100)

    res = dict(summary)
    res.update({'tag': args.tag, 'ckpt': args.ckpt, 'split': args.split, 'interval': args.interval, 'cfg': args.cfg})
    best_json = os.path.join(os.path.dirname(args.ckpt), 'best.json')
    if os.path.basename(args.ckpt) == 'best.pth' and os.path.exists(best_json):
        res['best'] = json.load(open(best_json, encoding='utf-8'))
    if args.ads:
        reports = ads_eval(recs, raw_whs, ds.W, ds.H, seq, args.ads_profiles.split(','), args.ads_out, args.tag)
        for prof, s in reports.items():
            print('\n' + '#' * 30 + ' LAA3D_ADS 口径=%s ' % prof + '#' * 30)
            print(s)
        res['ads_reports'] = reports
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('写出 %s' % args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
