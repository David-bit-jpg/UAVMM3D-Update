# -*- coding: utf-8 -*-
"""夜间实验各配置的冒烟：建数据集 + 模型，跑 N 个 batch 的前向/反向，打印全部损失项与 state_dict 情况。
在 tools/ 下运行（配置路径相对 tools）：
    D:/Miniconda3/envs/city/python.exe smoke_night.py
教师权重用现成的 teacher_mm（mm20 第 6 轮，结构同 teacher_mix）；保留式微调的教师用 M0_mix 第 12 轮。
"""
import logging
import os
import sys
import time

sys.path.insert(0, 'E:/Open3DUAVDet')
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402

from uavdet3d.config import cfg_from_list, cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu  # noqa: E402

OUT = 'E:/Open3DUAVDet/output/models/uavdet_3d'
T_CK = OUT + '/mmcache/teacher/teacher_mm/ckpt/checkpoint_epoch_6.pth'
M0_CK = OUT + '/mmcache/student_rgb_mix/M0_mix/ckpt/checkpoint_epoch_12.pth'

CASES = [
    ('teacher_mix', 'cfgs/models/uavdet_3d/mmcache/teacher_mix.yaml', [], None),
    ('student_mt_mix', 'cfgs/models/uavdet_3d/mmcache/student_mt_mix.yaml', [], None),
    ('student_kd_mix', 'cfgs/models/uavdet_3d/mmcache/student_kd_mix.yaml', ['MODEL.DISTILL.TEACHER_CKPT', T_CK], None),
    ('student_kdmt_mix', 'cfgs/models/uavdet_3d/mmcache/student_kdmt_mix.yaml', ['MODEL.DISTILL.TEACHER_CKPT', T_CK], None),
    ('centerdet_retain', 'cfgs/models/uavdet_3d/mav6d/centerdet_retain.yaml',
     ['MODEL.DISTILL.TEACHER_CKPT', M0_CK, 'DATA_CONFIG.DATA_PATH', 'E:/MAV6D', 'DATA_CONFIG.SAMPLED_INTERVAL.train', '100'], M0_CK),
]


def run(name, cfg_path, sets, pretrained, n=3, bs=2):
    print('== %s' % name, flush=True)
    t0 = time.time()
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_path, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    logger = logging.getLogger('smoke')
    logging.basicConfig(level=logging.INFO)
    ds, dl, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=bs, dist=False, workers=0, training=True, logger=logger)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    if pretrained:
        dst = float(cfg.DATA_CONFIG.MAX_DIS)
        model.load_params_from_file(filename=pretrained, to_cpu=True, skip_patterns=['hm'], dis_rescale=40.0 / dst)
    model = model.cuda().train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    it = iter(dl)
    for i in range(n):
        bd = next(it)
        load_data_to_gpu(bd)
        ret = model(bd)
        loss = ret['loss']
        opt.zero_grad()
        loss.backward()
        opt.step()
        terms = {k: round(v, 4) for k, v in model.dense_head_2d.loss_terms.items()}
        print('  it%d loss %.4f  %s' % (i, float(loss), terms), flush=True)
    sd = model.state_dict()
    n_aux = sum(k.startswith('aux_heads') for k in sd)
    n_teacher = sum('_teacher' in k for k in sd)
    print('  数据集 %d 帧, image %s, state_dict %d 项（aux %d, 教师 %d）, 显存峰值 %.2f GB, %.0fs'
          % (len(ds), tuple(bd['image'].shape), len(sd), n_aux, n_teacher,
             torch.cuda.max_memory_allocated() / 1e9, time.time() - t0), flush=True)
    del model, opt, dl, ds
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    os.chdir('E:/Open3DUAVDet/tools')
    only = sys.argv[1:] or None
    ok = True
    for name, cfg_path, sets, pre in CASES:
        if only and name not in only:
            continue
        try:
            run(name, cfg_path, sets, pre)
        except Exception as e:  # noqa: BLE001
            ok = False
            import traceback
            traceback.print_exc()
            print('  FAILED %s: %s' % (name, e), flush=True)
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)
