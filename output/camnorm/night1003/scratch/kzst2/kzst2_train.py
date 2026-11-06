# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2, S5 (GPU): fine-tune ONE LOO student  LOO_k  from release_20260923/kz_s{k}.pth.

Same building blocks as tools/train.py (repo untouched): set_random_seed(k), DataLoader(shuffle, batch 8, workers 0,
drop_last False, worker_init_fn seed k), build_network, load_params_from_file, build_optimizer / build_scheduler
(adam_onecycle, LR 2e-4), tools/train_utils/train_utils.train_model (grad clip 35, checkpoint_epoch_<n>.pth).
train.py itself is not used only because it derives its output folder from ROOT_DIR + the cfg path (it would write
outside kzst2/); everything else is the same code path.
Data = KZST2_Crop_Dataset (kzst2_torch.py): deployed crops of the selected gated train frames with the LOO teacher
targets of student k; loss = CenterHeadKZST (quotient L1 only on frames whose LOO teacher needed a D alignment).
Recipe (frozen in prereg_stage2.json): full fine-tune, LOSS_WEIGHTS {hm 1, center_res 2, kp2d 12, others 0},
3 epochs (TRAIN_REPEAT 1, ~3 passes per frame), the LAST checkpoint is the student. No validation, no test evaluation.

Output (only under kzst2/train/LOO_s<k>/): ckpt/checkpoint_epoch_3.pth, train_*.log, cfg_merged.json,
DONE.json (written last = success marker; records sha256 and a reload check into the stock eval network).

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst2\\kzst2_train.py --student 0
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse      # noqa: E402
import datetime      # noqa: E402
import json          # noqa: E402
import time          # noqa: E402
from functools import partial   # noqa: E402
from pathlib import Path        # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass

import numpy as np   # noqa: E402

import kzst2_common as S   # noqa: E402


def to_device(batch, dev, torch):
    """= uavdet3d.model.load_data_to_gpu, but to `dev` (cuda in the real run, cpu in the smoke test)."""
    skip = ['batch_size', 'scene_id', 'seq_id', 'frame_id', 'brightness', 'intrinsic', 'extrinsic', 'distortion', 'raw_im_size',
            'new_im_size', 'obj_size', 'stride', 'im_num', 'obj_num', 'gt_name', 'relative_im_path', 'setting_id', 'gt_diff']
    for key, val in batch.items():
        if isinstance(val, np.ndarray) and key not in skip:
            batch[key] = torch.from_numpy(val).float().to(dev)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', type=int, required=True, choices=range(5))
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--data-dir', default=S.DATA)
    ap.add_argument('--out-root', default=S.TRAIN_DIR)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=S.TRAIN['workers'])
    ap.add_argument('--max-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--epochs', type=int, default=None, help='smoke tests only (the real run uses the frozen 3)')
    ap.add_argument('--log-interval', type=int, default=50)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    smoke = bool(a.max_frames or a.epochs is not None)
    k = a.student
    pre, _ = S.check_prereg(a.prereg)
    run_dir = Path(S.ensure_under_out(os.path.join(os.path.abspath(a.out_root), 'LOO_s%d' % k)))
    done = run_dir / 'DONE.json'
    if done.exists() and not a.force:
        print('[train] %s exists -- student %d already trained (use --force to retrain)' % (done, k), flush=True)
        return json.load(open(done, encoding='utf-8'))
    if done.exists():
        done.unlink()
    ckpt_dir = run_dir / 'ckpt'
    for old in ckpt_dir.glob('checkpoint_epoch_*.pth') if ckpt_dir.exists() else []:
        old.unlink()                                  # train_model never resumes here: a fresh run every time
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # build marker of the pseudo labels / crops must verify (prereg + sha256)
    bj = os.path.join(a.data_dir, 'pseudo_stage2.json')
    B = json.load(open(bj, encoding='utf-8'))
    pseudo, crops = os.path.join(a.data_dir, B['npz']), os.path.join(a.data_dir, B['crops'])
    if os.environ.get('UAV_STRICT_HASH') == '1' and (B['prereg_sha256'] != pre or S.sha256(pseudo) != B['npz_sha256']):  # [release]
        raise SystemExit('pseudo labels do not verify against %s / the prereg' % bj)
    if B.get('smoke') and not smoke:
        raise SystemExit('the pseudo labels in %s are a SMOKE build' % a.data_dir)

    import torch
    for p in (S.ROOT, S.TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import kzst2_torch as KT
    from torch.utils.data import DataLoader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    from tools.train_utils.optimization import build_optimizer, build_scheduler
    from tools.train_utils.train_utils import train_model

    dev = torch.device(a.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA requested but not available'
    log_file = run_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)
    cfg = KT.build_cfg(k, pseudo, crops, a.max_frames)
    KT.assert_cfg(cfg)
    epochs = int(cfg.OPTIMIZATION.NUM_EPOCHS) if a.epochs is None else int(a.epochs)
    if not smoke:
        assert epochs == S.TRAIN['epochs'] and a.workers in (0, 1)
    json.dump(cfg, open(run_dir / 'cfg_merged.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
    logger.info('KZ-ST stage 2: student LOO_s%d from %s; prereg %s; device %s; epochs %d; workers %d; smoke %s' % (
        k, S.RELEASE_KZ[k], pre, dev, epochs, a.workers, smoke))
    logger.info('merged AUG %s (== prereg, asserted) | LOSS_WEIGHTS %s | LR %.1e %s' % (
        dict(cfg.DATA_CONFIG.AUG), dict(cfg.MODEL.DENSE_HEAD_2D.LOSS_WEIGHTS), cfg.OPTIMIZATION.LR, cfg.OPTIMIZATION.OPTIMIZER))

    common_utils.set_random_seed(k)                       # train.py --fix_random_seed --seed k
    ds = KT.KZST2_Crop_Dataset(cfg.DATA_CONFIG, training=True, logger=logger)
    loader = DataLoader(ds, batch_size=int(cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU), pin_memory=(dev.type == 'cuda'), num_workers=a.workers,
                        shuffle=True, collate_fn=ds.collate_batch, drop_last=S.TRAIN['drop_last'], timeout=0,
                        worker_init_fn=partial(common_utils.worker_init_fn, seed=k))
    n_iter = len(loader) * epochs
    if not smoke:
        assert n_iter <= S.TRAIN['iter_cap'], n_iter
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    n_ld, n_tot = model.load_params_from_file(filename=S.RELEASE_KZ[k], to_cpu=(dev.type == 'cpu'), logger=logger)
    assert n_ld == n_tot, 'kz_s%d did not load completely (%d / %d)' % (k, n_ld, n_tot)
    n_bn = KT.count_bn(model)
    assert n_bn > 0 and type(model.dense_head_2d).__name__ == 'CenterHeadKZST'
    model.to(dev)
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    lr_scheduler, _ = build_scheduler(optimizer, total_iters_each_epoch=len(loader), total_epochs=epochs, last_epoch=-1,
                                      optim_cfg=cfg.OPTIMIZATION)
    model.train()
    logger.info('frames %d (quotient %d) | iterations %d (= %d epochs x %d) | BN layers %d | all %d parameter tensors trainable: %s' % (
        len(ds.frame_idx), int(ds.P['quotient'][ds.frame_idx, k].sum()), n_iter, epochs, len(loader), n_bn,
        sum(1 for _ in model.parameters()), all(p.requires_grad for p in model.parameters())))
    terms_seen = {}

    def model_func(model_, batch):
        to_device(batch, dev, torch)
        ret = model_(batch)
        loss = ret['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss %s (terms %s)' % (loss, model_.dense_head_2d.loss_terms))
        model_.update_global_step()
        for kk, vv in model_.dense_head_2d.loss_terms.items():
            terms_seen.setdefault(kk, []).append(float(vv))
        return loss

    t0 = time.time()
    train_model(model, optimizer, loader, model_func=model_func, lr_scheduler=lr_scheduler, optim_cfg=cfg.OPTIMIZATION,
                start_epoch=0, total_epochs=epochs, start_iter=0, rank=0, tb_log=None, ckpt_save_dir=ckpt_dir,
                train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=epochs + 1,
                logger=logger, log_interval=a.log_interval, use_amp=False)
    train_s = time.time() - t0
    last = ckpt_dir / ('checkpoint_epoch_%d.pth' % epochs)
    assert last.exists(), last
    # reload check: the student loads 289/289 into the STOCK deployed eval network (CenterHead, mav6d_r34_kp.yaml)
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    cwd = os.getcwd()
    os.chdir(S.TOOLS)
    try:
        ecfg = EasyDict()
        cfg_from_yaml_file(S.KP_CFG, ecfg)
        ecfg.DATA_CONFIG.DATA_SPLIT['test'] = 'val'          # dataset object only constructs the network
        eds, _, _ = build_dataloader(ecfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=logger, training=False)
    finally:
        os.chdir(cwd)
    em = build_network(ecfg.MODEL, eds)
    n2, t2 = em.load_params_from_file(str(last), to_cpu=True, logger=logger)
    assert n2 == t2, 'student checkpoint does not load into the stock eval network'
    sd_s = torch.load(str(last), map_location='cpu', weights_only=False)['model_state']
    sd_0 = torch.load(S.RELEASE_KZ[k], map_location='cpu', weights_only=False)['model_state']
    changed = [kk for kk in sd_0 if not torch.equal(sd_0[kk].cpu(), sd_s[kk].cpu())]
    bn_changed = [kk for kk in changed if kk.endswith('running_mean') or kk.endswith('running_var')]
    del em
    summ = {kk: {'first50': float(np.mean(v[:50])), 'last50': float(np.mean(v[-50:]))} for kk, v in terms_seen.items() if v}
    J = {'student': k, 'init': S.RELEASE_KZ[k], 'init_sha256': S.sha256(S.RELEASE_KZ[k]), 'prereg_sha256': pre, 'smoke': smoke,
         'device': str(dev), 'ckpt': str(last), 'ckpt_sha256': S.sha256(str(last)), 'epochs': epochs, 'iterations': n_iter,
         'iters_per_epoch': len(loader), 'n_frames': int(len(ds.frame_idx)), 'n_quotient_frames': int(ds.P['quotient'][ds.frame_idx, k].sum()),
         'n_bn_layers': n_bn, 'tensors_changed': len(changed), 'bn_stat_tensors_changed': len(bn_changed), 'tensors_total': len(sd_0),
         'loss_terms_first_last50': summ, 'train_seconds': train_s, 'pseudo_npz_sha256': B['npz_sha256'], 'crops_sha256': B['crops_sha256'],
         'script_sha256': S.sha256(os.path.abspath(__file__)), 'torch_module_sha256': S.sha256(os.path.join(S.HERE, 'kzst2_torch.py')),
         'common_sha256': S.sha256(os.path.join(S.HERE, 'kzst2_common.py')), 'reload_into_stock_eval_network': '%d/%d' % (n2, t2),
         'log': str(log_file), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    S.jdump(J, str(done))
    logger.info('[train] LOO_s%d done in %.0f s: %s (loss terms first/last 50 it: %s)' % (k, train_s, last, summ))
    return J


if __name__ == '__main__':
    main()
