# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2, S7 (GPU): simulation-forgetting measurements for the 5 LOO students (sim labels are allowed).

Measurement = exactly stage 1's (kzst1/sim_diag_adabn.py, = run_kz_0923 line 115): tools/annot_audit/kp_arm_diag.run(
sim_full_r34_kp_w12_zoom.yaml, ckpt, parts='rot', interval=4) on the SIM test split, heading fold-180 median.

  (A) H_own(LOO_k)    the student checkpoint as trained (its BN running statistics are REAL-crop statistics) -- the
                      stage-1 procedure verbatim; reported against the logged kz_s{k} baselines 7.3/7.4/7.1/8.0/7.8 and
                      against stage 1's AdaBN_k numbers (a pure BN-statistics swap of the same seed).
  (B) H_simBN(m)      the same measurement after the BN running statistics of model m are re-estimated on SIM TRAIN crops
                      (stage-1 AdaBN recipe: reset_running_stats, momentum None, BN-only train mode, no_grad, batch 16,
                      numpy default_rng(0) order; frames = sim TRAIN split at the KZ evaluation view (VAL_ZOOMS 2.5),
                      every n-th frame with n = max(1, N_train // 4000)), for m in {LOO_k, kz_s{k}}; weights untouched.
                      This removes the reversible component (BN statistics are re-estimated per domain) and leaves the
                      irreversible one (the trained weights).
Pre-registered decision guardrail (prereg_stage2.json): for every k  H_simBN(LOO_k) <= H_simBN(kz_s{k}) + 1.5 deg.
(A) is reported with its pre-registered interpretation; it does not by itself block deployment.

Output (only under kzst2/sim/): simbn/<name>_simbn.pth, sim_diag_loo.json (written last = success marker).

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst2\\kzst2_sim_diag.py
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
for _p in ('@@ROOT@@/tools/annot_audit', '@@ROOT@@/tools', '@@ROOT@@'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass
import argparse      # noqa: E402
import json          # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import kzst2_common as S   # noqa: E402

LOGGED = [7.3, 7.4, 7.1, 8.0, 7.8]
SIMBN = {'target_frames': 4000, 'batch': 16, 'order_seed': 0, 'split': 'train', 'margin': 1.5}


def measure_with(KD, ck, interval):
    o = KD.run(S.SIM_KP_CFG, ck, 'rot', interval)
    he = np.asarray(o['he'])
    return {'n': int(len(he)), 'heading_fold180_median': float(np.median(np.minimum(he, 180 - he))),
            'heading_unfolded_median': float(np.median(he)), 'tilt_median': float(np.median(o['tilt'])),
            'gt150_frac': float((he > 150).mean())}


def sim_bn_loader(torch, max_frames=0):
    """Sim TRAIN frames at the KZ evaluation view (dataset in test mode -> VAL_ZOOMS window, no augmentation)."""
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.utils import common_utils
    cwd = os.getcwd()
    os.chdir(S.TOOLS)
    try:
        c = EasyDict()
        cfg_from_yaml_file(S.SIM_KP_CFG, c)
        c.DATA_CONFIG.DATA_SPLIT['test'] = SIMBN['split']
        c.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
        ds, _, _ = build_dataloader(c.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=common_utils.create_logger(), training=False)
    finally:
        os.chdir(cwd)
    n_all = len(ds.valid_idx)
    step = max(1, n_all // SIMBN['target_frames'])
    ds.valid_idx = ds.valid_idx[::step]
    if max_frames:
        ds.valid_idx = ds.valid_idx[:max_frames]
    order = np.random.default_rng(SIMBN['order_seed']).permutation(len(ds.valid_idx))
    ld = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, order.tolist()), batch_size=SIMBN['batch'], shuffle=False,
                                     num_workers=0, collate_fn=ds.collate_batch, drop_last=False)
    return ld, {'n_train_all': int(n_all), 'step': int(step), 'n_frames': int(len(ds.valid_idx)), 'cfg': S.SIM_KP_CFG,
                'view': 'test-mode VAL_ZOOMS %s' % list(c.DATA_CONFIG.VAL_ZOOMS), 'norm': 'sim NORM_MEAN/STD of the KZ config'}


def reestimate_bn(models, loader, torch, dev):
    bn_all = []
    for km in models:
        km.eval()
        bns = [m for m in km.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        bn_all.append(bns)
    nb = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            x = torch.from_numpy(np.asarray(batch['image'], np.float32)).to(dev)
            for km in models:
                tb = {'image': x, 'batch_size': int(x.shape[0])}
                for mod in km.module_list:
                    tb = mod(tb)
            nb += 1
            if nb % 50 == 0:
                print('[sim-bn]   %d batches, %.0f s' % (nb, time.time() - t0), flush=True)
    for bns in bn_all:
        for bn in bns:
            bn.eval()
        assert all(int(bn.num_batches_tracked) == nb for bn in bns)
    return nb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--train-root', default=S.TRAIN_DIR)
    ap.add_argument('--out-dir', default=S.SIM_DIR)
    ap.add_argument('--interval', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--with-orig', action='store_true', help='also re-measure the original kz_s{k} as is (reproduction of the logged values, +4 min)')
    ap.add_argument('--max-sim-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--mock-measure', action='store_true', help='CPU smoke test only: kp_arm_diag needs CUDA; returns NaN numbers')
    ap.add_argument('--allow-smoke-students', action='store_true', help='smoke tests only')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    pre, _ = S.check_prereg(a.prereg)
    out = S.ensure_under_out(os.path.abspath(a.out_dir))
    os.makedirs(os.path.join(out, 'simbn'), exist_ok=True)
    res_p = os.path.join(out, 'sim_diag_loo.json')
    if os.path.exists(res_p) and not a.force:
        print('[sim] %s exists (use --force)' % res_p, flush=True)
        return json.load(open(res_p, encoding='utf-8'))
    smoke = bool(a.max_sim_frames or a.mock_measure or a.allow_smoke_students)
    import torch
    import kzst2_test_infer as TI
    dev = torch.device(a.device)
    ckpts, cinfo = TI.student_ckpts(a.train_root, a.allow_smoke_students)
    if a.mock_measure:
        KD = None

        def measure(ck):
            return {'n': 0, 'heading_fold180_median': float('nan'), 'mock': True}
    else:
        import kp_arm_diag as KD          # chdir to tools at import

        def measure(ck):
            return measure_with(KD, ck, a.interval)
    t0 = time.time()
    R = {'prereg_sha256': pre, 'cfg': S.SIM_KP_CFG, 'parts': 'rot', 'interval': a.interval, 'logged_baseline': LOGGED, 'smoke': smoke,
         'students': cinfo, 'own': {}, 'simbn': {'LOO': {}, 'KZ': {}}, 'orig_as_is': {}, 'simbn_recipe': dict(SIMBN)}
    # ---------------- (B) sim-BN re-estimation of the 10 models (same batches for all)
    import adabn_kz as AB
    names = ['LOO_s%d' % k for k in range(5)] + ['KZ_s%d' % k for k in range(5)]
    srcs = list(ckpts) + list(S.RELEASE_KZ)
    models, _, _, _ = AB.build_models(torch, dev, srcs)
    loader, linfo = sim_bn_loader(torch, a.max_sim_frames)
    R['simbn_frames'] = linfo
    nb = reestimate_bn(models, loader, torch, dev)
    simbn_ck = []
    for name, src, km in zip(names, srcs, models):
        src_ck = torch.load(src, map_location='cpu', weights_only=False)
        sd = {kk: v.detach().cpu() for kk, v in km.state_dict().items()}
        for kk, v in src_ck['model_state'].items():
            if not any(kk.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked')):
                assert torch.equal(sd[kk], v.cpu()), (name, kk)
        p = os.path.join(out, 'simbn', '%s_simbn.pth' % name)
        torch.save({'model_state': sd, 'epoch': src_ck.get('epoch'), 'src': src,
                    'simbn': {'n_batches': nb, 'frames': linfo, 'recipe': dict(SIMBN), 'prereg_sha256': pre, 'smoke': smoke}}, p)
        simbn_ck.append(p)
    del models
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    print('[sim] BN statistics of %d models re-estimated on %d sim train frames (%d batches) in %.0f s' % (
        len(names), linfo['n_frames'], nb, time.time() - t0), flush=True)
    # ---------------- measurements
    for k in range(5):
        R['own'][k] = dict(measure(ckpts[k]), ckpt=ckpts[k])
        R['simbn']['LOO'][k] = dict(measure(simbn_ck[k]), ckpt=simbn_ck[k], sha256=S.sha256(simbn_ck[k]))
        R['simbn']['KZ'][k] = dict(measure(simbn_ck[5 + k]), ckpt=simbn_ck[5 + k], sha256=S.sha256(simbn_ck[5 + k]))
        if a.with_orig:
            R['orig_as_is'][k] = dict(measure(S.RELEASE_KZ[k]), ckpt=S.RELEASE_KZ[k])
        print('[sim] s%d: own %.2f | simBN LOO %.2f vs simBN KZ %.2f | logged base %.1f' % (
            k, R['own'][k]['heading_fold180_median'], R['simbn']['LOO'][k]['heading_fold180_median'],
            R['simbn']['KZ'][k]['heading_fold180_median'], LOGGED[k]), flush=True)
    # ---------------- guardrail + reports
    g = {k: bool(R['simbn']['LOO'][k]['heading_fold180_median'] <= R['simbn']['KZ'][k]['heading_fold180_median'] + SIMBN['margin'])
         for k in range(5)}
    R['guardrail_sim_forgetting_simBN'] = {'rule': 'H_simBN(LOO_k) <= H_simBN(kz_s{k}) + 1.5 for every k', 'per_seed': g,
                                           'pass': bool(all(g.values()) and not a.mock_measure)}
    stage1 = {}
    if os.path.exists(S.STAGE1['bn_sim']):
        J1 = json.load(open(S.STAGE1['bn_sim'], encoding='utf-8'))
        stage1 = {int(k): v['heading_fold180_median'] for k, v in J1.get('adabn', {}).items()}
    R['report_own_stage1_procedure'] = {
        k: {'own': R['own'][k]['heading_fold180_median'], 'logged_base': LOGGED[k],
            'own_minus_logged': R['own'][k]['heading_fold180_median'] - LOGGED[k],
            'stage1_rule_pass_(own<=logged+1.5)': bool(R['own'][k]['heading_fold180_median'] <= LOGGED[k] + 1.5),
            'adabn_k_stage1': stage1.get(k), 'own_minus_adabn_k': (R['own'][k]['heading_fold180_median'] - stage1[k]) if k in stage1 else None,
            'simbn_kz_minus_logged': R['simbn']['KZ'][k]['heading_fold180_median'] - LOGGED[k]} for k in range(5)}
    R['elapsed_s'] = time.time() - t0
    R['script_sha256'] = S.sha256(os.path.abspath(__file__))
    R['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    S.jdump(R, res_p)
    print('[sim] guardrail (simBN, pre-registered): %s -> %s ; stage-1 procedure (own BN stats) reported (%.0f s)' % (
        g, R['guardrail_sim_forgetting_simBN']['pass'], time.time() - t0), flush=True)
    return R


if __name__ == '__main__':
    main()
