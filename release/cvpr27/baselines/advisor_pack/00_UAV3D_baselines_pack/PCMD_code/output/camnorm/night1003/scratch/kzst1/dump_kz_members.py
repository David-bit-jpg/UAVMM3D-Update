# -*- coding: utf-8 -*-
"""E1 / SA-TM GPU dump (2026-10-03). Wrapper around tools/annot_audit/dump_real_preds.py -- repo is NOT modified.

What it does
  * runs dump_real_preds.main() with exactly the command line that produced output/camnorm/kz/real_ens5flip_kpKZavg_zoom.pkl
    (run_kz_0923.py dump('kpKZavg_zoom', kz)): 5 ramp position models + flip TTA + causal100 temporal bg + self-calibrated lwh
    + KZ zoom 95, real test split, greedy 1 m;
  * the only difference: --kp2d-from gets 11 checkpoints = 5 RGB-KZ (release_20260923/kz_s0..s4.pth, bit-identical to the
    checkpoint_epoch_30.pth files of the original dump) followed by the 6 multimodal KZ students;
  * kp_zoom.install_zoom_kp2d_hook is monkeypatched with a verbatim copy whose run_b forwards all 11 members but hands the
    pipeline exactly the deployed value: torch.stack(maps[:5]).mean(0) and torch.stack(hms[:5]).mean(0) (same ops, same order
    as the original with 5 members) -> deployed outputs stay bit-identical;
  * per frame it records, for every member, the sigmoid hm and the 16-d kp2d vector (raw, before /z) at all cells of the 5x5
    search window (NaN-padded where the window is clipped at the map border), plus read-out cell, z, u, v, crop, the position
    peak cell + center_res, K_in, distortion, raw/new image size, stride, KNOWN_SIZE, and the deployed kp2d value the decoder
    actually reads.

Writes ONLY into this folder:  real_ens5flip_kpKZavg_zoom_<tag>.pkl  kz_members_<tag>.npz  kz_members_<tag>.json

    cd @@ROOT@@\\tools
    set PYTHONUTF8=1 & set PYTHONPATH=@@ROOT@@ & set TORCH_HOME=@@TORCH_HOME@@
    python <scratchpad>\\e1sa\\dump_kz_members.py
"""
import sys

sys.dont_write_bytecode = True          # do not drop .pyc files into @@ROOT@@

import argparse   # noqa: E402
import hashlib    # noqa: E402
import json       # noqa: E402
import os         # noqa: E402
import time       # noqa: E402

import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
REL23 = ROOT + '/output/release_20260923'
MODELS = ROOT + '/output/models/uavdet_3d/camnorm'
KP_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml'      # same as run_kz_0923.REAL_KP / run_kd_0923.REAL_KP
ZOOM = '95'
RGB = [('KZs%d' % i, REL23 + '/kz_s%d.pth' % i) for i in range(5)]
MM = [('KZKD_s0', MODELS + '/sim_full_r34_kp_w12_zoom_kdmm2/KZKD_s0/ckpt/checkpoint_epoch_30.pth'),
      ('KZKD_s1', MODELS + '/sim_full_r34_kp_w12_zoom_kdmm2/KZKD_s1/ckpt/checkpoint_epoch_30.pth'),
      ('KZKDir_s0', MODELS + '/sim_full_r34_kp_w12_zoom_kdir/KZKDir_s0/ckpt/checkpoint_epoch_30.pth'),
      ('KZKDir_s1', MODELS + '/sim_full_r34_kp_w12_zoom_kdir/KZKDir_s1/ckpt/checkpoint_epoch_30.pth'),
      ('KZMMH_s0', MODELS + '/sim_full_r34_kp_w12_zoom_mmh/KZMMH_s0/ckpt/checkpoint_epoch_30.pth'),
      ('KZMMH_s1', MODELS + '/sim_full_r34_kp_w12_zoom_mmh/KZMMH_s1/ckpt/checkpoint_epoch_30.pth')]
N_RGB = 5

REC = {'rows': [], 'info': {}, 't0': time.time()}


def _np(x, dtype=np.float64):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype)


def make_hook(member_names, n_rgb=N_RGB, rec=REC):
    """Returns a drop-in replacement for kp_zoom.install_zoom_kp2d_hook (same signature, same return value).
    Everything that reaches the pipeline is computed by the same statements as tools/kp_zoom.py (2026-09-22 version);
    the recording happens on copies and never writes back."""
    import cv2
    import torch
    import kp_zoom as KZ

    def install_zoom_kp2d_hook(model, kp_models, source, target_w2d=95.0, z_range=(1.0, 4.0), search=2,
                               norm_mean=None, norm_std=None, log=None, oracle_gt=False):
        assert not oracle_gt, 'E1 dump never uses the oracle crop (it would read ground truth)'
        assert len(kp_models) == len(member_names), (len(kp_models), member_names)
        head = model.dense_head_2d
        orig = head.forward
        pp = model.model_cfg.POST_PROCESSING
        pp.POSE_FROM_KP2D = True
        pp.KP2D_POSE_PARTS = 'rot'
        stride = int(model.dataset.dataset_cfg.STRIDE)
        in_w, in_h = int(model.dataset.new_im_width), int(model.dataset.new_im_hight)
        nm = np.asarray(norm_mean, np.float32).reshape(3, 1, 1) if norm_mean is not None else None
        ns = np.asarray(norm_std, np.float32).reshape(3, 1, 1) if norm_std is not None else None
        stats = {'n': 0, 'z': [], 'rows': []}     # rows: (seq, frame, z, w2d_pred)
        dev = next(kp_models[0].parameters()).device          # == .cuda() on the GPU run; lets the CPU self-test run too
        S = 2 * search + 1
        M = len(kp_models)
        try:
            from uavdet3d.datasets.pre_processor.pre_processor import KP_SCALE
        except Exception:      # noqa: BLE001
            KP_SCALE = None
        rec['info'].update({
            'member_names': list(member_names), 'n_rgb': n_rgb, 'search': search, 'target_w2d': float(target_w2d),
            'z_range': list(map(float, z_range)), 'stride': stride, 'in_wh': [in_w, in_h], 'KP_SCALE': KP_SCALE,
            'W': float(getattr(model.dataset, 'W', np.nan)), 'H': float(getattr(model.dataset, 'H', np.nan)),
            'dec_kwargs': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                           for k, v in dict(getattr(model, 'dec_kwargs', {})).items()},
            'KNOWN_SIZE_at_install': [float(v) for v in pp.get('KNOWN_SIZE', [])] if pp.get('KNOWN_SIZE', None) is not None else None,
            'SIZE_SOURCE_at_install': str(pp.get('SIZE_SOURCE', '')),
            'norm_mean': None if norm_mean is None else list(map(float, np.ravel(norm_mean))),
            'norm_std': None if norm_std is None else list(map(float, np.ravel(norm_std))),
            'device': str(dev),
        })

        def run_b(img_in):
            x = img_in.astype(np.float32).transpose(2, 0, 1) / 255.0
            if nm is not None:
                x = (x - nm) / ns
            t = torch.from_numpy(x[None, None]).to(dev)
            maps, hms = [], []
            with torch.no_grad():
                for km in kp_models:                              # RGB members first (same order as deployed), then MM
                    tb = {'image': t, 'batch_size': 1}
                    for mod in km.module_list:
                        tb = mod(tb)
                    maps.append(tb['pred_center_dict']['kp2d'].float()[0])
                    hms.append(torch.sigmoid(tb['pred_center_dict']['hm'].float()[0, 0]))
            # deployed value: identical statement to kp_zoom.run_b with the 5 RGB members
            return torch.stack(maps[:n_rgb]).mean(0), torch.stack(hms[:n_rgb]).mean(0), maps, hms

        def fwd(batch_dict):
            out = orig(batch_dict)
            pc = batch_dict['pred_center_dict']
            hm = torch.sigmoid(pc['hm'].float())                      # (B*K, 1, h, w)
            B = hm.shape[0]
            kp_out = torch.zeros((B, 16, hm.shape[-2], hm.shape[-1]), dtype=torch.float32, device=hm.device)
            seqs = [str(s) for s in batch_dict['seq_id']]
            frames = [str(f) for f in batch_dict['frame_id']]
            pending = []
            for b in range(B):
                flat = hm[b, 0].reshape(-1)
                idx = int(torch.argmax(flat))
                r, c = idx // hm.shape[-1], idx % hm.shape[-1]
                u = (c + float(pc['center_res'][b, 0, r, c])) * stride
                v = (r + float(pc['center_res'][b, 1, r, c])) * stride
                w2d = float(torch.exp(pc['size2d'][b, 0, r, c].float().clamp(-5, 9)))
                z = float(np.clip(target_w2d / max(w2d, 1e-3), z_range[0], z_range[1]))
                img, K_raw = source.get(seqs[b], frames[b])
                raw_h, raw_w = img.shape[:2]
                x0, y0, w, h = KZ.crop_window(u, v, z, raw_w, raw_h, in_w, in_h)
                s = in_w / w
                Maff = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
                crop = cv2.warpAffine(img, Maff, (in_w, in_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                kp_map, hm_b, maps_all, hms_all = run_b(crop)
                uc = (u * raw_w / in_w - x0) * s
                vc = (v * raw_h / in_h - y0) * s
                cc, rc = int(uc / stride), int(vc / stride)
                r0, r1 = max(0, rc - search), min(hm_b.shape[0], rc + search + 1)
                c0, c1 = max(0, cc - search), min(hm_b.shape[1], cc + search + 1)
                sub = hm_b[r0:r1, c0:c1]
                j = int(torch.argmax(sub.reshape(-1)))
                rb, cb = r0 + j // sub.shape[1], c0 + j % sub.shape[1]
                kp_out[b, :, r, c] = kp_map[:, rb, cb] / z                 # window input px -> full input px
                stats['n'] += 1
                stats['z'].append(z)
                stats['rows'].append((seqs[b], frames[b], z, w2d))
                # ---------------- E1 recording (copies only) ----------------
                hw = torch.stack(hms_all)[:, r0:r1, c0:c1].float().cpu().numpy()          # (M, nr, nc)
                kw = torch.stack(maps_all)[:, :, r0:r1, c0:c1].float().cpu().numpy()      # (M, 16, nr, nc)
                hm_win = np.full((M, S, S), np.nan, np.float32)
                kp_win = np.full((M, 16, S, S), np.nan, np.float32)
                ar, ac = r0 - (rc - search), c0 - (cc - search)
                hm_win[:, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = hw
                kp_win[:, :, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = kw
                raw_size = _np(batch_dict['raw_im_size'][b]).reshape(-1)
                new_size = _np(batch_dict['new_im_size'][b]).reshape(-1)
                pending.append((b, r, c, {
                    'seq': seqs[b], 'frame': frames[b],
                    'pos_rc': (r, c), 'center_res': (float(pc['center_res'][b, 0, r, c]), float(pc['center_res'][b, 1, r, c])),
                    'hm_peak': float(hm[b, 0, r, c]), 'u': u, 'v': v, 'w2d': w2d, 'z': z,
                    'crop': (x0, y0, w, h, s), 'uc_vc': (uc, vc), 'rc_cc': (rc, cc), 'win': (r0, r1, c0, c1), 'rb_cb': (rb, cb),
                    'raw_img_wh': (raw_w, raw_h), 'K_raw_src': np.asarray(K_raw, np.float64),
                    'K_in': _np(batch_dict['intrinsic'][b]).reshape(-1, 3, 3)[0],
                    'dist': _np(batch_dict['distortion'][b]).reshape(-1),
                    'raw_size': raw_size[:2], 'new_size': new_size[:2], 'stride_b': float(_np(batch_dict['stride'][b]).reshape(-1)[0]),
                    'known_size': np.asarray(pp.get('KNOWN_SIZE', [np.nan] * 3), np.float64).reshape(-1)[:3],
                    'hm_win': hm_win, 'kp_win': kp_win,
                }))
            pc['kp2d'] = kp_out.to(pc['hm'].dtype)
            # the value the decoder will actually read (after the dtype cast above)
            for b, r, c, row in pending:
                row['kp_deployed'] = pc['kp2d'][b, :, r, c].float().cpu().numpy().astype(np.float32)
                row['kp2d_dtype'] = str(pc['kp2d'].dtype)
                rec['rows'].append(row)
            n = len(rec['rows'])
            if n % 400 < B:
                print('[e1sa] %d frames recorded, %.0f s' % (n, time.time() - rec['t0']), flush=True)
            return out
        head.forward = fwd
        return stats

    return install_zoom_kp2d_hook


def save_records(out_dir, tag, rec=REC, extra_info=None):
    rows = rec['rows']
    if not rows:
        print('[e1sa] nothing recorded, nothing saved', flush=True)
        return None
    ld = {len(r['dist']) for r in rows}
    assert len(ld) == 1, 'distortion length not constant: %s' % ld
    A = {
        'seq': np.array([r['seq'] for r in rows]), 'frame': np.array([r['frame'] for r in rows]),
        'pos_rc': np.array([r['pos_rc'] for r in rows], np.int64), 'center_res': np.array([r['center_res'] for r in rows], np.float64),
        'hm_peak': np.array([r['hm_peak'] for r in rows], np.float64),
        'u': np.array([r['u'] for r in rows], np.float64), 'v': np.array([r['v'] for r in rows], np.float64),
        'w2d': np.array([r['w2d'] for r in rows], np.float64), 'z': np.array([r['z'] for r in rows], np.float64),
        'crop': np.array([r['crop'] for r in rows], np.float64), 'uc_vc': np.array([r['uc_vc'] for r in rows], np.float64),
        'rc_cc': np.array([r['rc_cc'] for r in rows], np.int64), 'win': np.array([r['win'] for r in rows], np.int64),
        'rb_cb': np.array([r['rb_cb'] for r in rows], np.int64), 'raw_img_wh': np.array([r['raw_img_wh'] for r in rows], np.int64),
        'K_raw_src': np.stack([r['K_raw_src'] for r in rows]), 'K_in': np.stack([r['K_in'] for r in rows]),
        'dist': np.stack([r['dist'] for r in rows]), 'raw_size': np.stack([r['raw_size'] for r in rows]),
        'new_size': np.stack([r['new_size'] for r in rows]), 'stride_b': np.array([r['stride_b'] for r in rows], np.float64),
        'known_size': np.stack([r['known_size'] for r in rows]),
        'hm_win': np.stack([r['hm_win'] for r in rows]), 'kp_win': np.stack([r['kp_win'] for r in rows]),
        'kp_deployed': np.stack([r['kp_deployed'] for r in rows]),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'kz_members_%s.npz' % tag)
    with open(path + '.tmp', 'wb') as f:
        np.savez(f, **A)
    os.replace(path + '.tmp', path)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    info = dict(rec['info'])
    info.update({'n_frames': len(rows), 'npz': os.path.basename(path), 'npz_sha256': h,
                 'kp2d_dtype': sorted({r['kp2d_dtype'] for r in rows}), 'elapsed_s': time.time() - rec['t0'],
                 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    if extra_info:
        info.update(extra_info)
    with open(os.path.join(out_dir, 'kz_members_%s.json' % tag), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print('[e1sa] wrote %s (%d frames, %.1f MB, sha256 %s)' % (path, len(rows), os.path.getsize(path) / 1e6, h), flush=True)
    return path


def build_argv(pkl):
    """exactly run_kz_0923.dump('kpKZavg_zoom', kz): dump_real_preds.py --kp2d-from <ckpts> --kp2d-cfg REAL_KP --out <pkl> --kp2d-zoom 95
    (all other options at their defaults = 5 ramp models + flip + self-calib + causal100 + real test split)."""
    return ['dump_real_preds.py', '--kp2d-from'] + [p for _, p in RGB + MM] + ['--kp2d-cfg', KP_CFG, '--out', pkl, '--kp2d-zoom', ZOOM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=HERE, help='must be this scratchpad folder (or below it)')
    ap.add_argument('--tag', default='e1sa')
    ap.add_argument('--force', action='store_true', help='overwrite an existing dump of the same tag')
    a, rest = ap.parse_known_args()
    if rest:
        raise SystemExit('unexpected arguments %s -- the dump command line is fixed on purpose' % rest)
    out_dir = os.path.abspath(a.out_dir)
    if not os.path.normcase(out_dir).startswith(os.path.normcase(HERE)):
        raise SystemExit('refusing to write outside %s' % HERE)
    pkl = os.path.join(out_dir, 'real_ens5flip_kpKZavg_zoom_%s.pkl' % a.tag)
    if os.path.exists(pkl) and not a.force:
        raise SystemExit('%s exists (use --force to redo)' % pkl)
    members = RGB + MM
    missing = [p for _, p in members if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing checkpoints: %s' % missing)
    names = [n for n, _ in members]

    sys.argv = build_argv(pkl)
    for p in (ROOT, TOOLS, TOOLS + '/annot_audit'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import kp_zoom                         # noqa: E402  (tools/kp_zoom.py; dump_real_preds imports the same module object)
    import dump_real_preds as DRP          # noqa: E402  (chdir to tools at import, like the original run)
    orig_hook = kp_zoom.install_zoom_kp2d_hook
    kp_zoom.install_zoom_kp2d_hook = make_hook(names)
    print('[e1sa] cwd %s; argv %s' % (os.getcwd(), ' '.join(sys.argv[1:])), flush=True)
    REC['t0'] = time.time()
    ok = False
    try:
        DRP.main()
        ok = True
    finally:
        kp_zoom.install_zoom_kp2d_hook = orig_hook
        save_records(out_dir, a.tag, extra_info={
            'members': [{'name': n, 'path': p, 'role': 'rgb_deployed' if i < N_RGB else 'mm_recorded_only'} for i, (n, p) in enumerate(members)],
            'dump_pkl': os.path.basename(pkl), 'dump_argv': sys.argv[1:], 'dump_finished_ok': ok,
            'mirrors': 'output/camnorm/kz/real_ens5flip_kpKZavg_zoom.pkl (run_kz_0923.py dump kpKZavg_zoom)'})
    if ok and os.path.exists(pkl):
        print('[e1sa] dump pkl %s sha256 %s' % (pkl, hashlib.sha256(open(pkl, 'rb').read()).hexdigest()), flush=True)


if __name__ == '__main__':
    main()
