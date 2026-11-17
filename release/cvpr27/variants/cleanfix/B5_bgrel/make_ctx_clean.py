# -*- coding: utf-8 -*-
"""[cleanfix B5] Writes ctx_check_clean.py = wf20/design/ctx_check.py (source of X_suppl:68, 7.89/8.61/16.13/18.65 deg) with ONLY these changes:
  * real single-frame background bank (variants B_ir, B_v): dev bank wf20/design/bank_kz_sf.npy (dev detector irloc P0_s3)
      -> CLEAN orientation bank D:/S2MUAV3D/srcx_work/in2mav/ou_bank.npy (4,420 windows, clean det1, = the bank the clean students composite with)
  * variant B_sf bank: dev xpaste/bank_bg_sf.npy -> CLEAN refiner bank srcx/in2mav/ref/bank.npy (4,976 128x128 windows; resized as before)
  * networks: the 5 RGB-only simulation KZ nets (byte-identical to release_20260923/kz_s0..4, checked: max |diff| 0) loaded from their
      training dirs output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom/KZs{0..4}/ckpt/checkpoint_epoch_30.pth (KZs0/1 = the clean
      in2mav_none teachers / initabl RGB init). The dev AdaBN family (night1003 kzst1, adapted on dev crops) is dropped (not in the paper).
  * batch size from $CC_BS (default 16 = original) for OOM halving; outputs + example image under review1113/cleanfix/B5_bgrel/.
Sim crops / inpainted backgrounds / masks (wf20/design/sim_samples_test.npz, 552 sim TEST crops, simulation only, no real label) unchanged.
No real label is read anywhere."""
S = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/wf20/design/ctx_check.py'
D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B5_bgrel/ctx_check_clean.py'
t = open(S, encoding='utf-8').read()
REP = [
    ("OUT = ROOT + '/output/camnorm/geo1004/gproto/wf20/design'",
     "OUT = ROOT + '/output/camnorm/geo1004/gproto/review1113/cleanfix/B5_bgrel'\nSIMD = ROOT + '/output/camnorm/geo1004/gproto/wf20/design'"),
    ("DD = os.environ.get('CC_DIR', OUT)", "DD = SIMD"),
    ("bankK = np.load(DD + '/bank_kz_sf.npy', mmap_mode='r')", "bankK = np.load('D:/S2MUAV3D/srcx_work/in2mav/ou_bank.npy', mmap_mode='r')   # [cleanfix] clean orientation bank"),
    ("bankS = np.load((DD if 'CC_DIR' in os.environ else G + '/xpaste') + '/bank_bg_sf.npy', mmap_mode='r')",
     "bankS = np.load(G + '/gproto/srcx/in2mav/ref/bank.npy', mmap_mode='r')   # [cleanfix] clean refiner bank"),
    ("WD = DD + '/w' if 'CC_DIR' in os.environ else None      # 同一份权重的 NVMe 副本（USB 机械盘被别的任务占满时读权重会卡住）\n", ""),
    ("NETS = [('KZ_s%d' % k, (WD + '/kz_s%d.pth' % k) if WD else ROOT + '/output/release_20260923/kz_s%d.pth' % k) for k in range(5)] + \\\n"
     "       [('BN_s%d' % k, (WD + '/kz_s%d_adabn.pth' % k) if WD else ROOT + '/output/camnorm/night1003/scratch/kzst1/adabn/kz_s%d_adabn.pth' % k) for k in range(5)]",
     "NETS = [('KZ_s%d' % k, ROOT + '/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom/KZs%d/ckpt/checkpoint_epoch_30.pth' % k) for k in range(5)]   # [cleanfix]"),
    ("    for key in VAR:\n        outk = np.zeros((N, 16), np.float32)",
     "    if os.path.exists(OUT + '/kp_%s.npz' % name):   # [cleanfix] resume per network\n        KP[name] = dict(np.load(OUT + '/kp_%s.npz' % name)); print(name, 'cached'); del m; continue\n"
     "    for key in VAR:\n        outk = np.zeros((N, 16), np.float32)"),
    ("        for c0 in range(0, N, 16):\n            ch = list(range(c0, min(c0 + 16, N)))",
     "        BS = int(os.environ.get('CC_BS', '16'))\n        for c0 in range(0, N, BS):\n            ch = list(range(c0, min(c0 + BS, N)))"),
    ("        KP[name][key] = outk\n", "        KP[name][key] = outk\n    np.savez(OUT + '/kp_%s.npz' % name, **KP[name])\n"),
    ("for fam in ('KZ', 'BN'):\n    for key in VAR:", "for fam in ('KZ',):\n    for key in VAR:"),
    ("for fam in ('KZ', 'BN'):\n    print(", "for fam in ('KZ',):\n    print("),
    # smoke mode (CC_SMOKE=1): CPU, first 8 crops, 1 net, outputs to OUT/smoke (never mixed with the real outputs)
    ("torch.cuda.set_per_process_memory_fraction(0.15)",
     "SMOKE = os.environ.get('CC_SMOKE') == '1'\nDEV = 'cpu' if SMOKE else 'cuda'\nif SMOKE:\n    OUT = OUT + '/smoke'; os.makedirs(OUT, exist_ok=True)\nelse:\n    torch.cuda.set_per_process_memory_fraction(0.15)"),
    ("N = len(rgb)", "N = 8 if SMOKE else len(rgb)"),
    ("    m = m.cuda().eval()", "    m = m.to(DEV).eval()"),
    ("torch.from_numpy(x[:, None]).cuda()", "torch.from_numpy(x[:, None]).to(DEV)"),
    ("    torch.cuda.empty_cache()", "    torch.cuda.empty_cache() if not SMOKE else None"),
    ("KP = {}\nt0 = time.time()", "KP = {}\nt0 = time.time()\nNETS = NETS[:1] if SMOKE else NETS"),
    ("        kpm = np.mean([KP['%s_s%d' % (fam, k)][key] for k in range(5)], 0)", "        kpm = np.mean([KP[n_][key] for n_ in KP if n_.startswith(fam)], 0)"),
    ("    res['mean_' + fam] = {key: {mk: float(np.mean([res['per_net']['%s_s%d' % (fam, k)][key][mk] for k in range(5)])) for mk in res['per_net']['%s_s0' % fam][key]} for key in VAR}",
     "    res['mean_' + fam] = {key: {mk: float(np.mean([res['per_net'][n_][key][mk] for n_ in KP if n_.startswith(fam)])) for mk in res['per_net']['%s_s0' % fam][key]} for key in VAR}"),
]
for a, b in REP:
    assert t.count(a) == 1, a[:80]
    t = t.replace(a, b)
t = "# [cleanfix B5] generated by make_ctx_clean.py from wf20/design/ctx_check.py — see that file's docstring for the changes\n" + t
open(D, 'w', encoding='utf-8').write(t)
print('ok', D)
