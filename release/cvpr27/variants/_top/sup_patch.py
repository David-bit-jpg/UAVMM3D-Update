# -*- coding: utf-8 -*-
"""[review1113 sup2/sup3] text patches that turn srcx/lib/*.py into isolated, restartable copies (shared by sup2/make_nogate_lib.py and
sup3/make_seed_lib.py). Every patch asserts its anchor, so a silent mismatch with srcx/lib is impossible. The method code is untouched except:
  * sys.path points at the copy dir (so `import sx` picks the redirected constants);
  * batch sizes become env-controlled (defaults = deployed values) so the wrapper scripts can halve them on CUDA OOM;
  * training loops save a resume state (refiner: every epoch / 250 steps; orientation student: every 250 steps) and continue from it;
  * optional: SX_SEEDSHIFT is added to every RNG seed of a TRAINED component (refiner stages 1-3, orientation students); file names keep
    the nominal seed (ref_st*_s5/s6, <tag>_s0/s1) so the glue code is unchanged.
Also writes infer_wrap.py: runs any inference script under torch.cuda.set_per_process_memory_fraction($SX_INFER_MEMFRAC, default 0.2)."""
import os

SRC = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'
OLDP = "sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')"


def rd(f):
    return open(SRC + '/' + f, encoding='utf-8').read()


def rep(t, a, b, n=1):
    assert t.count(a) == n, (a[:80], t.count(a), n)
    return t.replace(a, b)


def path(t, D):
    return rep(t, OLDP, "sys.path.insert(0, '%s')" % D) if OLDP in t else t


def wr(D, f, t, hdr):
    os.makedirs(D, exist_ok=True)
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113 copy of srcx/lib/%s -> %s] %s\n' % (f, D, hdr) + t)


SX_OW = "O = HOME + '/' + COMBO                      # per-combo outputs (recs, ref, ou, preds, logs)\nW = 'D:/S2MUAV3D/srcx_work/' + COMBO        # per-combo large crop arrays\n"


def ori_bs(t):
    BS = "int(os.environ.get('ORI_BS', '16'))"
    for x, y in (('def batches(X, idx, bs=16):', 'def batches(X, idx, bs=None):\n    bs = bs or %s' % BS), ('np.array(el), 16)', 'np.array(el), %s)' % BS),
                 ('np.arange(len(info)), 16)', 'np.arange(len(info)), %s)' % BS)):
        assert x in t, x
        t = t.replace(x, y)   # all occurrences (adabn + kps BN passes), as make_nomir_lib.py
    return t


def det_bs(t):
    return rep(t, 'batch_size=8, dist=False, workers=0', "batch_size=int(os.environ.get('DET_BS', '8')), dist=False, workers=0", 2)


def refiner(t, seedshift=False):
    """inference batch $REF_BS (128), stage-1 batch $REF_ST1_BS (128), stage-2/3 batch $REF_TRAIN_BS (64); resume state ref/resume_st<k>_s<j>.pt"""
    t = rep(t, 'if len(buf) == 128:', "if len(buf) == int(os.environ.get('REF_BS', '128')):")
    TB = "int(os.environ.get('REF_TRAIN_BS', '64'))"
    t = rep(t, 'rng.choice(len(S), 64, replace=False)', 'rng.choice(len(S), %s, replace=False)' % TB)
    t = rep(t, 'ri = rng.choice(pool, 64, replace=len(pool) < 64)', 'ri = rng.choice(pool, %s, replace=len(pool) < %s)' % (TB, TB))
    t = rep(t, "BS, EP = 128, int(os.environ.get('REF_EP', '30'))", "BS, EP = int(os.environ.get('REF_ST1_BS', '128')), int(os.environ.get('REF_EP', '30'))")
    t = rep(t, "log = open('%s/train_st%d_s%d%s.log' % (RO, stage, j, tag), 'w')", "log = open('%s/train_st%d_s%d%s.log' % (RO, stage, j, tag), 'a')\n"
            "    RS = '%s/resume_st%d_s%d%s.pt' % (RO, stage, j, tag)")
    if seedshift:
        t = rep(t, 'torch.manual_seed(j); rng = np.random.default_rng(1000 + j)',
                "js = j + int(os.environ.get('SX_SEEDSHIFT', '0')); torch.manual_seed(js); rng = np.random.default_rng(1000 + js)\n"
                "    print('[sup3] refiner stage %d nominal seed %d -> effective seed %d' % (stage, j, js), flush=True)")
    # stage 1: resume per epoch (only with the same batch size: OneCycleLR total_steps depends on it)
    t = rep(t, """        net.train(); t0 = time.time()
        for ep in range(EP):
            perm = rng.permutation(len(S))""", """        net.train(); t0 = time.time(); ep0 = 0
        if os.path.exists(RS):
            z = torch.load(RS, map_location=dev, weights_only=False)
            if z.get('BS') == BS:
                net.load_state_dict(z['net']); opt.load_state_dict(z['opt']); sch.load_state_dict(z['sch']); rng.bit_generator.state = z['rng']
                torch.set_rng_state(z['trng']); ep0 = z['ep']; print('RESUME stage 1 from epoch %d (%s)' % (ep0, RS), file=log, flush=True); print('RESUME stage 1 from epoch %d' % ep0, flush=True)
            else:
                print('resume state has batch %s != %d (OOM halving): stage 1 restarts from scratch' % (z.get('BS'), BS), file=log, flush=True)
        for ep in range(ep0, EP):
            perm = rng.permutation(len(S))""")
    t = rep(t, """            print('ep %d loss %.4f %.0fs' % (ep, loss.item(), time.time() - t0), file=log, flush=True)
        torch.save(net.state_dict(), ckp(1, j, tag)); return""", """            print('ep %d loss %.4f %.0fs' % (ep, loss.item(), time.time() - t0), file=log, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(), ep=ep + 1, BS=BS), RS)
        torch.save(net.state_dict(), ckp(1, j, tag)); os.path.exists(RS) and os.remove(RS); return""")
    # stages 2/3: resume every 250 steps (cosine schedule over STEPS does not depend on the batch)
    t = rep(t, """    net.train(); t0 = time.time()
    pool = np.arange(len(xr)) if stage == 2 else ok
    for st in range(STEPS):""", """    net.train(); t0 = time.time(); st0 = 0
    pool = np.arange(len(xr)) if stage == 2 else ok
    if os.path.exists(RS):
        z = torch.load(RS, map_location=dev, weights_only=False)
        net.load_state_dict(z['net']); ema.load_state_dict(z['ema']); opt.load_state_dict(z['opt']); sch.load_state_dict(z['sch'])
        if disc is not None:
            disc.load_state_dict(z['disc'])
        rng.bit_generator.state = z['rng']; torch.set_rng_state(z['trng']); st0 = z['st']
        print('RESUME stage %d from step %d (%s)' % (stage, st0, RS), file=log, flush=True); print('RESUME stage %d from step %d' % (stage, st0), flush=True)
    for st in range(st0, STEPS):""")
    t = rep(t, """        if st % 200 == 0:
            print('step %d loss %.4f %.0fs' % (st, loss.item(), time.time() - t0), file=log, flush=True)
    torch.save(ema.state_dict(), ckp(stage, j, tag))""", """        if st % 200 == 0:
            print('step %d loss %.4f %.0fs' % (st, loss.item(), time.time() - t0), file=log, flush=True)
        if (st + 1) % 250 == 0 and st + 1 < STEPS:
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), disc=(disc.state_dict() if disc is not None else None),
                            rng=rng.bit_generator.state, trng=torch.get_rng_state(), st=st + 1), RS)
    torch.save(ema.state_dict(), ckp(stage, j, tag)); os.path.exists(RS) and os.remove(RS)""")
    return t


def ou_train(t, seedshift=False):
    """resume state <run>/resume.pth every 250 steps (model, optimizer, step, torch + loader generator states; dataset-internal numpy RNGs are
    not restored, so the data order after a resume differs from an uninterrupted run -> logged)."""
    t = rep(t, "    a = ap.parse_args()\n", "    a = ap.parse_args()\n    SEED = a.seed + %s\n" % ("int(os.environ.get('SX_SEEDSHIFT', '0'))" if seedshift else '0'))
    if seedshift:
        for x in ('common_utils.set_random_seed(a.seed)', 'xbg=xbg, seed=a.seed)', 'g_src.manual_seed(a.seed)', 'np.random.default_rng(1000 + a.seed)',
                  "np.load(sx.W + '/ou_bank.npy', mmap_mode='r'), a.seed)", 'g_t.manual_seed(2000 + a.seed)', 'L0, L1, a.seed)'):
            t = rep(t, x, x.replace('a.seed', 'SEED'))
        assert t.count('a.seed') == 3, t.count('a.seed')   # remaining: SEED definition, run-dir name, default init member (nominal index)
    t = rep(t, "logger.info('args %s' % vars(a))", "logger.info('args %s | effective RNG seed %d' % (vars(a), SEED))")
    t = rep(t, "    it_s, it_t = iter(L_src), (iter(L_tgt) if L_tgt else None)\n", """    RS = os.path.join(run, 'resume.pth'); start = 0
    if os.path.exists(RS):
        z = torch.load(RS, map_location='cpu', weights_only=False)
        model.load_state_dict(z['model']); optimizer.load_state_dict(z['opt']); start = z['step']; torch.set_rng_state(z['trng']); g_src.set_state(z['gsrc'])
        if L_tgt is not None and z.get('gt') is not None:
            g_t.set_state(z['gt'])
        logger.info('RESUME from step %d (%s); batch now %d+%d; data order after the resume differs from an uninterrupted run' % (start, RS, a.bs_src, a.bs_tgt))
    it_s, it_t = iter(L_src), (iter(L_tgt) if L_tgt else None)
""")
    t = rep(t, "    for step in range(steps):\n        sched.step(step)", "    for step in range(start, steps):\n        sched.step(step)")
    t = rep(t, "        acc['_n'] = acc.get('_n', 0) + 1\n", """        acc['_n'] = acc.get('_n', 0) + 1
        if (step + 1) % 250 == 0 and step + 1 < steps:
            torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(), step=step + 1, trng=torch.get_rng_state(), gsrc=g_src.get_state(),
                            gt=(g_t.get_state() if L_tgt is not None else None)), RS)
""")
    t = rep(t, "    logger.info('DONE %s' % J)", "    os.path.exists(RS) and os.remove(RS)\n    logger.info('DONE %s' % J)")
    return t


INFER_WRAP = '''# -*- coding: utf-8 -*-
"""[review1113 sup] run an inference script (argv[1], its own argv after it) as __main__ under a CUDA memory cap:
torch.cuda.set_per_process_memory_fraction($SX_INFER_MEMFRAC, default 0.2). No other change."""
import os, runpy, sys
import torch
fr = float(os.environ.get('SX_INFER_MEMFRAC', '0.2')); assert 0 < fr <= 0.2, fr
torch.cuda.set_per_process_memory_fraction(fr)
script = os.path.abspath(sys.argv[1]); sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
print('[infer_wrap] memory fraction %.2f -> %s' % (fr, ' '.join(sys.argv)), flush=True)
runpy.run_path(script, run_name='__main__')
'''


def infer_wrap(D):
    os.makedirs(D, exist_ok=True)
    open(D + '/infer_wrap.py', 'w', encoding='utf-8').write(INFER_WRAP)
