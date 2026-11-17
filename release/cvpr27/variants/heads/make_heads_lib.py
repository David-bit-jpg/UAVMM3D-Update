# -*- coding: utf-8 -*-
"""[review1113 heads] isolated lib copies of srcx/lib for ONE per-head ablation run.
usage: python make_heads_lib.py <arm> <tgt mav|antifs> <shift 0|10|20>      (idempotent; writes only review1113/heads/<arm>/<tgt>_s<shift>/lib)
Built on review1113/sup_patch.py (the same patches as the seed runs: SX_SEEDSHIFT on every training seed, env batch sizes for OOM halving,
resume states, sys.path -> copy). shift 0 = the nominal seeds of the reported run (SX_SEEDSHIFT=0 -> identical seed values).

Modes (the arm decides; run_arm.sh drives them):
  FULL  H1 H2 H3  refiner changed -> whole downstream chain recomputed (as review1113/seeds/run_seed_chain.sh, teach = full):
                  sx.O -> heads/<arm>/<tgt>_s<shift>, sx.W -> D:/S2MUAV3D/srcx_work/heads_<arm>_<C>_s<shift>,
                  REAL_CFG / ST_CFG / PSEUDO_TAG -> *_hd<arm>s<shift> (new names, nothing existing overwritten).
                  H1  refiner_x: SILW = 0 (IR-silhouette BCE+Dice on sim AND the stage-2/3 silhouette-consistency terms off); composites ON
                      (the IR silhouette is still the alpha matte, sim_batch unchanged). chain_x: one colour class (sx.ONECLASS) because the
                      colour class is the brightness inside the IR-SUPERVISED predicted silhouette, which no longer exists (= the rule the
                      SX_TEACH=none/lid controls already use, sx.NOIR). The distance head still receives sil_feats of the now-unsupervised
                      silhouette logits (they become a free learned feature path; H2 is the arm that removes that path).
                  H2  refiner_x Net.fwd: the 3 predicted-silhouette features (sil_feats) are replaced by zeros before the distance head
                      (= dropped: zero inputs give zero weight gradient); silhouette loss, composites, colour classes unchanged.
                  H3  refiner_x sim_batch: no silhouette composites (the rng draw that decides a composite is kept so the remaining random
                      stream stays as close as possible); silhouette loss, sil_feats, colour classes unchanged.
  ORI   H4 H5     orientation teachers changed -> teacher readings + students + deployment; W = new crop dir whose crop files are HARD LINKS
                  of the reference run's crop dir (read only; the only file written into W is the new pseudo_tr1.npz).
  HGT   H6        height source changed at deployment -> kpsbox_x.py (box-corner height ratio of the height network) + deploy.
  DEP   H1c       deploy-only diagnostic for H1: the reference run's files, deployment with one colour class (sx.ONECLASS). CPU only.
  BH    H7        box-hull silhouette control (review1113/boxhull) at seed shifts 10 / 20: refiner stages 1-3 (seed-shifted) + deploy at the
                  reference run's frozen w; PSEUDO_TAG = the reference run's tag (read only, used by chain_x targets); W = O/work.
Every copied file must point at this copy (asserted)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113')
import sup_patch as P  # noqa

# reference (reported / seed) runs: (run dir, crop dir, pseudo tag)
REF = {('mav', 0): (GP + '/srcx/in2mav', 'D:/S2MUAV3D/srcx_work/in2mav', 'srcx_in2mav'),
       ('mav', 10): (GP + '/review1113/sup3/run', 'D:/S2MUAV3D/srcx_work/in2mav_seed2', 'srcx_in2mav_seed2'),
       ('mav', 20): (GP + '/review1113/seeds/mav_full_s20', 'D:/S2MUAV3D/srcx_work/in2mav_sd20', 'srcx_in2mav_sd20'),
       ('antifs', 0): (GP + '/srcx/out2antifs', 'D:/S2MUAV3D/srcx_work/out2antifs', 'srcx_out2antifs'),
       ('antifs', 10): (GP + '/review1113/seeds/antifs_full_s10', 'D:/S2MUAV3D/srcx_work/out2antifs_sd10', 'srcx_out2antifs_sd10'),
       ('antifs', 20): (GP + '/review1113/seeds/antifs_full_s20', 'D:/S2MUAV3D/srcx_work/out2antifs_sd20', 'srcx_out2antifs_sd20')}
MODE = {'H1': 'FULL', 'H2': 'FULL', 'H3': 'FULL', 'H4': 'ORI', 'H5': 'ORI', 'H6': 'HGT', 'H1c': 'DEP', 'H7': 'BH'}
SRC = {'mav': 'in', 'antifs': 'out'}


def paths(arm, tgt, shift):
    C = '%s2%s' % (SRC[tgt], tgt)
    B = '%s/%s/%s_s%d' % (HERE, arm, tgt, shift)
    mode = MODE[arm]
    W = B + '/work' if mode in ('BH', 'DEP') else 'D:/S2MUAV3D/srcx_work/heads_%s_%s_s%d' % (arm, C, shift)
    return C, B, W, mode


if __name__ == '__main__':
    arm, tgt, shift = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert arm in MODE and tgt in SRC and shift in (0, 10, 20), sys.argv
    C, B, W, mode = paths(arm, tgt, shift)
    if mode not in ('FULL', 'ORI', 'HGT'):   # [11-17] ORI (H4/H5) and HGT (H6) ported to antifs
        assert tgt == 'mav', 'only the FULL / ORI / HGT arms are prepared for AntiUAV3D'
    REFDIR, REFW, REFTAG = REF[(tgt, shift)]
    D = B + '/lib'
    TAGC = '%s_hd%ss%d' % (C, arm, shift)

    t = P.rd('sx.py')
    t = P.rep(t, P.SX_OW, "O = '%s'          # [heads] %s %s shift %d (%s)\nW = '%s'\nO_REF = '%s'; W_REF = '%s'   # reference run, read only\n"
              % (B, arm, tgt, shift, mode, W, REFDIR, REFW))
    if mode == 'FULL':
        t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%%s_hd%ss%d.yaml' %% COMBO" % (arm, shift))
        t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%%s_hd%ss%d.yaml' %% COMBO" % (arm, shift))
        t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '_hd%ss%d'" % (arm, shift))
    elif mode == 'BH':
        t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = '%s'   # [heads H7] the reference run's Module-3 pseudo-labels, read only" % REFTAG)
    t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
              "assert (SRC, TGT, TEACH) == (%r, %r, 'full'), (SRC, TGT, TEACH)   # [heads] this copy is for %s %s s%d only\n"
              "HEADS_ARM = %r; ONECLASS = %r   # [heads] one colour class (H1 / H1c)\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)"
              % (SRC[tgt], tgt, arm, tgt, shift, arm, arm in ('H1', 'H1c')))
    t = P.rep(t, "for d in (O, W):\n    os.makedirs(d, exist_ok=True)", "for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):\n    os.makedirs(d, exist_ok=True)")
    P.wr(D, 'sx.py', t, '[heads %s %s s%d, mode %s] O -> %s, W -> %s' % (arm, tgt, shift, mode, B, W))

    # chain_x: one colour class for H1 / H1c (sx.ONECLASS False elsewhere -> code path identical to srcx/lib)
    t = P.path(P.rd('chain_x.py'), D)
    t = P.rep(t, "    if sx.NOIR:", "    if sx.NOIR or sx.ONECLASS:", 2)   # [heads] H1/H1c: no IR-supervised silhouette -> one colour class
    P.wr(D, 'chain_x.py', t, 'sys.path -> copy; one colour class if sx.ONECLASS (H1/H1c only)')
    P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
    for f in ('score_x.py', 'avg_inf.py', 'to_pred.py'):
        P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
    P.infer_wrap(D)

    if mode in ('FULL', 'BH'):
        t = P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=True)
        hdr = 'SX_SEEDSHIFT on training seeds; env batches; resume states'
        if arm == 'H1':
            t = P.rep(t, "SILW = 0.0 if sx.NOIR else 1.0", "SILW = 0.0   # [heads H1] IR-silhouette losses OFF (all stages); composites stay ON")
            hdr += '; [H1] SILW = 0'
        elif arm == 'H2':
            t = P.rep(t, "        z = torch.cat([feat, self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype)], 1)",
                      "        z = torch.cat([feat, self.sc((s - self.s_mean) / self.s_std), torch.zeros(x.shape[0], 3, device=x.device, dtype=c4.dtype)], 1)   # [heads H2] sil_feats dropped")
            hdr += '; [H2] predicted-silhouette features NOT fed to the distance head (zeros)'
        elif arm == 'H3':
            t = P.rep(t, "        if not sx.NOIR and rng.random() < 0.5 and sil[idx[n]].max() > 128:   # [11-12] no IR: no silhouette composites",
                      "        if not sx.NOIR and rng.random() < 0.5 and sil[idx[n]].max() > 128 and False:   # [heads H3] silhouette composites OFF (draw kept)")
            hdr += '; [H3] no silhouette composites'
        P.wr(D, 'refiner_x.py', t, hdr)
    if mode == 'FULL':
        P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged')
        P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
        for f in ('make_pseudo_x.py', 'mkcfg.py', 'wsim_x.py'):
            P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
    if mode == 'ORI':
        P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged')
    if mode == 'HGT':
        t = open(HERE + '/kpsbox_x.py', encoding='utf-8').read()
        open(D + '/kpsbox_x.py', 'w', encoding='utf-8').write(t)
    for f in os.listdir(D):
        if f.endswith('.py'):
            s = open(D + '/' + f, encoding='utf-8').read()
            assert 'gproto/srcx/lib' not in s.split('\n', 1)[1], f
    print('ok', arm, tgt, shift, mode, D)
