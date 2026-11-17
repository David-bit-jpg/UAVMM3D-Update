# -*- coding: utf-8 -*-
"""[review1113 B3] copies of srcx/lib for route_commands.sh -> review1113/B3/lib (nothing else is written).
Same transformations as sup2/make_nogate_lib.py (sx.O/W -> $CL_AO, crop arrays read in place from the deployed work dir via _wf(), env batch
sizes ORI_BS / REF_BS / REF_TRAIN_BS with the deployed defaults, resume states, infer_wrap.py with memory fraction 0.2), plus ONE addition to
ou_train_x.py: the teacher-knowledge transfer ROUTE at the unchanged insertion point (orientation students on real train crops).

  --route hard (default)  the deployed recipe, bit-for-bit (this arm is NOT retrained: it is srcx/in2mav, test 60.81 / val 59.71)
  --route feat            no hard pseudo label enters any loss. Per real crop (same gated 80 % bag, same augmentation) the 6 frozen teachers
                          (srcx/in2mav/ou/T_{irlid,kdir,lid}{0,1}.pth = the same AdaBN'd teacher states that produced the pseudo-orientations)
                          run on the SAME augmented image; targets = their mean.
                            read cell (r*, c*) = argmax of the teachers' mean sigmoid heatmap within the 5x5 cells around the crop's
                                     detection cell (= the deployment read rule of ori_x read / ensemble; detection centre from oc_tr1.pkl,
                                     mirrored when the crop is h-flipped). A global argmax was tried first and produced single-batch
                                     outliers (L_kp 15 vs 0.1) from off-object peaks;
                            L_feat = mean_t mean over the 5x5 cells around (r*, c*) of 1 - cos(A_t f_s, f_t), A_t = a 1x1 conv adapter per
                                     teacher (identity init, Adam 1e-3, trained jointly, discarded after training; --feat-target adapt, default)
                                     because the 6 independently trained teachers' features are near-orthogonal (pairwise cos 0.09-0.15 on
                                     the read window, kdir0/kdir1 0.60), so their plain mean is no coherent target; --feat-target mean = the
                                     literal 1 - cos(f_s, mean_t f_t),
                                     f = 'features_2d' = the shared input of every CenterHead branch, i.e. exactly the input of the kp2d head
                                     (B, C, 36, 64 for a 288x512 crop); 5x5 = the receptive field of the kp2d head's two 3x3 convs at (r*, c*);
                            L_hm   = sum_cells KL_Bernoulli(mean_t sigmoid(h_t) || sigmoid(h_s)) / sum_cells mean_t sigmoid(h_t);
                            L_kp   = mean_t min_{q=0..3} || k_s(r*,c*) - Pi_q k_t(r*,c*) ||_1 / 16   (Pi_q = corner renumbering of R Rz(90q),
                                     the same quadrotor symmetry as the hard L_pl^ori kp term);
                            L_soft = L_hm + 12 L_kp                                (the hard recipe's hm : kp2d weights 1 : 12; no centre-offset term)
                            loss_t = lam_t (w_feat L_feat + w_soft L_soft) + lam_t * 12 * L_mir, mirror cell = (r*, c*)
  --route both            hard recipe unchanged (L_pl^ori + mirror loss on the pseudo-label cell) + lam_t w_feat L_feat
  --calib-out <json>      no training: on --calib-n real batches of the seed-0 target loader, at initialisation (student = IR+LiDAR member,
                          train-mode BN as in a training step, no gradient) record mean L_pl^ori (the hard target loss: hm + 2 centre_res
                          + 12 kp2d), L_feat, L_hm, L_kp, L_soft, L_mir and the teachers' pairwise feature cosine; writes
                          w_feat = L_pl / L_feat and w_soft = L_pl / L_soft (each KD term has the magnitude of L_pl^ori at init). The weights
                          are then FIXED for all arms / seeds (route_commands.sh reads them once); nothing is tuned on val or test.
Teacher list: env OU_TEACHERS (comma-separated, 6 checkpoints). Run: python make_route_lib.py   (idempotent)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sup_patch as P  # noqa

D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/lib'
os.makedirs(D, exist_ok=True)

t = P.rd('sx.py')
P.wr(D, 'sx.py', P.rep(t, P.SX_OW, "O = os.environ['CL_AO']\nW = O + '/work'\nW_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO\nO_ORIG = HOME + '/' + COMBO\n"),
     'O/W redirected to $CL_AO (writes only under review1113/B3/<arm>)')

# ---------------------------------------------------------------- ou_train_x: resume (sup_patch) + transfer route
t = P.ou_train(P.path(P.rd('ou_train_x.py'), D))
ROUTE_FUNCS = r'''
# ======================================================================= [review1113 B3] teacher-knowledge transfer route
SIG_C4 = [[(i + q) % 4 + 4 * (i // 4) for i in range(8)] for q in range(4)]          # corner i of R Rz(90q) = corner SIG[q][i] of R (l = w)
KP_PERM = [[2 * SIG_C4[q][i] + c for i in range(8) for c in (0, 1)] for q in range(4)]  # kp2d channel = 2 * corner + (u, v), as unmirror_kp


def load_teachers(build_network, cfg, dataset, logger):
    paths = [p for p in os.environ.get('OU_TEACHERS', '').split(',') if p]
    assert len(paths) == 6, ('OU_TEACHERS must list the 6 frozen teachers', paths)
    T = []
    for p in paths:
        m = build_network(model_cfg=cfg.MODEL, dataset=dataset)
        a_, b_ = m.load_params_from_file(filename=p, to_cpu=False, logger=logger)
        assert a_ == b_, (p, a_, b_)
        m.cuda().eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        T.append(m)
    logger.info('[B3] %d frozen teachers (eval mode, their AdaBN statistics): %s' % (len(T), paths))
    return T


def teacher_forward(T, image):
    """same input as the student -> per-teacher features_2d (B, C, h, w), sigmoid heatmaps (B, 1, h, w), kp2d (B, 16, h, w)"""
    Fs, Hs, Ks = [], [], []
    with torch.no_grad():
        for m in T:
            tb = {'image': image, 'batch_size': image.shape[0]}
            for mod in m.module_list:
                tb = mod(tb)
                if mod is m.dense_head_2d:
                    break
            f = tb['features_2d']
            Fs.append(f.reshape(-1, *f.shape[-3:]).float())
            pc = tb['pred_center_dict']
            Hs.append(torch.sigmoid(pc['hm'].float()))
            Ks.append(pc['kp2d'].float())
    return Fs, Hs, Ks


CROP_UV = None   # [B3] (N, 2) detection centre (uc, vc) of every train crop in crop pixels (oc_tr1.pkl info), set in main


def read_cell(Hbar, dc):
    """= ori_x read / ensemble rule: argmax of the member-averaged heatmap within the 5x5 cells around the crop's detection cell dc (B, 2: row, col)"""
    B, _, h, w = Hbar.shape
    rr = torch.arange(h, device=Hbar.device)[None, :, None]; cc = torch.arange(w, device=Hbar.device)[None, None, :]
    m = ((rr - dc[:, 0, None, None]).abs() <= 2) & ((cc - dc[:, 1, None, None]).abs() <= 2)
    ix = Hbar[:, 0].masked_fill(~m, -1.0).reshape(B, -1).argmax(1)
    return ix // w, ix % w


def route_terms(fs, hs_logit, ks, Fs, Hs, Ks, ADP, dc):
    """student features fs (B, C, h, w), heatmap logits (B, 1, h, w), kp2d (B, 16, h, w) vs the teachers' -> L_feat, L_hm, L_kp, (r*, c*)
    ADP = per-teacher 1x1 adapters (feat target 'adapt'): L_feat = mean_t (1 - cos(A_t f_s, f_t)); None ('mean'): 1 - cos(f_s, mean_t f_t)
    dc = detection cell of each crop (B, 2); read cell (r*, c*) = deployment rule read_cell"""
    B, C, h, w = fs.shape
    Fbar = torch.stack(Fs).mean(0); Hbar = torch.stack(Hs).mean(0)
    r, c = read_cell(Hbar, dc)
    rr = torch.arange(h, device=fs.device)[None, :, None]; cc = torch.arange(w, device=fs.device)[None, None, :]
    win = ((rr - r[:, None, None]).abs() <= 2) & ((cc - c[:, None, None]).abs() <= 2)            # (B, h, w) 5x5 around the read cell
    if ADP is None:
        cos = torch.nn.functional.cosine_similarity(fs.float(), Fbar, dim=1, eps=1e-6)             # (B, h, w)
        l_feat = (1.0 - cos)[win].mean()
    else:
        l_feat = torch.stack([(1.0 - torch.nn.functional.cosine_similarity(ad(fs.float()), ft, dim=1, eps=1e-6))[win].mean()
                              for ad, ft in zip(ADP, Fs)]).mean()
    p = torch.sigmoid(hs_logit.float()).clamp(1e-4, 1 - 1e-4); q = Hbar.clamp(1e-4, 1 - 1e-4)
    kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    l_hm = kl.sum() / Hbar.sum().clamp(min=1.0)
    bi = torch.arange(B, device=fs.device)
    k_s = ks.float()[bi, :, r, c]                                                                  # (B, 16)
    lk = []
    for kt in Ks:
        k_t = kt[bi, :, r, c]
        lk.append(torch.stack([(k_s - k_t[:, KP_PERM[q_]]).abs().sum(1) for q_ in range(4)], 0).min(0).values / 16)
    l_kp = torch.stack(lk, 0).mean()
    return l_feat, l_hm, l_kp, (r, c)


def teacher_cos_matrix(Fs, Hs, dc):
    """diagnostic: pairwise cosine of the teachers' features on the 5x5 read window (are their feature spaces comparable?)"""
    B, C, h, w = Fs[0].shape
    Hbar = torch.stack(Hs).mean(0); r, c = read_cell(Hbar, dc)
    rr = torch.arange(h, device=Fs[0].device)[None, :, None]; cc = torch.arange(w, device=Fs[0].device)[None, None, :]
    win = ((rr - r[:, None, None]).abs() <= 2) & ((cc - c[:, None, None]).abs() <= 2)
    n = len(Fs); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = float(torch.nn.functional.cosine_similarity(Fs[i], Fs[j], dim=1)[win].mean())
    return M


def main():'''
t = P.rep(t, "\ndef main():", ROUTE_FUNCS)
# [B3] real-crop dataset: hflip bookkeeping + the crop's detection cell (deployment read window), emitted only when CROP_UV is set (feat/both/calib)
t = P.rep(t, "        image, boxes, names, K = self._augment(image, boxes, names, K)\n        assert len(boxes) == 4\n",
          "        tx0 = float(boxes[0, 0])\n        image, boxes, names, K = self._augment(image, boxes, names, K)\n        assert len(boxes) == 4\n"
          "        flipped = tx0 != 0.0 and abs(float(boxes[0, 0]) + tx0) < 1e-9   # [B3] hflip bookkeeping only (_augment negates t_x)\n")
t = P.rep(t, "        dd['pw'] = np.array([float(self.P['w'][n])], np.float32)\n        return dd\n",
          "        dd['pw'] = np.array([float(self.P['w'][n])], np.float32)\n"
          "        if CROP_UV is not None:   # [B3] detection cell of this crop, mirrored with the image\n"
          "            uc, vc = CROP_UV[ci]\n"
          "            uc = (self.W - 1 - uc) if flipped else uc\n"
          "            dd['dcell'] = np.array([min(max(int(vc // self.stride), 0), self.H // self.stride - 1),\n"
          "                                    min(max(int(uc // self.stride), 0), self.W // self.stride - 1)], np.int64)\n"
          "        return dd\n")
t = P.rep(t, "            pw = torch.from_numpy(np.asarray(b_t.pop('pw'), np.float32).reshape(-1)).to(dev)\n",
          "            pw = torch.from_numpy(np.asarray(b_t.pop('pw'), np.float32).reshape(-1)).to(dev)\n"
          "            dc = torch.as_tensor(np.asarray(b_t.pop('dcell')).reshape(-1, 2), device=dev) if 'dcell' in b_t else None   # [B3]\n")
t = P.rep(t, "    ap.add_argument('--max-steps-smoke', type=int, default=0)\n", """    ap.add_argument('--max-steps-smoke', type=int, default=0)
    ap.add_argument('--route', default='hard', choices=['hard', 'feat', 'both'], help='[B3] how teacher knowledge reaches the student')
    ap.add_argument('--w-feat', type=float, default=0.0, help='[B3] weight of L_feat (feat, both); from the init calibration')
    ap.add_argument('--w-soft', type=float, default=0.0, help='[B3] weight of L_soft = L_hm + 12 L_kp (feat); from the init calibration')
    ap.add_argument('--calib-out', default='', help='[B3] write the init calibration json and exit (no training)')
    ap.add_argument('--calib-n', type=int, default=50)
    ap.add_argument('--feat-target', default='adapt', choices=['adapt', 'mean'], help='[B3] adapt = per-teacher 1x1 adapters (identity init); mean = literal mean feature')
    ap.add_argument('--adapt-lr', type=float, default=1e-3, help='[B3] Adam LR of the adapters (fixed a priori)')
""")
t = P.rep(t, "    src_lw = dict(head.loss_weights)\n", """    src_lw = dict(head.loss_weights)
    ROUTE = a.route
    TEACH = load_teachers(build_network, cfg, src, logger) if (ROUTE != 'hard' or a.calib_out) else []
    global CROP_UV
    if TEACH:   # detection centre of every train crop (crop pixels): the deployment read window for the teachers' heatmap peak
        import pickle
        _I = pickle.load(open(sx.W + '/oc_tr1.pkl', 'rb'))['info']
        CROP_UV = np.array([[float(r_['uc']), float(r_['vc'])] for r_ in _I], np.float64)
        assert len(CROP_UV) == len(crops), (len(CROP_UV), len(crops))
    if ROUTE == 'feat':
        assert a.w_feat > 0 and a.w_soft > 0 and a.mir > 0 and a.tgt != 'none'
    if ROUTE == 'both':
        assert a.w_feat > 0 and a.tgt != 'none'
    ADP = ADPO = None
    if TEACH and a.feat_target == 'adapt':
        C_ = int(cfg.MODEL.DENSE_HEAD_2D.INPUT_CHANNELS)
        ADP = torch.nn.ModuleList([torch.nn.Conv2d(C_, C_, 1, bias=False) for _ in TEACH]).cuda()
        with torch.no_grad():
            for cv in ADP:
                cv.weight.copy_(torch.eye(C_, device=cv.weight.device).reshape(C_, C_, 1, 1))
        ADPO = torch.optim.Adam(ADP.parameters(), lr=a.adapt_lr)
    logger.info('[B3] route %s | feat target %s (adapters %s, LR %.1e) | w_feat %.6g | w_soft %.6g | mirror %.1f' % (
        ROUTE, a.feat_target, 'none' if ADP is None else '%d x 1x1 %d->%d identity init' % (len(ADP), C_, C_), a.adapt_lr, a.w_feat, a.w_soft, a.mir))
""")
# calibration (after the resume block, before the training loop)
CALIB = """    if a.calib_out:
        assert L_tgt is not None
        model.train(); head.loss_weights.clear(); head.loss_weights.update(TGT_LW)
        R_ = {k: [] for k in ('L_pl', 'L_feat', 'L_hm', 'L_kp', 'L_soft', 'L_mir')}; CM = []
        it_c = iter(L_tgt)
        with torch.no_grad():
            for _ in range(a.calib_n):
                b_t, it_c = nxt(it_c, L_tgt)
                b_t.pop('pw'); dc = torch.as_tensor(np.asarray(b_t.pop('dcell')).reshape(-1, 2), device=dev); KP.to_device(b_t, dev, torch)
                ret_t = model(b_t)
                pcs = head.forward_loss_dict['pred_center_dict']
                fs = b_t['features_2d']; fs = fs.reshape(-1, *fs.shape[-3:])
                Fs, Hs, Ks = teacher_forward(TEACH, b_t['image'])
                l_feat, l_hm, l_kp, (pr, pc_) = route_terms(fs, pcs['hm'], pcs['kp2d'], Fs, Hs, Ks, ADP, dc)
                p1 = pcs['kp2d']; Bm = p1.shape[0]
                tb = dict(b_t); tb['image'] = torch.flip(b_t['image'], dims=[-1])
                for mod in model.module_list:
                    tb = mod(tb)
                p2 = unmirror_kp(head.forward_loss_dict['pred_center_dict']['kp2d'])
                bi = torch.arange(Bm, device=p1.device)
                R_['L_mir'].append(float((p1 - p2)[bi, :, pr, pc_].abs().sum(1).mean() / 16))
                R_['L_pl'].append(float(ret_t['loss'])); R_['L_feat'].append(float(l_feat)); R_['L_hm'].append(float(l_hm))
                R_['L_kp'].append(float(l_kp)); R_['L_soft'].append(float(l_hm + TGT_LW['kp2d'] * l_kp))
                CM.append(teacher_cos_matrix(Fs, Hs, dc))
        J = {k: float(np.mean(v)) for k, v in R_.items()}
        J['w_feat'] = J['L_pl'] / max(J['L_feat'], 1e-8); J['w_soft'] = J['L_pl'] / max(J['L_soft'], 1e-8)
        J.update(feat_target=a.feat_target, n_batches=a.calib_n, bs_tgt=a.bs_tgt, seed=a.seed, init=init, teachers=os.environ.get('OU_TEACHERS', '').split(','),
                 rule='w_feat = mean L_pl^ori / mean L_feat; w_soft = mean L_pl^ori / mean L_soft; at init, seed-0 target loader, no grad',
                 teacher_feature_cos=np.mean(CM, 0).round(4).tolist(), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(J, open(a.calib_out, 'w', encoding='utf-8'), indent=1)
        logger.info('[B3] calibration %s' % J)
        return
"""
t = P.rep(t, "    t0 = time.time()\n    acc = {}\n    model.train()\n", CALIB + "    t0 = time.time()\n    acc = {}\n    model.train()\n")
# the target step
OLD = """            ret_t = model(b_t)
            head._sample_w = None
            loss = loss + a.lam_t * ret_t['loss']
            if a.mir > 0:
                p1 = head.forward_loss_dict['pred_center_dict']['kp2d']
                gk = b_t['kp2d']
                Bm, Km = gk.shape[0], gk.shape[1]
                mm = gk[:, :, :16].abs().sum(dim=2) > 0
"""
NEW = """            if ROUTE == 'feat':      # [B3] no hard pseudo label enters any loss (the head's train-mode gt bookkeeping is never read)
                for mod in model.module_list:
                    b_t = mod(b_t)
            else:
                ret_t = model(b_t)
                loss = loss + a.lam_t * ret_t['loss']
            head._sample_w = None
            if ROUTE != 'hard':
                pcs = head.forward_loss_dict['pred_center_dict']
                fs = b_t['features_2d']; fs = fs.reshape(-1, *fs.shape[-3:])
                Fs, Hs, Ks = teacher_forward(TEACH, b_t['image'])
                l_feat, l_hm, l_kp, (pr, pc_) = route_terms(fs, pcs['hm'], pcs['kp2d'], Fs, Hs, Ks, ADP, dc)
                loss = loss + a.lam_t * a.w_feat * l_feat
                terms['kd_feat'] = float(l_feat.detach()); terms['kd_hm'] = float(l_hm.detach()); terms['kd_kp'] = float(l_kp.detach())
                if ROUTE == 'feat':
                    l_soft = l_hm + TGT_LW['kp2d'] * l_kp
                    loss = loss + a.lam_t * a.w_soft * l_soft
                    terms['kd_soft'] = float(l_soft.detach())
            if a.mir > 0:
                p1 = head.forward_loss_dict['pred_center_dict']['kp2d']
                gk = b_t['kp2d']
                Bm, Km = gk.shape[0], gk.shape[1]
                mm = gk[:, :, :16].abs().sum(dim=2) > 0
                if ROUTE == 'feat':  # [B3] mirror-consistency cell = the teachers' heatmap peak, not the pseudo-label centre
                    mm = torch.zeros_like(mm); mm[torch.arange(Bm, device=mm.device), 0, pr, pc_] = True
"""
t = P.rep(t, OLD, NEW)
t = P.rep(t, "            terms.update({'tgt_' + k: v for k, v in (head.loss_terms or {}).items()})\n",
          "            if ROUTE != 'feat':      # [B3] in the feat route the head loss is never evaluated on target crops\n"
          "                terms.update({'tgt_' + k: v for k, v in (head.loss_terms or {}).items()})\n")
t = P.rep(t, "n_tgt=(len(L_tgt.dataset) if L_tgt else 0), written_at=",
          "n_tgt=(len(L_tgt.dataset) if L_tgt else 0), route=ROUTE, max_mem_gb=torch.cuda.max_memory_allocated() / 1e9, written_at=")
t = P.rep(t, "        optimizer.zero_grad()\n        loss.backward()\n", "        optimizer.zero_grad()\n        if ADPO is not None:\n            ADPO.zero_grad()\n        loss.backward()\n")
t = P.rep(t, "        optimizer.step()\n", "        optimizer.step()\n        if ADPO is not None:\n            ADPO.step()\n")
t = P.rep(t, "gt=(g_t.get_state() if L_tgt is not None else None)), RS)",
          "gt=(g_t.get_state() if L_tgt is not None else None), adp=(ADP.state_dict() if ADP is not None else None), adpo=(ADPO.state_dict() if ADPO is not None else None)), RS)")
t = P.rep(t, "            g_t.set_state(z['gt'])\n", "            g_t.set_state(z['gt'])\n        if ADP is not None and z.get('adp') is not None:\n            ADP.load_state_dict(z['adp']); ADPO.load_state_dict(z['adpo'])\n")
t = P.rep(t, "    os.path.exists(RS) and os.remove(RS)\n    logger.info('DONE %s' % J)",
          "    if ADP is not None:\n        torch.save(ADP.state_dict(), os.path.join(run, 'adapters.pth'))   # [B3] audit only; not used at deployment\n    os.path.exists(RS) and os.remove(RS)\n    logger.info('DONE %s' % J)")
P.wr(D, 'ou_train_x.py', t, 'sx = redirected copy; resume state every 250 steps; [B3] --route hard|feat|both + --calib-out (hard = recipe unchanged)')

# ---------------------------------------------------------------- the rest = sup2/make_nogate_lib.py
t = P.path(P.rd('ori_x.py'), D)
a = "ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W\n"
t = P.rep(t, a, a + "\n\ndef _wf(p):\n    return p if os.path.exists(p) or not p.startswith(sx.W + '/') else sx.W_ORIG + p[len(sx.W):]\n")
for pat in ["np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, name), 'rb')", "np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r')",
            "open('%s/oc_%s.pkl' % (W, names[0]), 'rb')", "np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, nm_), 'rb')"]:
    inner = re.search(r"'%s/oc_%s\.(npy|pkl)' % \(W, \w+(\[0\])?\)", pat).group(0)
    t = t.replace(pat, pat.replace(inner, '_wf(%s)' % inner))
P.wr(D, 'ori_x.py', P.ori_bs(t), 'sx = redirected copy; _wf crop fallback to deployed W (read only); batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D)), 'sx = redirected copy; env batches (deployed defaults); resume states')
P.wr(D, 'chain_x.py', P.path(P.rd('chain_x.py'), D), 'sx = redirected copy (reads/writes under $CL_AO)')
P.wr(D, 'score_x.py', P.rd('score_x.py'), 'unchanged (sx imported from this dir)')
P.infer_wrap(D)
for f in os.listdir(D):
    if f.endswith('.py'):
        assert 'gproto/srcx/lib' not in open(D + '/' + f, encoding='utf-8').read().split('\n', 1)[1], f
print('ok', D)
