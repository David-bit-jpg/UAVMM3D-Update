# -*- coding: utf-8 -*-
"""[review1113 lidraw] teacher-level orientation of single orientation nets on the clean chain's MAV6D TEST crops.
  predict <read_te1.pkl> <ckpt key> <out.npz>   (NO labels) single-net rotation per test crop = ori_x.ensemble with that one member (the chain's
                                                symmetric-ensemble code, MODULES 0.2: read cell = heatmap peak in the 5x5 around the detection cell,
                                                corners from centre offset + kp2d, PnP with the crop K and the run's fixed size s_bar); PnP failure -> NaN
  score <arms.json> <out_prefix>                (reads the TEST labels, once) per net: mean symmetric AOE (pack scorer sym(): min over the 4
                                                quadrotor-equivalent rotations; PnP failure = 90 deg as the pack scorer) and median folded yaw error
                                                (yaw of R_gt^T R_pred about the body vertical, folded to [0, 45] deg by the 90-deg symmetry);
                                                per arm 3-seed mean +- sd; paired differences (b - a), (c - a) with a paired SEQUENCE bootstrap over the
                                                10 test sequences (B = 10000; frames pooled within resampled sequences, frame-weighted as the scorer).
Crop set = srcx/in2mav oc_te1 (one crop per test frame with a det1 detection; identical for every net -> paired)."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = HERE + '/eval/lib'
os.environ.setdefault('SX_SRC', 'in'); os.environ.setdefault('SX_TGT', 'mav'); os.environ.setdefault('SX_TEACH', 'full')
sys.path.insert(0, LIB)


def predict(read_pkl, key, out):
    import ori_x as OO
    import sx
    D = pickle.load(open('%s/oc_te1.pkl' % sx.W, 'rb')); info = D['info']; lwh = np.asarray(D['size'], np.float64)
    R = pickle.load(open(read_pkl, 'rb'))
    keys = [k for k in R['order'] if k == key or os.path.normcase(k) == os.path.normcase(key)]
    assert len(keys) == 1, (key, R['order'])
    Rs = np.full((len(info), 3, 3), np.nan)
    for i, f in enumerate(info):
        s = OO.ensemble(R['reads'], keys, i, R['cells'][i], f['K_crop'], lwh)
        if s is not None:
            Rs[i] = s[0]
    np.savez(out, R=Rs, seq=np.array([f['seq'] for f in info]), frame=np.array([f['frame'] for f in info]), ckpt=key)
    print('predict %s: %d crops, PnP failures %d -> %s' % (os.path.basename(key), len(info), int(np.isnan(Rs[:, 0, 0]).sum()), out), flush=True)


def score(arms_json, prefix):
    spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    A = json.load(open(arms_json, encoding='utf-8'))            # {arm: {seed: pred.npz}}
    M = np.load('E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz')   # TEST labels (read here only)
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    E, Y, seq0 = {}, {}, None
    for arm, seeds in A.items():
        for sd, fn in seeds.items():
            z = np.load(fn)
            seq = np.array([str(s) for s in z['seq']]); fr = np.array([str(f) for f in z['frame']])
            if seq0 is None:
                seq0, fr0 = seq, fr
            assert (seq == seq0).all() and (fr == fr0).all(), 'crop sets differ'
            e = np.zeros(len(seq)); y = np.zeros(len(seq))
            for i in range(len(seq)):
                g = M['gt'][key[(seq[i], fr[i])]]; Rg = Rot.from_euler('xyz', g[6:9]).as_matrix(); Rp = z['R'][i]
                if not np.isfinite(Rp).all():
                    e[i] = 90.0; y[i] = 45.0; continue
                e[i] = S.sym(Rp, Rg)
                Rr = Rg.T @ Rp; yw = np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0]))
                y[i] = abs(((yw + 45.0) % 90.0) - 45.0)
            E[(arm, sd)] = e; Y[(arm, sd)] = y
    useq = sorted(set(seq0)); idx = [np.where(seq0 == s)[0] for s in useq]
    rng = np.random.default_rng(0); NB = 10000
    draws = rng.integers(0, len(useq), (NB, len(useq)))
    res = dict(n_crops=int(len(seq0)), n_test_frames=int(len(M['seq'])), n_sequences=len(useq), per_net={}, per_arm={}, paired={})
    for (arm, sd), e in E.items():
        res['per_net']['%s_s%s' % (arm, sd)] = dict(AOE=float(e.mean()), median_yaw=float(np.median(Y[(arm, sd)])), pnp_fail=int((e == 90.0).sum()),
                                                    ckpt=str(np.load(A[arm][sd])['ckpt']))
    arms = list(A)
    seeds = sorted(set.intersection(*[set(A[a]) for a in arms]))
    for arm in arms:
        a_ = [res['per_net']['%s_s%s' % (arm, s)]['AOE'] for s in seeds]; m_ = [res['per_net']['%s_s%s' % (arm, s)]['median_yaw'] for s in seeds]
        res['per_arm'][arm] = dict(seeds=seeds, AOE_mean=float(np.mean(a_)), AOE_sd=float(np.std(a_, ddof=1)) if len(a_) > 1 else 0.0,
                                   medyaw_mean=float(np.mean(m_)), medyaw_sd=float(np.std(m_, ddof=1)) if len(m_) > 1 else 0.0)

    def boot_stats(arm):   # per resample: 3-seed mean of (frame-pooled AOE) and of (median folded yaw)
        ao = np.zeros(NB); my = np.zeros(NB)
        for s in seeds:
            e = E[(arm, s)]; y = Y[(arm, s)]
            se = np.array([e[ix].sum() for ix in idx]); n = np.array([len(ix) for ix in idx])
            ao += se[draws].sum(1) / n[draws].sum(1)
        ao /= len(seeds)
        for b in range(NB):
            sel = np.concatenate([idx[j] for j in draws[b]])
            my[b] = np.mean([np.median(Y[(arm, s)][sel]) for s in seeds])
        return ao, my
    ref = arms[0]
    BS = {arm: boot_stats(arm) for arm in arms}
    for arm in arms[1:]:
        dA = BS[arm][0] - BS[ref][0]; dY = BS[arm][1] - BS[ref][1]
        pt = res['per_arm'][arm]['AOE_mean'] - res['per_arm'][ref]['AOE_mean']; py = res['per_arm'][arm]['medyaw_mean'] - res['per_arm'][ref]['medyaw_mean']
        res['paired']['%s-%s' % (arm, ref)] = dict(
            dAOE=float(pt), dAOE_ci95=[float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))], dAOE_p_gt0=float((dA > 0).mean()),
            dmedyaw=float(py), dmedyaw_ci95=[float(np.percentile(dY, 2.5)), float(np.percentile(dY, 97.5))], dmedyaw_p_gt0=float((dY > 0).mean()),
            per_seed_dAOE={s: float(res['per_net']['%s_s%s' % (arm, s)]['AOE'] - res['per_net']['%s_s%s' % (ref, s)]['AOE']) for s in seeds},
            per_seed_dmedyaw={s: float(res['per_net']['%s_s%s' % (arm, s)]['median_yaw'] - res['per_net']['%s_s%s' % (ref, s)]['median_yaw']) for s in seeds})
    json.dump(res, open(prefix + '.json', 'w'), indent=1)
    L = ['# LiDAR body map vs raw points vs no LiDAR head - teacher-level orientation, MAV6D test (scored once)', '',
         '%d test crops (of %d test frames; frames without a det1 detection have no crop), %d sequences; AOE = mean symmetric rotation error (deg), '
         'yaw = median folded yaw error (deg); PnP failure = 90 deg.' % (res['n_crops'], res['n_test_frames'], res['n_sequences']), '',
         '| arm | seeds | AOE mean +- sd | median yaw mean +- sd | per seed AOE |', '|---|---|---|---|---|']
    for arm in arms:
        r = res['per_arm'][arm]
        L.append('| %s | %s | %.2f +- %.2f | %.2f +- %.2f | %s |' % (arm, ','.join(seeds), r['AOE_mean'], r['AOE_sd'], r['medyaw_mean'], r['medyaw_sd'],
                 ' / '.join('%.2f' % res['per_net']['%s_s%s' % (arm, s)]['AOE'] for s in seeds)))
    L += ['', '| paired (sequence bootstrap, B=10000) | dAOE [95% CI] P(>0) | d median yaw [95% CI] P(>0) |', '|---|---|---|']
    for k, p in res['paired'].items():
        L.append('| %s | %+.2f [%+.2f, %+.2f] %.3f | %+.2f [%+.2f, %+.2f] %.3f |' % (k, p['dAOE'], *p['dAOE_ci95'], p['dAOE_p_gt0'], p['dmedyaw'], *p['dmedyaw_ci95'], p['dmedyaw_p_gt0']))
    open(prefix + '.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'predict':
        predict(*sys.argv[2:5])
    elif c == 'score':
        score(*sys.argv[2:4])
