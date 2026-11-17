# R4: target 2D width (px) at each detector's input resolution: indoor sim train vs AntiUAV3D v2 val
import pickle, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('S', 'D:/UAV3D_pack_update_1112/scripts/score.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
Sm = pickle.load(open('D:/cmpfull/sim_train/labels.pkl', 'rb'))
ws, zs = [], []
for r in Sm[::5]:
    K = np.asarray(r['K'], float)
    for b in r['boxes']:
        bb = S.box2d(np.asarray(b['center'], float), np.asarray(b['lwh'], float), np.asarray(b['euler_xyz'], float), K)
        if bb is not None:
            ws.append((bb[2] - bb[0]) / r['W']); zs.append(b['center'][2])
ws = np.array(ws); V = pickle.load(open('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/val.pkl', 'rb'))
wa = np.array([r['w2d_px'] / 1920. for r in V])
out = {}
for nm, w in (('sim_indoor_train', ws), ('antiuav3d_v2_val', wa)):
    out[nm] = {('at_%d' % R): np.percentile(w * R, [5, 25, 50, 75, 95]).round(1).tolist() for R in (512, 960)}
    out[nm]['frac_below_16px_at_512'] = float((w * 512 < 16).mean())
out['sim_indoor_depth_pct'] = np.percentile(zs, [5, 50, 95]).round(2).tolist()
print(json.dumps(out, indent=1)); json.dump(out, open('r4_sizes.json', 'w'), indent=1)
