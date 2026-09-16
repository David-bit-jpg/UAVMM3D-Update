# -*- coding: utf-8 -*-
"""Offline analysis of the UE dump: put every drone's 'Drone Body' vertices into the actor frame
(actor spawned at identity, so component world transform == actor-relative), then compare with the
BoundingCheck box, find the arm directions (angular histogram of far vertices) and the nose direction
(landing-gear axis / camera protrusion), and print ASCII top / side occupancy maps.

Input : E:/Open3DUAVDet/output/camnorm/annot_audit/sim_assets/drone_bp_dump.json + verts/*.f32
Output: stdout + E:/Open3DUAVDet/output/camnorm/annot_audit/sim_assets/verts_analysis.json
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/sim_assets'
d = json.load(open(os.path.join(OUT, 'drone_bp_dump.json'), encoding='utf-8'))


def ue_rotmat(rot):
    """UE FRotationMatrix (roll,pitch,yaw in deg) -> 3x3 matrix in column convention (v' = Rm @ v).
    UE row-vector form: M[0]=(CP*CY, CP*SY, SP), M[1]=(SR*SP*CY-CR*SY, SR*SP*SY+CR*CY, -SR*CP), M[2]=(-(CR*SP*CY+SR*SY), CY*SR-CR*SP*SY, CR*CP)."""
    SR, CR = np.sin(np.radians(rot['roll'])), np.cos(np.radians(rot['roll']))
    SP, CP = np.sin(np.radians(rot['pitch'])), np.cos(np.radians(rot['pitch']))
    SY, CY = np.sin(np.radians(rot['yaw'])), np.cos(np.radians(rot['yaw']))
    M = np.array([[CP * CY, CP * SY, SP],
                  [SR * SP * CY - CR * SY, SR * SP * SY + CR * CY, -SR * CP],
                  [-(CR * SP * CY + SR * SY), CY * SR - CR * SP * SY, CR * CP]])
    return M.T


def ue_xform(v_local, comp):
    """UE component world transform (from world_location / world_rotation / world_scale of the dump) applied to
    mesh-local points (cm). UE FTransform: scale, then rotate, then translate."""
    t = np.array(comp['world_location'])
    s = np.array(comp['world_scale'])
    Rm = ue_rotmat(comp['world_rotation'])
    return (Rm @ (v_local * s).T).T + t


def ascii_map(x, y, nx=48, ny=24, title='', xlab='x', ylab='y'):
    """Occupancy map of points; rows = y descending (so +y is up on screen), cols = x."""
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    H, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])
    H = H.T[::-1]
    thr = np.percentile(H[H > 0], [30, 70]) if (H > 0).any() else [1, 2]
    lines = ['   %s  (%s: %.3f..%.3f  %s: %.3f..%.3f ; +%s right, +%s up)' % (title, xlab, xmin, xmax, ylab, ymin, ymax, xlab, ylab)]
    for row in H:
        lines.append('   |' + ''.join(' ' if v == 0 else ('.' if v <= thr[0] else ('+' if v <= thr[1] else '#')) for v in row) + '|')
    return '\n'.join(lines)


summary = {}
for m, rec in d['models'].items():
    comps = {c['name']: c for c in rec['components']}
    body = comps['Drone Body']
    box = comps['BoundingCheck']
    vf = body['verts']['file']
    v = np.fromfile(vf, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    pa = ue_xform(v, body) / 100.0     # actor frame, meters (UE: x fwd, y right, z up)
    # sanity: RotorR1 capsule world location from UE vs our transform of its relative location
    chk = None
    if 'RotorR1' in comps and comps['RotorR1']['parent'] == 'Drone Body':
        rl = np.array(comps['RotorR1']['relative_location'])
        ours = ue_xform(rl[None], body)[0]
        chk = {'ue': comps['RotorR1']['world_location'], 'ours': ours.tolist()}
    bc = np.array(box['world_location']) / 100.0
    be = np.array(box['box_scaled_extent']) / 100.0          # half sizes, m
    byaw = box['world_rotation']['yaw']
    lo, hi = pa.min(0), pa.max(0)
    p_lo, p_hi = np.percentile(pa, 0.2, axis=0), np.percentile(pa, 99.8, axis=0)
    inside = np.all(np.abs(pa - bc) <= be + 1e-9, axis=1).mean()
    # per-axis fraction outside box
    out_axis = [(np.abs(pa[:, i] - bc[i]) > be[i]).mean() for i in range(3)]
    # arm directions: far vertices in XY (r > 0.7 max r), angular histogram 10 deg bins
    cxy = (p_lo[:2] + p_hi[:2]) / 2
    rel = pa[:, :2] - cxy
    r = np.linalg.norm(rel, axis=1)
    far = r > 0.7 * np.percentile(r, 99.5)
    ang = np.degrees(np.arctan2(rel[far, 1], rel[far, 0]))
    hist, edges = np.histogram(ang, bins=36, range=(-180, 180))
    top_bins = np.argsort(hist)[::-1][:8]
    arms = sorted([(float(edges[i] + 5), int(hist[i])) for i in top_bins if hist[i] > 0.02 * far.sum()])
    # rotated-45 AABB
    c45, s45 = np.cos(np.radians(45)), np.sin(np.radians(45))
    xy45 = rel @ np.array([[c45, -s45], [s45, c45]])
    span45 = np.percentile(xy45, 99.8, axis=0) - np.percentile(xy45, 0.2, axis=0)
    # z bands
    zr = p_hi[2] - p_lo[2]
    low = pa[:, 2] < p_lo[2] + 0.12 * zr
    mid = (pa[:, 2] >= p_lo[2] + 0.12 * zr) & (pa[:, 2] < p_lo[2] + 0.45 * zr)
    top = pa[:, 2] > p_lo[2] + 0.75 * zr
    def band_stats(mask):
        if mask.sum() < 10:
            return None
        q = pa[mask, :2]
        cen = q.mean(0)
        cov = np.cov((q - cen).T)
        w, vec = np.linalg.eigh(cov)
        main = vec[:, np.argmax(w)]
        return {'n': int(mask.sum()), 'centroid_xy': cen.tolist(), 'centroid_minus_boxcenter_xy': (cen - cxy).tolist(),
                'pca_main_axis_deg': float(np.degrees(np.arctan2(main[1], main[0]))), 'pca_ratio': float(np.sqrt(w.max() / max(w.min(), 1e-12)))}
    # all visible static meshes (template leftovers r1..r4 / pCamera included) -> union AABB + box fit
    allv = [pa]
    extra = []
    for cn, c in comps.items():
        if cn in ('Drone Body', 'PhysicObject') or not c.get('verts') or not c['verts'].get('file') or c.get('visible') is not True:
            continue
        vv = np.fromfile(c['verts']['file'], dtype=np.float32).reshape(-1, 3).astype(np.float64)
        pv = ue_xform(vv, c) / 100.0
        allv.append(pv)
        extra.append({'comp': cn, 'mesh': c.get('static_mesh', '').split('.')[-1], 'n': len(pv), 'aabb_min': pv.min(0).tolist(), 'aabb_max': pv.max(0).tolist(),
                      'size': (pv.max(0) - pv.min(0)).tolist(), 'center': ((pv.max(0) + pv.min(0)) / 2).tolist(),
                      'frac_inside_box': float(np.all(np.abs(pv - bc) <= be, axis=1).mean())})
    allp = np.concatenate(allv, 0)
    res = {
        'drone_body_mesh': body.get('static_mesh'), 'drone_body_rel_rot': body['relative_rotation'], 'drone_body_rel_scale': body['relative_scale3d'],
        'box_rel_rot': box['relative_rotation'], 'box_rel_loc': box['relative_location'], 'box_rel_scale': box['relative_scale3d'], 'box_parent': box['parent'],
        'extra_visible_meshes': extra,
        'all_visible_aabb_size_m': (allp.max(0) - allp.min(0)).tolist(), 'all_visible_aabb_min': allp.min(0).tolist(), 'all_visible_aabb_max': allp.max(0).tolist(),
        'rotor_check': chk,
        'body_world_scale': body['world_scale'], 'body_world_rot': body['world_rotation'],
        'mesh_aabb_actor_m': {'min': lo.tolist(), 'max': hi.tolist(), 'size': (hi - lo).tolist()},
        'mesh_aabb_robust_m': {'min': p_lo.tolist(), 'max': p_hi.tolist(), 'size': (p_hi - p_lo).tolist()},
        'mesh_span_rot45_xy_m': span45.tolist(),
        'box_center_m': bc.tolist(), 'box_size_m': (2 * be).tolist(), 'box_yaw_deg': byaw,
        'frac_verts_inside_box': float(inside), 'frac_outside_per_axis': out_axis,
        'centroid_actor_m': pa.mean(0).tolist(),
        'arm_angle_peaks_deg(count)': arms,
        'band_low': band_stats(low), 'band_mid': band_stats(mid), 'band_top': band_stats(top),
    }
    summary[m] = res
    print('=' * 100)
    print(m, '| body world scale', np.round(body['world_scale'], 4), 'body world rot', body['world_rotation'])
    print('  Drone Body mesh', res['drone_body_mesh'].split('/')[-1], 'relRot', body['relative_rotation'], 'relScale', np.round(body['relative_scale3d'],3))
    print('  BOX parent', box['parent'], 'relRot', box['relative_rotation'], 'relLoc', np.round(box['relative_location'],2), 'relScale', np.round(box['relative_scale3d'],3), 'unscaled ext', box['box_unscaled_extent'])
    for e in extra:
        print('  extra visible mesh %-18s %-22s n=%6d size=%s center=%s inside_box=%.2f' % (e['comp'], e['mesh'], e['n'], np.round(e['size'],3), np.round(e['center'],3), e['frac_inside_box']))
    print('  ALL visible AABB size', np.round(allp.max(0) - allp.min(0), 4), 'min', np.round(allp.min(0), 3), 'max', np.round(allp.max(0), 3))
    print('  rotor check (ue vs ours):', chk)
    print('  mesh AABB actor-frame (m): size', np.round(hi - lo, 4), ' robust size', np.round(p_hi - p_lo, 4), ' min', np.round(p_lo, 3), ' max', np.round(p_hi, 3))
    print('  mesh span after 45deg rot (m):', np.round(span45, 4))
    print('  BOX center', np.round(bc, 3), 'size', np.round(2 * be, 4), 'yaw', byaw, '| verts inside box %.3f' % inside, ' outside per axis', np.round(out_axis, 3))
    print('  centroid', np.round(pa.mean(0), 4), ' box-center minus AABB-center xy', np.round(bc[:2] - cxy, 4))
    print('  arm angle peaks (deg, count):', arms)
    for k in ('band_low', 'band_mid', 'band_top'):
        print('  %s: %s' % (k, res[k]))
    print(ascii_map(pa[:, 0], pa[:, 1], title='TOP view (all verts)', xlab='x', ylab='y'))
    print(ascii_map(pa[low, 0], pa[low, 1], title='TOP view (lowest 12%% z band)', xlab='x', ylab='y'))
    print(ascii_map(pa[:, 0], pa[:, 2], nx=48, ny=14, title='SIDE view', xlab='x', ylab='z'))
    print(ascii_map(pa[:, 1], pa[:, 2], nx=48, ny=14, title='FRONT view', xlab='y', ylab='z'))
json.dump(summary, open(os.path.join(OUT, 'verts_analysis.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'verts_analysis.json'))
