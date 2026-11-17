# R2: overlap of AntiUAV3D v1 (sequence-only) and v2 (focal-stratified) splits, sequence and frame level
import os, json
V1 = 'D:/S2MUAV3D/real_outdoor_antiuav3d/labels'; V2 = 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels'
def rd(p): return [l.strip() for l in open(p) if l.strip()]
S = {}
for tag, root in (('v1', V1), ('v2', V2)):
    for sp in ('train', 'val', 'test'):
        S[(tag, sp)] = set(rd('%s/%s_frames.txt' % (root, sp)))
# cmpfull folders (what the baselines actually ran on)
def cmp_list(d):
    return set(f.replace('__', '/') for f in os.listdir(d) if f.endswith('.jpg'))
CF = {'cmpfull/outdoor_train': 'D:/cmpfull/outdoor_train/images', 'cmpfull/outdoor_val': 'D:/cmpfull/outdoor_val/images',
      'cmpfull/outdoor_test': 'D:/cmpfull/outdoor_test/images', 'cmpfull/_antiuav3d_v1/outdoor_test': 'D:/cmpfull/_antiuav3d_v1/outdoor_test/images'}
for k, d in CF.items():
    if os.path.isdir(d):
        S[('cf', k)] = cmp_list(d)
seq = lambda fs: set(f.split('/')[0] for f in fs)
out = {'sizes': {}, 'seqs': {}, 'pairs': []}
for k, v in S.items():
    out['sizes']['%s:%s' % k] = len(v); out['seqs']['%s:%s' % k] = sorted(seq(v))
for k, v in S.items():
    if k[0] != 'cf': print('%s %s: %d frames, seqs %s' % (k[0], k[1], len(v), sorted(seq(v))))
# exact match of cmpfull folders to split lists
for k in [x for x in S if x[0] == 'cf']:
    m = [('%s:%s' % j) for j in S if j[0] != 'cf' and S[j] == S[k]]
    print('%s (%d) identical to %s' % (k[1], len(S[k]), m))
    out['pairs'].append(dict(cf=k[1], identical_to=m))
# frame-level: exact frame id; also within +-k frames of same sequence (strides differ)
def near(a, b, k):
    idx = {}
    for f in b:
        s, n = f.split('/'); idx.setdefault(s, []).append(int(n.split('.')[0]))
    import bisect
    for s in idx: idx[s].sort()
    c = 0
    for f in a:
        s, n = f.split('/'); n = int(n.split('.')[0])
        L = idx.get(s)
        if not L: continue
        i = bisect.bisect_left(L, n - k)
        if i < len(L) and L[i] <= n + k: c += 1
    return c
print()
rows = []
for a in ('train', 'val', 'test'):
    for b in ('train', 'val', 'test'):
        A, B = S[('v1', a)], S[('v2', b)]
        sa, sb = seq(A), seq(B)
        r = dict(v1=a, v2=b, seq_common=sorted(sa & sb), frames_exact=len(A & B),
                 v1_frames_in_common_seqs=sum(f.split('/')[0] in sb for f in A), v2_frames_in_common_seqs=sum(f.split('/')[0] in sa for f in B),
                 v1_within3=near(A, B, 3), v2_within3=near(B, A, 3), nA=len(A), nB=len(B))
        rows.append(r)
        print('v1 %-5s x v2 %-5s | common seqs %-55s | exact frames %4d | v1 frames in shared seqs %4d/%d | v2 frames in shared seqs %4d/%d | v1 within+-3 %4d | v2 within+-3 %4d'
              % (a, b, r['seq_common'], r['frames_exact'], r['v1_frames_in_common_seqs'], len(A), r['v2_frames_in_common_seqs'], len(B), r['v1_within3'], r['v2_within3']))
out['rows'] = rows
json.dump(out, open('r2_overlap.json', 'w'), indent=1)
