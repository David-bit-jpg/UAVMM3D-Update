# Copy (never move) the paper's experiment code into the release folder. Read-only on every source.
# usage: python copy_release.py <release_root>   (e.g. D:/Open3DUAVDet_clean_release/release/cvpr27)
# Writes <release_root>/repro/{FILES_SHA256.tsv, ARTIFACTS_skipped.tsv}; never deletes anything.
import os, sys, re, json, shutil, hashlib, time

DST = sys.argv[1].rstrip('/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
GP = G + '/gproto'; R = GP + '/review1113'; X = GP + '/srcx'; CFG = 'E:/Open3DUAVDet/tools/cfgs'
TEXT_EXT = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.csv', '.ps1', '.html', '.cfg', '.ini', '.toml', '.cu', '.cpp', '.c', '.h', '.hpp',
            '.pyx', '.log', '.out', '.tex', '.bat', '.gitignore', '.gitattributes', ''}
BIN_KEEP_EXT = {'.pkl'}            # only where allow_pkl=True (pseudo-label tables of the advisor patches)
PRUNE = {'__pycache__', 'recs', 'ou', 'ref', 'ckpt', 'checkpoints', 'checkpoint', 'preds', 'runs', 'smoke', '_smoke_archive', 'data', 'pdf', 'picture',
         'figures', 'cache', 'crops', 'wavedata', 'assets', 'output', 'outputs', 'logs_big', '.git', 'ImageSets', 'mmcache'}
PRUNE_BL = {'__pycache__', 'checkpoints', 'checkpoint', 'ckpt', 'smoke', '_smoke_archive', 'pdf', 'picture', 'figures', '.git', 'outputs', 'ImageSets', 'wavedata', 'assets', 'cache'}
MAX_TEXT = 2 << 20                 # 2 MB for text files (logs above this are listed, not copied)
MAX_BIN = 20 << 20
files, skipped = [], []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def ext_of(n):
    n = n.lower()
    for e in ('.gitignore', '.gitattributes'):
        if n.endswith(e): return e
    b = re.sub(r'\.bak[_0-9a-z]*$', '', n)          # foo.py.bak_1113 -> .py
    return os.path.splitext(b)[1]


def put(src, rel, allow_pkl=False):
    st = os.stat(src); e = ext_of(os.path.basename(src)); dst = DST + '/' + rel
    ok = (e in TEXT_EXT and st.st_size <= MAX_TEXT) or (allow_pkl and e in BIN_KEEP_EXT and st.st_size <= MAX_BIN)
    if not ok:
        skipped.append((src, st.st_size, time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime)), e)); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    files.append((rel, sha(dst), st.st_size, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)), src))


def tree(src, rel, allow_pkl=False, prune=PRUNE, depth=8):
    if depth < 0 or not os.path.isdir(src): return
    for de in sorted(os.scandir(src), key=lambda d: d.name):
        if de.is_dir(follow_symlinks=False):
            if de.name in prune: skipped.append((de.path.replace('\\', '/') + '/', -1, '', 'dir (not traversed)')); continue
            tree(de.path, rel + '/' + de.name, allow_pkl, prune, depth - 1)
        elif de.is_file():
            put(de.path, rel + '/' + de.name, allow_pkl)


# ---------- pipeline: the clean chain srcx
for n in sorted(os.listdir(X)):
    p = X + '/' + n
    if os.path.isfile(p): put(p, 'pipeline/srcx/' + n)
tree(X + '/lib', 'pipeline/srcx/lib')
# per-run evidence of the reported srcx runs (run logs, small json; no recs/crops/preds)
for run in ('in2mav', 'in2mav_none', 'out2antifs', 'out2antifs_none', 'out2antifs_k249'):
    for n in sorted(os.listdir(X + '/' + run)):
        p = X + '/' + run + '/' + n
        if os.path.isfile(p) and n.endswith('.json'): put(p, 'pipeline/runs/%s/%s' % (run, n))
    if os.path.isdir(X + '/' + run + '/logs'): tree(X + '/' + run + '/logs', 'pipeline/runs/%s/logs' % run)
tree(X + '/logs', 'pipeline/srcx_logs')
for p in (GP + '/gpu_slot.sh', GP + '/mem_guard.sh'):
    put(p, 'pipeline/gproto/' + os.path.basename(p))

# ---------- configs: srcx/, ourdet_baseproto/, plus everything reachable through _BASE_CONFIG_ from the configs named in sx.py / mkcfg
roots = set()
for d in ('models/uavdet_3d/camnorm/srcx', 'models/uavdet_3d/camnorm/ourdet_baseproto'):
    for n in os.listdir(CFG + '/' + d):
        if n.endswith('.yaml'): roots.add(d + '/' + n)
for f in [X + '/lib/sx.py', X + '/mix_source.json', X + '/lib/mkcfg.py']:
    for m in re.findall(r"(?:CFGD \+ |cfgs/models/uavdet_3d/camnorm/)'?([A-Za-z0-9_./]+\.yaml)", open(f, encoding='utf-8', errors='replace').read()):
        roots.add('models/uavdet_3d/camnorm/' + m.split('camnorm/')[-1])
seen = set(); todo = sorted(roots)
while todo:
    r = todo.pop()
    if r in seen or not os.path.isfile(CFG + '/' + r): continue
    seen.add(r); put(CFG + '/' + r, 'pipeline/tools/cfgs/' + r)
    for m in re.findall(r'cfgs/([A-Za-z0-9_./-]+\.yaml)', open(CFG + '/' + r, encoding='utf-8', errors='replace').read()):
        todo.append(m)

# ---------- variants: every review1113 experiment folder (code / notes / logs only)
for n in sorted(os.listdir(R)):
    p = R + '/' + n
    if os.path.isdir(p): tree(p, 'variants/' + n)
    else: put(p, 'variants/_top/' + n)
put(GP + '/RUNNING.md', 'variants/_top/gproto_RUNNING.md'); put(GP + '/REVIEW_QUEUE_1113.md', 'variants/_top/gproto_REVIEW_QUEUE_1113.md')
put(GP + '/ADVISOR_NOTES.md', 'variants/_top/gproto_ADVISOR_NOTES.md')

# ---------- baselines: ported code of the five baselines + pseudo-label builders
CM = G + '/cmpfull'
for n in sorted(os.listdir(CM)):
    p = CM + '/' + n
    if os.path.isfile(p): put(p, 'baselines/cmpfull/' + n)
for m in ('gupnet', 'monocd', 'monodgp', 'monodetr', 'monocop'):
    tree(CM + '/' + m, 'baselines/cmpfull/' + m, prune=PRUNE_BL)
# advisor pack + patches in the order they were applied
PACKS = ['UAV3D_baselines_pack', 'UAV3D_pack_update_1112', 'UAV3D_pack_hotfix_1112b', 'UAV3D_pack_prio_1113', 'UAV3D_pack_fix_1113', 'UAV3D_pack_fix_1114',
         'UAV3D_pack_fix_1115', 'UAV3D_pack_fix_1115b', 'UAV3D_pack_fix_1116', 'UAV3D_pack_fix_1117', 'UAV3D_pack_fix_1117b', 'UAV3D_pack_fix_1117c']
for i, pk in enumerate(PACKS):
    base = 'D:/' + pk; rel = 'baselines/advisor_pack/%02d_%s' % (i, pk)
    for n in sorted(os.listdir(base)):
        p = base + '/' + n
        if os.path.isfile(p): put(p, rel + '/' + n, allow_pkl=True)
        elif n == 'data':      # pseudo-label tables (small pkl) and their builders; GT metas and image caches are listed, not copied
            for dp, dn, fn in os.walk(p):
                dn[:] = [d for d in dn if d not in ('outdoor_test', 'real_test', 'outdoor_train', 'real_train', 'sim_train', 'sim_outdoor_train', 'sim_outdoor_town10_kept')]
                for f in fn:
                    s = os.path.join(dp, f).replace('\\', '/')
                    if f.endswith('_meta.npz'): skipped.append((s, os.path.getsize(s), '', 'GT meta (sealed test labels), not copied')); continue
                    put(s, rel + '/' + s[len(base) + 1:], allow_pkl=True)
        else:
            tree(p, rel + '/' + n, allow_pkl=False, prune=PRUNE_BL)
# shipped zips (small ones are the exact payload that went to the advisor)
for z in sorted(os.listdir('D:/')):
    if z.startswith('UAV3D_pack') and z.endswith('.zip'):
        s = 'D:/' + z
        skipped.append((s, os.path.getsize(s), '', 'zip as shipped (sha256 below if <= 50 MB)'))

os.makedirs(DST + '/repro', exist_ok=True)
with open(DST + '/repro/FILES_SHA256.tsv', 'w', encoding='utf-8', newline='\n') as f:
    f.write('release_path\tsha256\tsize\tsource_mtime\tsource_path\n')
    for r in files: f.write('\t'.join(map(str, r)) + '\n')
with open(DST + '/repro/ARTIFACTS_skipped.tsv', 'w', encoding='utf-8', newline='\n') as f:
    f.write('source_path\tsize\tmtime\treason_or_ext\tsha256\n')
    for s, sz, mt, why in skipped:
        h = sha(s) if (sz > 0 and sz <= (50 << 20)) else ''
        f.write('%s\t%s\t%s\t%s\t%s\n' % (s, sz, mt, why, h))
print('copied %d files, %.1f MB; skipped/listed %d' % (len(files), sum(r[2] for r in files) / 1e6, len(skipped)))
