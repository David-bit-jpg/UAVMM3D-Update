# Finish release/cvpr27 after copy_release.py: READMEs, repro/ files, prediction pickles (< 20 MB), ARTIFACTS.md, final sha256 list.
import os, sys, json, shutil, hashlib, csv
HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
DST = HERE + '/release/cvpr27'; D = HERE + '/release_docs'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def cp(src, rel):
    os.makedirs(os.path.dirname(DST + '/' + rel), exist_ok=True); shutil.copy2(src, DST + '/' + rel)


cp(D + '/README_top.md', 'README.md'); cp(D + '/README_pipeline.md', 'pipeline/README.md'); cp(D + '/README_variants.md', 'variants/README.md')
cp(D + '/README_baselines.md', 'baselines/README.md'); cp(D + '/ADVISOR_RUNS.md', 'baselines/ADVISOR_RUNS.md')
FEEDS = {
    'k249': 'Tab. 1, RealDrone3D row "CenterNet*, full pipeline of PST" (frames restricted to the 249 of the baselines).',
    'ourdet_baseproto': 'Tab. 1, CenterNet rows (simulation only, +DANN, +ST, +PST, real labels; RealDrone3D real labels = ubzoom).',
    'clean2': 'Tab. 2, rows 2-3 (buildup/: Step 1 = b_det1_test, Step 2 = c_students_test).',
    'heads': 'Tab. 2, row 5 = H3/mav_s0 (PST w/o composites in the depth refiner); H1-H7 = supplement head ablations.',
    'seeds': 'Tab. 3, runs 2-3 of PST (RealDrone3D, MAV6D shift 20) and of PST w/ RGB-only teachers (both targets).',
    'sup3': 'Tab. 3, run 2 of PST on MAV6D (seed shift 10).',
    'sensorfree': 'Tab. 3, "PST w/o sensors" = E1, "PST w/o sensors and composites" = E4; E1b/E2/E2a/E2b = supplement; E1/pooled_*_vsPST.json = Sec. 4 intervals.',
    'boxhull': 'Supplement: box-hull control of the depth refiner (pooled with sensorfree/E2).',
    'stage1seeds': 'Supplement: stage-1 networks retrained with new seeds.',
    'hparam_1117': 'Supplement: hyperparameter sensitivity.',
    'single_sbar': 'Sec. 4 (The full pipeline): single CenterNet + mirror + constant size.',
    'strata_1117': 'Supplement: per-stratum analysis of IR / LiDAR.',
    'lowres': 'Supplement: chain with crops cut from the 512x288 frames.',
    'corrupt_1117': 'Supplement: test-time corruptions.',
    'xdomain_1117': 'Supplement: cross-target transfer.',
    'cpu5': 'Supplement: RealDrone3D build-up of the steps.',
    'figdata': 'Fig. 4 (pseudo-label quality) and the student-follows-label analysis.',
}
for k, v in FEEDS.items():
    if os.path.isdir(DST + '/variants/' + k):
        open(DST + '/variants/%s/README_paper.md' % k, 'w', encoding='utf-8').write('# %s\n\nFeeds: %s\nOriginal: E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/%s/\n' % (k, v, k))
for f in ('MANIFEST.md', 'manifest.json', 'rescore_all.py', 'rescore_one.py', 'texparse.py', 'baseline_runs.py', 'baseline_runs.json', 'rescore_report.md',
          'code_time_audit.md', 'code_time_audit.json', 'audit_code_times.py', 'make_manifest_md.py', 'scan_preds.py', 'copy_release.py', 'finalize_release.py',
          'apply_to_release_branch.sh'):
    cp(HERE + '/' + f, 'repro/' + f)
cp(HERE + '/README.md', 'repro/README.md'); cp(HERE + '/work/match_baselines.py', 'repro/work/match_baselines.py')
for f in ('scan_baselines.json', 'scan_pst.json', 'scan_heads.json'): cp(HERE + '/work/' + f, 'repro/work/' + f)
# prediction files of every reported number (all < 20 MB): repro/preds/<drive>/<path>
M = json.load(open(HERE + '/manifest.json', encoding='utf-8'))
rows = []
for p, v in sorted(M['files'].items()):
    rel = 'repro/preds/' + p.replace(':', '', 1)
    if v['size'] <= (20 << 20):
        cp(p, rel); assert sha(DST + '/' + rel) == v['sha256'], p; rows.append((p, v['size'], v['sha256'], 'included at ' + rel))
    else: rows.append((p, v['size'], v['sha256'], 'not included (> 20 MB)'))
L = ['# Artifacts', '', 'Prediction files behind every reported number (included under `preds/`; run `REPRO_PRED_ROOT=<this folder>/preds python rescore_all.py`).',
     'The scorer (`D:/UAV3D_baselines_pack/scripts/score.py`, copy in `baselines/advisor_pack/00_UAV3D_baselines_pack/scripts/`) and the sealed test GT metas are read from their original paths:',
     ''] + ['- GT %s: `%s` sha256 %s' % (k, v['path'], v['sha256']) for k, v in M['metas'].items()] + \
    ['', '## Prediction files', '', '| path | size | sha256 | status |', '|---|---|---|---|'] + ['| `%s` | %d | %s | %s |' % r for r in rows]
L += ['', '## Not copied: data, caches, checkpoints, large or binary files found in the copied code folders', '',
      'Directories marked "dir (not traversed)" hold crops, caches, checkpoints, records or predictions of the runs (pruned by name). Files: size, mtime, sha256 (sha256 only when <= 50 MB).', '',
      '| path | size | mtime | reason / ext | sha256 |', '|---|---|---|---|---|']
for r in csv.reader(open(DST + '/repro/ARTIFACTS_skipped.tsv', encoding='utf-8'), delimiter='\t'):
    if r[0] == 'source_path': continue
    L.append('| `%s` | %s | %s | %s | %s |' % tuple(r[:5]))
open(DST + '/repro/ARTIFACTS.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
# final sha256 of every file in the release folder
with open(DST + '/repro/SHA256SUMS.txt', 'w', encoding='utf-8', newline='\n') as f:
    for dp, dn, fn in os.walk(DST):
        dn.sort()
        for n in sorted(fn):
            q = os.path.join(dp, n).replace('\\', '/')
            if q.endswith('repro/SHA256SUMS.txt'): continue
            f.write('%s  %s\n' % (sha(q), q[len(DST) + 1:]))
print('ok', len(rows), 'prediction files')
