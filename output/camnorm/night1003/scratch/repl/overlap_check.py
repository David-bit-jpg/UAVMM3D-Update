# -*- coding: utf-8 -*-
"""Experiment R, step 0 (CPU, LABEL-FREE): how does the MAV6D val split relate to train / test, and did any val frame
enter the AdaBN re-estimation?

Reads ONLY the fields seq / frame / cls of @@MMCACHE@@/mav6d_cn_obj/{train,val,test}/index.pkl metas (the pickles carry
boxes9d too; this script never touches them), the AdaBN frame list kzst1/adabn_frames_train5.json, and the label-free
info block of the stage-1 train dump. Writes overlap_R.json next to this file.

    python overlap_check.py
"""
import sys

sys.dont_write_bytecode = True
import hashlib       # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import pickle        # noqa: E402
import time          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
CACHE = '@@MMCACHE@@/mav6d_cn_obj'
ADABN_FRAMES = os.path.join(SCRATCH, 'kzst1', 'adabn_frames_train5.json')
TRAIN_DUMP = os.path.join(SCRATCH, 'kzst1', 'real_ens5flip_kpKZavg_zoom_train5.pkl')
VAL_SEQ_LIST = '@@ROOT@@/tools/cfgs/subsets/mav6d_camnorm/val_sequences.txt'
LABEL_FREE_FIELDS = ('seq', 'frame', 'cls')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load_keys(split):
    p = os.path.join(CACHE, split, 'index.pkl')
    idx = pickle.load(open(p, 'rb'))
    rows = []
    for i in idx['valid_idx']:
        m = idx['metas'][int(i)]
        rows.append(tuple(str(m[k]) for k in LABEL_FREE_FIELDS))          # only seq / frame / cls
    hdr = {k: idx.get(k) for k in ('split', 'val_frac', 'seed', 'object_frame', 'format') if k in idx}
    del idx
    return rows, hdr, sha256(p)


def scene_of(seq):
    p = seq.split('/')
    return '/'.join(p[:2]) if len(p) >= 3 else '?'


def main():
    t0 = time.time()
    S = {}
    for sp in ('train', 'val', 'test'):
        rows, hdr, h = load_keys(sp)
        S[sp] = {'rows': rows, 'hdr': hdr, 'sha256': h}
    seqs = {sp: {r[0] for r in S[sp]['rows']} for sp in S}
    frames = {sp: {(r[0], r[1]) for r in S[sp]['rows']} for sp in S}
    scenes = {sp: {scene_of(s) for s in seqs[sp]} for sp in S}
    out = {'cache': CACHE, 'index_sha256': {sp: S[sp]['sha256'] for sp in S}, 'index_header': {sp: S[sp]['hdr'] for sp in S},
           'fields_read': list(LABEL_FREE_FIELDS), 'per_split': {}}
    for sp in S:
        per_cls = {}
        for c in ('phantom4', 'mavic2'):
            rr = [r for r in S[sp]['rows'] if r[2] == c]
            per_cls[c] = {'frames': len(rr), 'sequences': len({r[0] for r in rr}), 'scenes': sorted({scene_of(r[0]) for r in rr})}
        out['per_split'][sp] = {'frames': len(S[sp]['rows']), 'sequences': len(seqs[sp]), 'scenes': sorted(scenes[sp]), 'per_class': per_cls}
    out['val_sequences'] = sorted(seqs['val'])
    out['val_frames_per_sequence'] = {s: sum(1 for r in S['val']['rows'] if r[0] == s) for s in sorted(seqs['val'])}
    out['overlap'] = {
        'seq_train_val': sorted(seqs['train'] & seqs['val']), 'seq_train_test': sorted(seqs['train'] & seqs['test']),
        'seq_val_test': sorted(seqs['val'] & seqs['test']),
        'frame_train_val': len(frames['train'] & frames['val']), 'frame_train_test': len(frames['train'] & frames['test']),
        'frame_val_test': len(frames['val'] & frames['test']),
        'scene_train_val': sorted(scenes['train'] & scenes['val']), 'scene_train_test': sorted(scenes['train'] & scenes['test']),
        'scene_val_test': sorted(scenes['val'] & scenes['test']),
        'val_scenes_not_in_train': sorted(scenes['val'] - scenes['train']),
        'test_scenes_not_in_train': sorted(scenes['test'] - scenes['train'])}
    # sequence-name collisions across splits: same name, different split (per class/scene/seq triple)
    if os.path.exists(VAL_SEQ_LIST):
        lst = [ln.strip() for ln in open(VAL_SEQ_LIST, encoding='utf-8', errors='replace') if ln.strip() and not ln.startswith('#')]
        out['val_sequences_txt'] = {'path': VAL_SEQ_LIST, 'n': len(lst), 'equals_index': sorted(lst) == sorted(seqs['val'])}
    # AdaBN re-estimation frames
    A = json.load(open(ADABN_FRAMES, encoding='utf-8'))
    ak = {(str(s), str(f)) for s, f in A['keys']}
    a_seq = {s for s, _ in ak}
    out['adabn'] = {'file': ADABN_FRAMES, 'sha256': sha256(ADABN_FRAMES), 'n_frames': len(ak), 'n_seq': len(a_seq),
                    'frames_in_val': len(ak & frames['val']), 'frames_in_test': len(ak & frames['test']),
                    'frames_in_train': len(ak & frames['train']), 'seq_in_val': sorted(a_seq & seqs['val']),
                    'seq_in_test': sorted(a_seq & seqs['test']), 'scenes': sorted({scene_of(s) for s in a_seq}),
                    'val_scenes_covered_by_adabn_seqs': sorted({scene_of(s) for s in a_seq} & scenes['val'])}
    # where val images entered the stage-1 pipeline label-free (self-calibration of lwh)
    if os.path.exists(TRAIN_DUMP):
        D = pickle.load(open(TRAIN_DUMP, 'rb'))
        info = D['info']
        out['train_dump_info'] = {k: info.get(k) for k in ('split', 'size_source', 'calib_lwh', 'kp2d_zoom', 'ckpts', 'gt_scrubbed')}
        del D
    out['elapsed_s'] = time.time() - t0
    out['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(HERE, 'overlap_R.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    o = out['overlap']
    print('splits:', {sp: (out['per_split'][sp]['frames'], out['per_split'][sp]['sequences'], out['per_split'][sp]['scenes']) for sp in S})
    print('sequence overlap train/val %d, train/test %d, val/test %d; frame overlap %d / %d / %d' % (
        len(o['seq_train_val']), len(o['seq_train_test']), len(o['seq_val_test']), o['frame_train_val'], o['frame_train_test'], o['frame_val_test']))
    print('scene overlap train/val %s, train/test %s' % (o['scene_train_val'], o['scene_train_test']))
    print('AdaBN frames: %d (%d seq); in val %d, in test %d, in train %d' % (
        out['adabn']['n_frames'], out['adabn']['n_seq'], out['adabn']['frames_in_val'], out['adabn']['frames_in_test'], out['adabn']['frames_in_train']))
    print('val sequences (%d): %s' % (len(seqs['val']), out['val_frames_per_sequence']))
    print('wrote %s (%.1f s)' % (os.path.join(HERE, 'overlap_R.json'), out['elapsed_s']))


if __name__ == '__main__':
    main()
