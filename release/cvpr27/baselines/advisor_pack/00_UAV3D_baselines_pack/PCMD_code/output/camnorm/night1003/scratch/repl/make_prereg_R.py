# -*- coding: utf-8 -*-
"""Writes prereg_R.json + prereg_R.json.sha256 for experiment R. Must run BEFORE any val ground truth is read: it refuses
to run when a real val dump (val_kz / val_adabn) or a val evaluation already exists under the output root.

Contents: the decision spec (eval_R.R_SPEC, verbatim), sha256 of every analysis / inference script and byte copy, sha256
of the member checkpoints (release KZ, stage-1 AdaBN == recorded in the stage-1 json, pure-sim position ensemble), the
label-free split facts (overlap_R.json), the stage-1 TEST reference and the planned commands.
"""
import sys

sys.dont_write_bytecode = True
import json          # noqa: E402
import os            # noqa: E402
import time          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import repl_common as RC   # noqa: E402
import eval_R              # noqa: E402

OUT = os.path.join(HERE, 'prereg_R.json')
CODE = ['eval_R.py', 'repl_common.py', 'kz_fuse_offline.py', 'kzst_common.py', 'adabn_kz.py', 'dump_kz_members.py',
        'dump_members_repl.py', 'replay_members_repl.py', 'overlap_check.py']


def main():
    for tag in ('val_kz', 'val_adabn', 'eval_R'):
        if os.path.exists(os.path.join(RC.OUT_ROOT, tag)):
            raise SystemExit('%s already exists under the output root -- too late to pre-register' % tag)
    if os.path.exists(OUT):
        raise SystemExit('%s exists; refusing to overwrite a pre-registration' % OUT)
    RC.check_byte_copies()
    s1 = json.load(open(RC.ADABN_STAGE1_JSON, encoding='utf-8'))
    bn_now = {os.path.basename(p): RC.sha256(p) for p in RC.ADABN_CKPTS}
    assert bn_now == s1['checkpoints_sha256'], 'AdaBN checkpoints on disk differ from the stage-1 record'
    ov_path = os.path.join(HERE, 'overlap_R.json')
    ov = json.load(open(ov_path, encoding='utf-8'))
    assert ov['per_split']['val']['frames'] == eval_R.R_SPEC['expected_frames']
    assert ov['per_split']['val']['sequences'] == eval_R.R_SPEC['expected_sequences']
    s1r = json.load(open(RC.S1_RESULTS, encoding='utf-8'))
    n_seq = eval_R.R_SPEC['expected_sequences']
    need = next(k for k in range(n_seq + 1) if k / n_seq >= eval_R.R_SPEC['primary']['seq_better_frac_min'])
    P = {
        'experiment': 'R -- replication of the stage-1 AdaBN + SA-TM heading gain on the MAV6D VAL split',
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statement': 'Written before any val ground truth was read in this experiment: no val member dump, no sealed val ground '
                     'truth and no val evaluation exist at this time (asserted). The val split was inspected only through '
                     'seq / frame / cls of index.pkl (overlap_check.py).',
        'spec': eval_R.R_SPEC,
        'seq_better_needed': '%d of %d val sequences (if all %d have >= 1 matched frame; otherwise ceil(0.60 x n))' % (need, n_seq, n_seq),
        'code_sha256': {f: RC.sha256(os.path.join(HERE, f)) for f in CODE},
        'checkpoints_sha256': {'adabn_stage1': bn_now, 'release_kz': {os.path.basename(p): RC.sha256(p) for p in RC.RELEASE_KZ},
                               'positions_ramp20': {os.path.basename(p): RC.sha256(p) for p in RC.RAMP20}},
        'adabn_recipe': {'stage1_json': RC.ADABN_STAGE1_JSON, 'reestimation': s1.get('reestimation'),
                         'frames_json_sha256': ov['adabn']['sha256'], 'stage1_prereg_sha256': s1.get('prereg_sha256')},
        'split_facts_labelfree': {'overlap_json': ov_path, 'overlap_json_sha256': RC.sha256(ov_path), 'per_split': ov['per_split'],
                                  'overlap': ov['overlap'], 'adabn_frames_in_val': ov['adabn']['frames_in_val'],
                                  'adabn_seq_in_val': ov['adabn']['seq_in_val'], 'val_frames_per_sequence': ov['val_frames_per_sequence']},
        'stage1_test_reference': {'results_json_sha256': RC.sha256(RC.S1_RESULTS), 'KZ_SATM': s1r['table']['A0']['all']['head'],
                                  'BN_SATM': s1r['table']['BN']['all']['head'], 'delta': s1r['P1_BN']['delta'],
                                  'single_level_ci95': s1r['P1_BN']['single_level_ci95'], 'ads_indoor': s1r['ads_indoor']},
        'planned_commands': ['dump_members_repl.py --split val --pos sim --members kz --tag val_kz   (GPU)',
                             'replay_members_repl.py --src-tag val_kz --tag val_adabn                (GPU, after the first)',
                             'eval_R.py all                                                          (CPU, after both)'],
        'caveats_known_before_data': [
            'val sequences are held out from the official MAV6D train list by sequence; every val scene (mavic2/02, mavic2/03, '
            'phantom4/01, phantom4/02) also occurs in train and therefore among the AdaBN re-estimation frames (as do all test scenes)',
            'the self-calibrated lwh of the val dump is estimated on val images themselves (label-free; identical for both arms)',
            '16 sequences only: the sequence-cluster interval is wide; mavic2/02/0202 has 16 frames'],
    }
    RC.jdump(P, OUT)
    h = RC.sha256(OUT)
    with open(OUT + '.sha256', 'w', encoding='utf-8') as f:
        f.write('%s  prereg_R.json\n' % h)
    print('wrote %s  sha256 %s' % (OUT, h))


if __name__ == '__main__':
    main()
