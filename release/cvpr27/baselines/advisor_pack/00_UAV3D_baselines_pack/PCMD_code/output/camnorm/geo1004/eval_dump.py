# -*- coding: utf-8 -*-
"""eval_camnorm 原样跑一遍，额外把逐帧记录（gt / pred / conf / K / seq_id / frame_id）存成 pickle（路径取环境变量 DUMP_PKL）。
不改包代码：只在本进程里包一层 pose_eval.evaluate。用于 PREREG_irloc「能拿到时另报 mavic2 检出」。"""
import os
import pickle
import sys

sys.path.insert(0, '@@ROOT@@/tools')
from uavdet3d.utils import pose_eval  # noqa: E402

OUT = os.environ['DUMP_PKL']
_ev = pose_eval.evaluate


def ev(model, loader, seq='xyz', match='top1', dist_thresh=1.0):
    s, recs = _ev(model, loader, seq=seq, match=match, dist_thresh=dist_thresh)
    keep = [{k: r[k] for k in ('gt', 'pred', 'conf', 'K', 'seq_id', 'frame_id')} for r in recs]
    with open(OUT + '.tmp', 'wb') as f:
        pickle.dump(dict(records=keep, seq=seq, match=match, dist_thresh=dist_thresh), f)
    os.replace(OUT + '.tmp', OUT)
    return s, recs


pose_eval.evaluate = ev
import eval_camnorm  # noqa: E402

sys.argv = ['eval_camnorm.py'] + sys.argv[1:]
sys.exit(eval_camnorm.main())
