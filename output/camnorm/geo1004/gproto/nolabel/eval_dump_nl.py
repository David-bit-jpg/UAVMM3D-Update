# -*- coding: utf-8 -*-
"""geo1004/eval_dump.py 的无标签副本（11-10）：eval_camnorm 原样推理，逐帧记录只存 pred / conf / K / seq_id / frame_id（不存 gt），
并把 pose_eval 的真值汇总换成空字典（不打印任何真值统计）。数据加载器内部仍按原代码载入索引，但真值不进入任何输出。
用法同 eval_dump.py（DUMP_PKL=输出路径）。"""
import os, pickle, sys
sys.path.insert(0, '@@ROOT@@/tools')
from uavdet3d.utils import pose_eval  # noqa: E402

OUT = os.environ['DUMP_PKL']
_ev = pose_eval.evaluate


def ev(model, loader, seq='xyz', match='top1', dist_thresh=1.0):
    s, recs = _ev(model, loader, seq=seq, match=match, dist_thresh=dist_thresh)
    keep = [{k: r[k] for k in ('pred', 'conf', 'K', 'seq_id', 'frame_id')} for r in recs]
    with open(OUT + '.tmp', 'wb') as f:
        pickle.dump(dict(records=keep, seq=seq, match=match, dist_thresh=dist_thresh, gt='not stored'), f)
    os.replace(OUT + '.tmp', OUT)
    return {}, recs


pose_eval.evaluate = ev
pose_eval.format_summary = lambda tag, s: '[nl] %s：真值汇总不计算（无标签链）' % tag
import eval_camnorm  # noqa: E402

sys.argv = ['eval_camnorm.py'] + sys.argv[1:]
sys.exit(eval_camnorm.main())
