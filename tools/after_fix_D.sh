#!/bin/bash
# D: 盘原始数据机头修正完成后：仿真缓存 index.pkl 换成机头版（原版改名保留），随机抽 20 个序列端到端核对。
L=/e/Open3DUAVDet/output/label_fix
until grep -q "=== 完成 D:/data_collect" $L/fix_D.log 2>/dev/null; do sleep 60; done
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
PY=/d/Miniconda3/envs/city/python.exe
$PY - <<'PYEOF' >> $L/after_fix_D.log 2>&1
import os, pickle, shutil, time
for c in ('indoor8cn', 'indoor8jpg'):
    for s in ('train', 'test'):
        d = 'E:/mmcache/%s/%s' % (c, s)
        src, old, nose = d + '/index.pkl', d + '/index_labelx_v1.pkl', d + '/index_nose.pkl'
        if not os.path.exists(old):
            os.replace(src, old)
        idx = pickle.load(open(nose, 'rb'))
        idx['raw_label_fix'] = 'nose-x-v1-20260915'
        idx['label_fix'] = dict(idx['label_fix'], note='index.pkl = 机头修正版；原始数据已就地修正；原版在 index_labelx_v1.pkl')
        with open(src + '.tmp', 'wb') as f:
            pickle.dump(idx, f, protocol=4)
        os.replace(src + '.tmp', src)
        print(time.strftime('%H:%M:%S'), d, 'index.pkl <- index_nose.pkl（原版 -> index_labelx_v1.pkl）')
PYEOF
combos=$(for c in /d/data_collect/*/carla_data/*/*/*; do [ -d "$c/boxes_rgb" ] && echo "$c"; done | shuf -n 20 --random-source=<(yes))
$PY "/e/Open3DUAVDet/tools/annot_audit/verify_raw_fix.py" D:/data_collect D:/data_collect_label_backup_20260915 $combos >> $L/after_fix_D.log 2>&1
echo "=== after_fix_D done $(date)" >> $L/after_fix_D.log
