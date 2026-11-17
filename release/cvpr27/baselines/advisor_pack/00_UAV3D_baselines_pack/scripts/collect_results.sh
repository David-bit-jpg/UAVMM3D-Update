#!/usr/bin/env bash
# 把所有预测结果和训练日志末尾打成一个小 zip（几 MB），发回给我们出表。用法：bash scripts/collect_results.sh
cd "$(dirname "$0")/.."
source scripts/env.sh
$PY scripts/score.py > /dev/null 2>&1 || echo "（打分失败，只打包预测）"
$PY - <<'EOF'
import glob, os, zipfile, time
out = 'results_to_send_%s.zip' % time.strftime('%Y%m%d_%H%M')
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob('runs/*/pred_*_test.pkl')):
        z.write(p); n += 1
    for p in sorted(glob.glob('runs/*/train.log')):
        lines = open(p, encoding='utf-8', errors='replace').read().splitlines()[-200:]
        z.writestr(p.replace('train.log', 'train_tail.log'), '\n'.join(lines))
    if os.path.exists('runs/SCORES.md'):
        z.write('runs/SCORES.md')
    for p in sorted(glob.glob('runs/*.out')):
        z.write(p)
print('已打包 %d 个预测文件 → %s（把这个文件发回即可）' % (n, out))
EOF
