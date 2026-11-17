# -*- coding: utf-8 -*-
"""把代码里的占位符 @@ROOT@@ 换成本包所在的绝对路径（解压后运行一次即可；再次运行也安全）。
用法：python setup/fix_paths.py            # 自动取本文件上一级目录为根
     python setup/fix_paths.py /new/root  # 包被挪动后，换成新位置
"""
import hashlib, os, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1 and sys.argv[1] == '--reset':      # 打包前恢复成占位符
    ROOT = '@@ROOT@@'
else:
    ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..')).replace('\\', '/')
STATE = os.path.join(HERE, '.root')
TEXT = ('.py', '.sh', '.yaml', '.yml', '.txt', '.md', '.json', '.cfg')
old = open(STATE).read().strip() if os.path.exists(STATE) else '@@ROOT@@'
SKIP = {'data', 'runs', '__pycache__', '.git'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


PKGDIR = os.path.abspath(os.path.join(HERE, '..'))
OLDS = [o for o in {old, '@@ROOT@@'} if o != ROOT]
n = 0
changed = []
for top in ('Open3DUAVDet', 'scripts', 'pretrained'):
    for root, dirs, files in os.walk(os.path.join(PKGDIR, top)):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if not f.endswith(TEXT):
                continue
            p = os.path.join(root, f)
            try:
                t = open(p, encoding='utf-8').read()
            except UnicodeDecodeError:
                continue
            t2 = t
            for o in OLDS:
                t2 = t2.replace(o, ROOT)
            if t2 != t:
                open(p, 'w', encoding='utf-8', newline='').write(t2)
                changed.append(p); n += 1
# 研究时登记的"预注册文件哈希"（*.sha256）随路径替换失效 → 按新内容重写
for p in changed:
    if os.path.exists(p + '.sha256'):
        open(p + '.sha256', 'w', encoding='utf-8').write(sha(p) + '  ' + os.path.basename(p) + '\n')
open(STATE, 'w').write(ROOT)
print('根目录 = %s ；改写 %d 个文件' % (ROOT, n))
