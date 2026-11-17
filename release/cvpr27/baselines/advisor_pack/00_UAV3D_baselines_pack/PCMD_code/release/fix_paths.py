# -*- coding: utf-8 -*-
"""把代码里的路径占位符换成本机的真实路径（clone 之后运行一次；再次运行也安全，可以改成新路径）。

占位符           含义                                   默认值（原机器上的位置）
@@ROOT@@         本仓库根目录                           本文件上一级目录（自动）
@@MMCACHE@@      数据缓存根（mav6d_cn_obj、仿真缓存等）  E:/mmcache
@@MAV6D@@        MAV6D 真实数据原图                     E:/MAV6D
@@TORCH_HOME@@   ImageNet 预训练权重（TORCH_HOME）       E:/torch_home
@@DATA_COLLECT@@ 仿真录制原始数据                       E:/data_collect

用法：
  python release/fix_paths.py                                   # 根目录自动取，其余用默认值
  python release/fix_paths.py --mmcache /data/mmcache --mav6d /data/MAV6D --torch-home /data/torch_home
  python release/fix_paths.py --reset                           # 换回占位符（提交代码前用）
研究时登记的"预注册文件哈希"（*.sha256）随内容变化而失效，这里按新内容重写；默认也不强制核对（UAV_STRICT_HASH=1 才核对）。
"""
import argparse, hashlib, json, os, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
STATE = os.path.join(HERE, '.paths_state.json')
TEXT = ('.py', '.sh', '.yaml', '.yml', '.txt', '.md', '.json', '.cfg')
SKIP = {'.git', '__pycache__', 'runs', 'data'}
KEYS = [('@@ROOT@@', 'root', None), ('@@MMCACHE@@', 'mmcache', 'E:/mmcache'), ('@@MAV6D@@', 'mav6d', 'E:/MAV6D'),
        ('@@TORCH_HOME@@', 'torch_home', 'E:/torch_home'), ('@@DATA_COLLECT@@', 'data_collect', 'E:/data_collect')]

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=REPO)
for ph, k, d in KEYS[1:]:
    ap.add_argument('--' + k.replace('_', '-'), dest=k, default=d)
ap.add_argument('--reset', action='store_true')
a = ap.parse_args()

old = json.load(open(STATE, encoding='utf-8')) if os.path.exists(STATE) else {}
new = {} if a.reset else {ph: os.path.abspath(getattr(a, k)).replace('\\', '/') if k == 'root' else getattr(a, k).replace('\\', '/').rstrip('/')
                          for ph, k, _ in KEYS}
# 当前文件里的写法 → 目标写法（先换长的，避免前缀互相吃掉）
pairs = []
for ph, _, _ in KEYS:
    cur = old.get(ph, ph)
    tgt = new.get(ph, ph)
    if cur != tgt:
        pairs.append((cur, tgt))
pairs.sort(key=lambda x: -len(x[0]))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


changed = []
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        p = os.path.join(root, f)
        if not f.endswith(TEXT) or os.path.abspath(p) == os.path.abspath(__file__) or os.path.abspath(p) == os.path.join(HERE, 'README.md'):
            continue
        try:
            t = open(p, encoding='utf-8', newline='').read()
        except UnicodeDecodeError:
            continue
        t2 = t
        for x, y in pairs:
            t2 = t2.replace(x, y)
        if t2 != t:
            open(p, 'w', encoding='utf-8', newline='').write(t2)
            changed.append(p)
for p in changed:
    if os.path.exists(p + '.sha256'):
        rest = open(p + '.sha256', encoding='utf-8').read().split(None, 1)
        open(p + '.sha256', 'w', encoding='utf-8').write(sha(p) + ((' ' + rest[1]) if len(rest) > 1 else '\n'))
if a.reset:
    if os.path.exists(STATE):
        os.remove(STATE)
else:
    json.dump(new, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('改写 %d 个文件；%s' % (len(changed), '已换回占位符' if a.reset else '  '.join('%s=%s' % kv for kv in new.items())))
