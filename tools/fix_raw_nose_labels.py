# -*- coding: utf-8 -*-
"""原始数据（carla_data 格式）标注就地修正：机头改为 +x（2026-09-15，用户要求新旧数据都改，7 个机型含 unk3 统一 −90°）。

问题：Drone Pack 网格机头朝 actor +Y，录制器/UDataLogger 记录的 BoundingCheck 盒 8 角点里 c1−c0（盒子局部 X）不是机头。
      证据见 docs/audit/ANNOTATION_CONSISTENCY_2026-09-15.md 第 1、5、6 节。
修正：盒子局部坐标换成 X' = 旧 +Y（机头）、Y' = 旧 −X（机体右侧，UE 左手系），Z 不变。
      对应 8 角点重排 new = old[[1, 2, 3, 0, 5, 6, 7, 4]]（每个面循环移一位），角点坐标本身一个都不改；
      drone_info.pkl 每行 "型号, 速度, L, W, H" 的 L、W 互换。只改朝向，不改尺寸比例和物理尺度。
      重排后 corners_to_9params 得到的 9 参数 == sim_asset_fix.fix_box(旧 9 参数, nose=True, scale=False)，每个序列自检。

安全：
  - 每个序列先把 boxes_rgb / boxes_ir / boxes_dvs / drone_info.pkl 原件打包成一个 zip（ZIP_STORED）放到备份根目录（同盘），
    zip 写完并校验后才动原文件；新文件一律从 zip 里的原件计算并原地覆盖（重跑不会转两次；标记落地前中断则重跑覆盖全部文件）；
  - 全部写完才在序列目录落标记 label_fix_nose_x.json；有标记的序列跳过。
  - 恢复：tools/fix_raw_nose_labels.py --restore <同样的 roots>（从 zip 还原并删除标记）。

    cd E:/Open3DUAVDet/tools
    python fix_raw_nose_labels.py --roots D:/data_collect --backup D:/data_collect_label_backup_20260915 [--limit 1] [--workers 6]
"""
import argparse
import glob
import io
import json
import os
import pickle
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(TOOLS))
import build_mm_cache as BMC   # noqa: E402
from uavdet3d.utils import sim_asset_fix as F   # noqa: E402

VERSION = 'nose-x-v1-20260915'
MARKER = 'label_fix_nose_x.json'
PERM = [1, 2, 3, 0, 5, 6, 7, 4]
BOX_DIRS = ('boxes_rgb', 'boxes_ir', 'boxes_dvs')


def protocol_of(b):
    return b[1] if len(b) > 1 and b[0] == 0x80 else 2


def fix_rows(rows):
    out = []
    for r in rows:
        name = r[0]
        if BMC.class_of(name) is None:
            raise ValueError('未知目标名 %r（不在机型表里，拒绝静默处理）' % (name,))
        corners = list(r[1:9])
        assert len(corners) == 8 and len(r) == 9, '行格式不是 [name, 8 corners]：len=%d' % len(r)
        out.append([name] + [corners[j] for j in PERM])
    return out


def fix_drone_info(s):
    lines = s.split('\n')
    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        parts = ln.split(',')
        assert len(parts) == 5, 'drone_info 行格式异常: %r' % ln
        parts[2], parts[3] = parts[3], parts[2]
        out.append(','.join(parts))
    return '\n'.join(out)


def check_rows(old_rows, new_rows):
    worst = 0.0
    for o, n in zip(old_rows, new_rows):
        co = np.array(o[1:], float).reshape(8, 3)
        cn = np.array(n[1:], float).reshape(8, 3)
        bo = BMC.corners_to_9params(co)
        bn = BMC.corners_to_9params(cn)
        ref = F.fix_box(bo, BMC.class_of(o[0]), nose=True, scale=False)
        Rr = R.from_euler('xyz', ref[6:9]).as_matrix()
        Rn = R.from_euler('xyz', bn[6:9]).as_matrix()
        d = max(np.abs(bn[:6] - ref[:6]).max(), np.abs(Rn - Rr).max())
        worst = max(worst, float(d))
    return worst


def combos_under(root):
    return sorted(p for p in glob.glob(os.path.join(root, '*', 'carla_data', '*', '*', '*')) if os.path.isdir(os.path.join(p, 'boxes_rgb')))


def zip_path(combo, data_root, backup_root):
    rel = os.path.relpath(combo, data_root)
    return os.path.join(backup_root, rel + '.zip')


def backup_combo(combo, zp):
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            if 'BACKUP_COMPLETE' in z.namelist():
                return 'exists'
    os.makedirs(os.path.dirname(zp), exist_ok=True)
    tmp = zp + '.tmp'
    n = 0
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as z:
        for d in BOX_DIRS:
            for f in sorted(glob.glob(os.path.join(combo, d, '*.pkl'))):
                z.write(f, arcname=d + '/' + os.path.basename(f)); n += 1
        z.write(os.path.join(combo, 'drone_info.pkl'), arcname='drone_info.pkl')
        z.writestr('BACKUP_COMPLETE', json.dumps({'combo': combo, 'n_box_files': n, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}))
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        assert bad is None, 'zip 校验失败: %s' % bad
    os.replace(tmp, zp)
    return 'written %d' % n


def write_atomic(path, data):
    """原地覆盖（不再 tmp + rename：机械盘/exFAT 上元数据操作翻倍）。正确性不依赖原子性：
    标记文件落地之前，重跑总是从备份 zip 重新计算并覆盖该序列的全部文件，写了一半的文件也会被覆盖。"""
    with open(path, 'wb') as f:
        f.write(data)


def fix_combo(combo, data_root, backup_root, n_check=3):
    if os.path.exists(os.path.join(combo, MARKER)):
        return combo, 'skip(marked)', 0, 0.0
    zp = zip_path(combo, data_root, backup_root)
    bstat = backup_combo(combo, zp)
    n_files, worst = 0, 0.0
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith('.pkl') and '/' in n]
        for i, arc in enumerate(names):
            raw = z.read(arc)
            rows = pickle.loads(raw)
            new = fix_rows(rows)
            if arc.startswith('boxes_rgb/') and i < n_check + 1:
                worst = max(worst, check_rows(rows, new))
            write_atomic(os.path.join(combo, arc.replace('/', os.sep)), pickle.dumps(new, protocol=protocol_of(raw)))
            n_files += 1
        raw = z.read('drone_info.pkl')
        s = pickle.loads(raw)
        write_atomic(os.path.join(combo, 'drone_info.pkl'), pickle.dumps(fix_drone_info(s), protocol=protocol_of(raw)))
    if worst > 1e-4:
        raise RuntimeError('自检失败 %s：重排角点得到的 9 参数与 fix_box 差 %.2e' % (combo, worst))
    with open(os.path.join(combo, MARKER), 'w', encoding='utf-8') as f:
        json.dump({'version': VERSION, 'perm_new_from_old': PERM, 'drone_info': 'L<->W swapped',
                   'nose_yaw_deg_applied': dict(F.NOSE_YAW_DEG), 'scale_applied': False, 'backup_zip': zp,
                   'n_box_files': n_files, 'selfcheck_max_diff': worst, 'backup': bstat,
                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, ensure_ascii=False, indent=1)
    return combo, 'fixed', n_files, worst


def restore_combo(combo, data_root, backup_root):
    zp = zip_path(combo, data_root, backup_root)
    if not os.path.exists(zp):
        return combo, 'no-backup'
    with zipfile.ZipFile(zp) as z:
        for arc in z.namelist():
            if arc == 'BACKUP_COMPLETE':
                continue
            write_atomic(os.path.join(combo, arc.replace('/', os.sep)), z.read(arc))
    mk = os.path.join(combo, MARKER)
    if os.path.exists(mk):
        os.remove(mk)
    return combo, 'restored'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True, help='数据根目录（下面是 <场景>/carla_data/...）')
    ap.add_argument('--backup', required=True, help='备份根目录（与 roots 同盘）')
    ap.add_argument('--scenes', nargs='*', default=None, help='只处理这些场景目录名')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--restore', action='store_true')
    ap.add_argument('--log', default=None)
    args = ap.parse_args()
    log = open(args.log, 'a', encoding='utf-8') if args.log else None

    def say(s):
        print(s, flush=True)
        if log:
            log.write(s + '\n'); log.flush()

    for root in args.roots:
        combos = combos_under(root)
        if args.scenes:
            combos = [c for c in combos if os.path.relpath(c, root).split(os.sep)[0] in args.scenes]
        if args.limit:
            combos = combos[:args.limit]
        say('=== %s %s：%d 个序列 %s' % ('还原' if args.restore else '修正', root, len(combos), time.strftime('%H:%M:%S')))
        t0, done, nfix, worst_all, errors = time.time(), 0, 0, 0.0, []
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(restore_combo if args.restore else fix_combo, c, root, args.backup): c for c in combos}
            for fu in as_completed(futs):
                c = futs[fu]
                try:
                    res = fu.result()
                    done += 1
                    if not args.restore and res[1] == 'fixed':
                        nfix += 1
                        worst_all = max(worst_all, res[3])
                except Exception as e:
                    errors.append((c, repr(e)))
                    say('ERROR %s: %s\n%s' % (c, e, traceback.format_exc()))
                if done % 20 == 0 or done == len(combos):
                    say('  %d/%d 序列（本次修正 %d，自检最大差 %.1e，错误 %d）%.0f s' % (done, len(combos), nfix, worst_all, len(errors), time.time() - t0))
        say('=== 完成 %s：修正 %d，跳过 %d，错误 %d，自检最大差 %.1e，%.0f s' % (root, nfix, done - nfix, len(errors), worst_all, time.time() - t0))
        for c, e in errors:
            say('   ERR %s %s' % (c, e))


if __name__ == '__main__':
    main()
