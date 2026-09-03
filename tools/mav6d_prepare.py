# -*- coding: utf-8 -*-
"""MAV6D 数据集体检 + split 生成 + GT 投影可视化。

MAV6D 官方下载下来是 RGB 图 + 标签 + mask，不带 split，目录命名也不一定和
本仓库 dataset 的期望完全一致。这个脚本负责在真正开跑之前把这些对齐，
并且把几个"会静默出错"的地方验一遍。

用法：
    # 1) 体检：看目录结构、帧数、距离分布、GT 投影是否落在画面内、相机是否固定
    python tools/mav6d_prepare.py check --root E:/dataset/MAV6D

    # 2) 生成 split（按序列切分，避免相邻帧泄漏）
    python tools/mav6d_prepare.py split --root E:/dataset/MAV6D --train-ratio 0.8

    # 3) 把 GT 3D 框投影回图上，肉眼确认标注和硬编码外参对不对
    python tools/mav6d_prepare.py vis --root E:/dataset/MAV6D --num 8 --out ./mav6d_vis

标签格式（每行 16 个数）：
    ts_cam  t_x t_y t_z  r_x r_y r_z r_w   ts_uav  t_x t_y t_z  r_x r_y r_z r_w
前 8 个是相机在 VICON 系下的位姿，后 8 个是 MAV 的。
本仓库的 read_truth_Rt 只用后者，相机位姿被一个硬编码矩阵代替 —— 所以这里
要专门检查"相机位姿是否真的全程不变"。
"""
import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from scipy.spatial.transform import Rotation as R

# 与 uavdet3d/datasets/mav6d/mav6d_utils.py 保持一致
INTRINSIC = np.array([[1979.4, 0.3984, 976.8189],
                      [0.0, 1979.1, 533.9717],
                      [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
CAM2VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                      [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                      [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                      [0, 0, 0, 1]])
RAW_W, RAW_H = 1920, 1080
OB_SIZE = np.array([0.240, 0.240, 0.230])

IMG_DIR_CANDIDATES = ['JPEGImages', 'images', 'image', 'JPEGimages', 'rgb']
LBL_DIR_CANDIDATES = ['labels', 'label']
IMG_EXTS = ('.jpg', '.jpeg', '.png')


# --------------------------------------------------------------------------- #
# 目录发现
# --------------------------------------------------------------------------- #
def _pick_dir(base, candidates):
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return c
    return None


def discover_classes(root):
    """MAV6D 根下每个子目录是一个目标型号（phantom4 / mavic2 ...）。

    如果 root 自己就长得像一个型号目录（直接含 JPEGImages），就把它当成单型号。
    """
    if _pick_dir(root, IMG_DIR_CANDIDATES):
        return ['']
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and _pick_dir(p, IMG_DIR_CANDIDATES):
            out.append(d)
    return out


def iter_sequences(cls_root, im_dir):
    """产出 (scene, seq, [frame 文件名...])，按数值顺序排好。"""
    im_root = os.path.join(cls_root, im_dir)
    for scene in sorted(os.listdir(im_root)):
        scene_p = os.path.join(im_root, scene)
        if not os.path.isdir(scene_p):
            continue
        for seq in sorted(os.listdir(scene_p)):
            seq_p = os.path.join(scene_p, seq)
            if not os.path.isdir(seq_p):
                continue
            frames = [f for f in os.listdir(seq_p) if f.lower().endswith(IMG_EXTS)]

            def _key(f):
                stem = os.path.splitext(f)[0]
                try:
                    return (0, float(stem))
                except ValueError:
                    return (1, stem)

            frames.sort(key=_key)
            yield scene, seq, frames


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #
def parse_label(path):
    """返回 (cam_pose7, uav_pose7)，各是 [tx,ty,tz,rx,ry,rz,rw]；解析失败返回 None。"""
    try:
        if os.path.getsize(path) == 0:
            return None
        with open(path, 'r') as f:
            line = f.readline().strip()
        vals = [float(x) for x in line.split()]
    except Exception:
        return None
    if len(vals) < 16:
        return None
    return np.array(vals[1:8]), np.array(vals[9:16])


def uav_pose_to_camera(uav_pose7):
    """按仓库里 read_truth_Rt 的做法，用硬编码外参把 VICON 系位姿转到相机系。"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(uav_pose7[3:7]).as_matrix()
    T[:3, 3] = uav_pose7[0:3]
    return CAM2VICON @ T


def project(xyz):
    if cv2 is not None:
        uv, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(1, 1, 3),
                                  np.zeros(3), np.zeros(3),
                                  INTRINSIC, DISTORTION.reshape(1, -1))
        return uv.reshape(2)
    p = INTRINSIC @ np.asarray(xyz, np.float64)
    return p[:2] / p[2]


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args):
    root = args.root
    if not os.path.isdir(root):
        print('!! 根目录不存在: %s' % root)
        return 1

    classes = discover_classes(root)
    if not classes:
        print('!! 在 %s 下没找到任何含图像目录的型号文件夹' % root)
        print('   期望结构: <root>/<型号>/JPEGImages/<scene>/<seq>/*.jpg')
        print('   实际内容:', sorted(os.listdir(root))[:20])
        return 1

    print('根目录     : %s' % root)
    print('目标型号   : %s' % ([c or '(根目录本身)' for c in classes]))

    problems = []
    for cls in classes:
        cls_root = os.path.join(root, cls) if cls else root
        im_dir = _pick_dir(cls_root, IMG_DIR_CANDIDATES)
        lb_dir = _pick_dir(cls_root, LBL_DIR_CANDIDATES)

        print('\n' + '=' * 68)
        print('型号: %s' % (cls or '(根目录本身)'))
        print('  图像目录: %s' % im_dir)
        print('  标签目录: %s' % lb_dir)
        if lb_dir is None:
            problems.append('%s: 找不到标签目录 (labels/label)' % cls)
            continue
        if lb_dir != 'labels':
            problems.append(
                "%s: 标签目录叫 '%s'，但 mav6d_*_dataset.py 里写死的是 'labels'，"
                "需要改名或改代码" % (cls, lb_dir))
        if im_dir != 'JPEGImages':
            problems.append(
                "%s: 图像目录叫 '%s'，但代码写死的是 'JPEGImages'" % (cls, im_dir))

        n_scene = n_seq = n_frame = 0
        n_lbl_ok = n_lbl_missing = n_lbl_bad = 0
        depths, uvs = [], []
        cam_poses = []
        ext_mismatch = 0
        non_jpg = 0

        for scene, seq, frames in iter_sequences(cls_root, im_dir):
            n_seq += 1
            n_frame += len(frames)
            for fr in frames:
                if not fr.lower().endswith('.jpg'):
                    non_jpg += 1
                stem = os.path.splitext(fr)[0]
                lp = os.path.join(cls_root, lb_dir, scene, seq, stem + '.txt')
                if not os.path.exists(lp):
                    n_lbl_missing += 1
                    continue
                parsed = parse_label(lp)
                if parsed is None:
                    n_lbl_bad += 1
                    continue
                n_lbl_ok += 1
                cam_pose, uav_pose = parsed
                cam_poses.append(cam_pose)
                T = uav_pose_to_camera(uav_pose)
                xyz = T[:3, 3]
                depths.append(xyz[2])
                if xyz[2] > 1e-6:
                    uvs.append(project(xyz))

        scenes = set()
        for scene, seq, _ in iter_sequences(cls_root, im_dir):
            scenes.add(scene)
        n_scene = len(scenes)

        print('  场景 %d  序列 %d  帧 %d' % (n_scene, n_seq, n_frame))
        print('  标签: 可用 %d / 缺失 %d / 解析失败 %d' % (n_lbl_ok, n_lbl_missing, n_lbl_bad))
        if non_jpg:
            print('  提示: 有 %d 张图不是 .jpg。dataset 推标签名已改用 splitext，'
                  '可以正常读取；仅与官方说明的 JPEGImages 命名略有出入。' % non_jpg)

        if n_lbl_ok == 0:
            problems.append('%s: 没有一条标签能解析成功' % cls)
            continue

        depths = np.array(depths)
        print('  相机系深度 Z (m): min %.3f  中位 %.3f  max %.3f' %
              (depths.min(), np.median(depths), depths.max()))
        n_behind = int((depths <= 0).sum())
        if n_behind:
            problems.append('%s: 有 %d 帧的目标在相机背后 (Z<=0)，'
                            '说明硬编码外参可能不适用于这批数据' % (cls, n_behind))
        if depths.max() > 8:
            problems.append('%s: 最大深度 %.2f m 超过了 eval 里默认的 max_dis=8，'
                            '超出部分会被截断' % (cls, depths.max()))

        uvs = np.array(uvs)
        if len(uvs):
            inside = ((uvs[:, 0] >= 0) & (uvs[:, 0] < RAW_W) &
                      (uvs[:, 1] >= 0) & (uvs[:, 1] < RAW_H))
            pct = 100.0 * inside.mean()
            print('  GT 投影落在 %dx%d 画面内: %.1f%%' % (RAW_W, RAW_H, pct))
            if pct < 90:
                problems.append('%s: 只有 %.1f%% 的 GT 投影落在画面内，'
                                '强烈提示外参/内参和这批数据对不上' % (cls, pct))

        # 关键检查：标签里逐帧记录的相机位姿是否恒定。
        # 判据要用物理量，不能直接卡四元数分量的 std —— 分量 std 1e-3 只相当于
        # 约 0.1°，属于动捕噪声，卡 1e-3 会把静止相机误判成在动。
        cam_poses = np.array(cam_poses)
        if np.all(np.abs(cam_poses) < 1e-9):
            print('  标签内相机位姿字段全是 0（占位），只能依赖硬编码外参')
        else:
            t_span_mm = 1000.0 * np.linalg.norm(
                cam_poses[:, 0:3] - cam_poses[:, 0:3].mean(axis=0), axis=1).max()
            rots = R.from_quat(cam_poses[:, 3:7])
            mean_rot = R.from_quat(np.mean(cam_poses[:, 3:7], axis=0) /
                                   np.linalg.norm(np.mean(cam_poses[:, 3:7], axis=0)))
            ang_span_deg = np.degrees((mean_rot.inv() * rots).magnitude()).max()
            print('  标签内相机位姿波动: 平移最大偏离 %.3f mm, 旋转最大偏离 %.4f°'
                  % (t_span_mm, ang_span_deg))
            # 把角度波动折算成中位深度处的横向位置误差，才好判断要不要管
            med_z = float(np.median(depths))
            induced_mm = 1000.0 * med_z * np.tan(np.radians(ang_span_deg)) + t_span_mm
            print('  -> 折算到中位深度 %.2f m 处，最坏情况位置偏差约 %.1f mm'
                  % (med_z, induced_mm))
            if induced_mm > 100.0:
                problems.append(
                    '%s: 相机位姿波动折算到 %.2f m 处已达 %.0f mm，超过目标尺寸的 1/3，'
                    'read_truth_Rt 的写死外参不再够用，应改成逐帧用标签里的相机位姿'
                    % (cls, med_z, induced_mm))
            else:
                print('     相对目标尺寸(约 340 mm)可忽略，沿用官方的写死外参即可')

    print('\n' + '=' * 68)
    if problems:
        print('发现 %d 个需要处理的问题：' % len(problems))
        for i, p in enumerate(problems, 1):
            print('  %d. %s' % (i, p))
    else:
        print('未发现结构性问题，可以直接生成 split 开跑。')
    return 0


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def _filter_official_split(cls_root, im_dir, lb_dir, name):
    """把官方 split 过滤成"图和标签都真实存在"的子集。

    官方 train.txt/test.txt 里是采集机上的绝对路径，且覆盖全量帧；
    而 OneDrive 打包下载有 1 万文件上限，实际拿到的往往只是其中一部分。
    直接拿官方 split 去训会大面积找不到文件，所以这里按磁盘实际情况过滤，
    同时保留官方的 train/test 归属（不重新随机切，保证与论文口径可比）。
    """
    p = os.path.join(cls_root, 'split', name + '.txt')
    if not os.path.exists(p):
        return None
    kept, missing, dup = [], 0, 0
    seen = set()
    with open(p, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('\\', '/').split('/')[-3:]
            if len(parts) < 3:
                continue
            scene, seq, frame = parts
            rel = '%s/%s/%s' % (scene, seq, frame)
            # 官方 mavic2/train.txt 里每帧重复了 3 次（35379 行 / 11793 唯一帧），
            # 而 phantom4 没有重复。照单全收会让 mavic2 被 3 倍过采样、两类失衡。
            if rel in seen:
                dup += 1
                continue
            seen.add(rel)
            stem = os.path.splitext(frame)[0]
            ip = os.path.join(cls_root, im_dir, scene, seq, frame)
            lp = os.path.join(cls_root, lb_dir, scene, seq, stem + '.txt')
            if os.path.exists(ip) and os.path.exists(lp):
                kept.append(rel)
            else:
                missing += 1
    return kept, missing, dup


def cmd_split(args):
    root = args.root
    classes = discover_classes(root)
    if not classes:
        print('!! 没找到型号目录')
        return 1

    # 优先沿用官方划分（过滤到实际存在的帧）
    if args.use_official:
        any_official = False
        for cls in classes:
            cls_root = os.path.join(root, cls) if cls else root
            im_dir = _pick_dir(cls_root, IMG_DIR_CANDIDATES)
            lb_dir = _pick_dir(cls_root, LBL_DIR_CANDIDATES)
            if im_dir is None or lb_dir is None:
                continue
            res = {n: _filter_official_split(cls_root, im_dir, lb_dir, n)
                   for n in ('train', 'test')}
            if not any(res.values()):
                continue
            any_official = True
            print('[%s] 沿用官方划分，过滤到实际存在的帧：' % (cls or 'root'))
            for n in ('train', 'test'):
                if res[n] is None:
                    print('  %-5s 官方文件不存在，跳过' % n)
                    continue
                kept, missing, dup = res[n]
                out = os.path.join(cls_root, 'split', n + '.txt')
                with open(out, 'w') as f:
                    f.write('\n'.join(kept) + ('\n' if kept else ''))
                uniq = len(kept) + missing
                print('  %-5s 保留 %6d / 官方唯一 %6d 帧 (本地缺 %6d, 去重丢弃 %6d) -> %s'
                      % (n, len(kept), uniq, missing, dup, out))
        if any_official:
            print('\n提示: 若想改成按序列重新随机切分，加 --no-use-official')
            return 0
        print('未找到任何官方 split，改为按序列重新切分')

    rng = np.random.RandomState(args.seed)

    for cls in classes:
        cls_root = os.path.join(root, cls) if cls else root
        im_dir = _pick_dir(cls_root, IMG_DIR_CANDIDATES)

        seqs = [(scene, seq, frames) for scene, seq, frames in iter_sequences(cls_root, im_dir)]
        if not seqs:
            print('[%s] 没有序列，跳过' % (cls or 'root'))
            continue

        # 按序列切分：同一段视频不会同时出现在 train 和 test，
        # 否则相邻帧几乎一样，指标会虚高
        idx = np.arange(len(seqs))
        rng.shuffle(idx)
        n_train = max(1, int(round(len(seqs) * args.train_ratio)))
        train_idx, test_idx = set(idx[:n_train].tolist()), set(idx[n_train:].tolist())
        if not test_idx:                      # 只有一条序列时退化成按帧切
            print('[%s] 只有 1 条序列，退化为按帧切分（注意会有帧间泄漏）'
                  % (cls or 'root'))
            scene, seq, frames = seqs[0]
            k = int(len(frames) * args.train_ratio)
            train_lines = ['%s/%s/%s' % (scene, seq, f) for f in frames[:k]]
            test_lines = ['%s/%s/%s' % (scene, seq, f) for f in frames[k:]]
        else:
            train_lines, test_lines = [], []
            for i, (scene, seq, frames) in enumerate(seqs):
                tgt = train_lines if i in train_idx else test_lines
                tgt += ['%s/%s/%s' % (scene, seq, f) for f in frames]

        split_dir = os.path.join(cls_root, 'split')
        os.makedirs(split_dir, exist_ok=True)
        for name, lines in [('train', train_lines), ('test', test_lines)]:
            p = os.path.join(split_dir, name + '.txt')
            with open(p, 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            print('[%s] %-5s %6d 帧 -> %s' % (cls or 'root', name, len(lines), p))
    return 0


# --------------------------------------------------------------------------- #
# vis
# --------------------------------------------------------------------------- #
def cmd_vis(args):
    if cv2 is None:
        print('!! 需要 opencv')
        return 1

    root = args.root
    os.makedirs(args.out, exist_ok=True)
    classes = discover_classes(root)

    corners_local = np.array([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    n_done = 0
    for cls in classes:
        cls_root = os.path.join(root, cls) if cls else root
        im_dir = _pick_dir(cls_root, IMG_DIR_CANDIDATES)
        lb_dir = _pick_dir(cls_root, LBL_DIR_CANDIDATES)
        if lb_dir is None:
            continue

        for scene, seq, frames in iter_sequences(cls_root, im_dir):
            step = max(1, len(frames) // max(1, args.num))
            for fr in frames[::step]:
                if n_done >= args.num:
                    break
                stem = os.path.splitext(fr)[0]
                lp = os.path.join(cls_root, lb_dir, scene, seq, stem + '.txt')
                parsed = parse_label(lp)
                if parsed is None:
                    continue
                img = cv2.imread(os.path.join(cls_root, im_dir, scene, seq, fr))
                if img is None:
                    continue

                T = uav_pose_to_camera(parsed[1])
                pts = corners_local * OB_SIZE
                pts = pts @ T[:3, :3].T + T[:3, 3]
                uv, _ = cv2.projectPoints(pts.astype(np.float64), np.zeros(3), np.zeros(3),
                                          INTRINSIC, DISTORTION.reshape(1, -1))
                uv = uv.reshape(-1, 2).astype(int)
                for a, b in edges:
                    cv2.line(img, tuple(uv[a]), tuple(uv[b]), (0, 0, 255), 2)
                c = project(T[:3, 3]).astype(int)
                cv2.circle(img, tuple(c), 6, (0, 255, 0), -1)
                cv2.putText(img, 'Z=%.2fm  %s/%s/%s' % (T[2, 3], scene, seq, stem),
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                op = os.path.join(args.out, '%s_%s_%s_%s.jpg' %
                                  (cls or 'root', scene, seq, stem))
                cv2.imwrite(op, img)
                print('写出', op)
                n_done += 1
            if n_done >= args.num:
                break
        if n_done >= args.num:
            break

    print('共写出 %d 张。红框贴在无人机上 = 标注和硬编码外参都对；'
          '整体偏移 = 外参或 0.2m 中心偏移需要重标。' % n_done)
    return 0


# --------------------------------------------------------------------------- #
# organize：把官方下载解压后的目录整理成本仓库期望的布局
# --------------------------------------------------------------------------- #
KNOWN_CLASSES = ['phantom4', 'mavic2', 'mavic3', 'avata', 'm300', 'mavicmini', 'mavic']


def _looks_like_mav6d_label(path):
    """判断一个 txt 是不是 MAV6D 标签：单行 16 个浮点数。"""
    try:
        if os.path.getsize(path) == 0 or os.path.getsize(path) > 4096:
            return False
        with open(path, 'r', errors='ignore') as f:
            vals = f.readline().strip().split()
        if len(vals) < 16:
            return False
        [float(x) for x in vals[:16]]
        return True
    except Exception:
        return False


def _infer_class(rel_parts, default):
    """从路径片段里猜目标型号。"""
    for p in rel_parts:
        low = p.lower().replace('-', '').replace('_', '')
        for c in KNOWN_CLASSES:
            if c in low:
                return c
    return default


def cmd_organize(args):
    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        print('!! 源目录不存在: %s' % src)
        return 1

    # 1) 收集所有图像，以及所有像 MAV6D 标签的 txt（按 stem 建索引）
    images = []          # (绝对路径, 相对 src 的片段列表)
    labels = {}          # stem -> [绝对路径...]
    for dirpath, _dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src).replace('\\', '/')
        parts = [] if rel == '.' else rel.split('/')
        low_parts = [p.lower() for p in parts]
        is_mask_dir = any('mask' in p for p in low_parts)
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            ap_ = os.path.join(dirpath, fn)
            if ext.lower() in IMG_EXTS:
                if is_mask_dir:          # mask 图本流水线用不到，跳过
                    continue
                images.append((ap_, parts, stem, ext))
            elif ext.lower() == '.txt' and _looks_like_mav6d_label(ap_):
                labels.setdefault(stem, []).append(ap_)

    print('扫描 %s' % src)
    print('  图像 %d 张（已跳过 mask 目录）' % len(images))
    print('  MAV6D 格式标签 %d 个 stem' % len(labels))
    if not images or not labels:
        print('!! 没找到成对的图像/标签，检查一下解压出来的目录')
        return 1

    # 2) 为每张图配标签、推型号与 scene/seq
    plan = []            # (src_img, src_lbl, cls, scene, seq, stem, ext)
    n_nolabel = 0
    cls_count = {}
    for ap_, parts, stem, ext in images:
        cands = labels.get(stem)
        if not cands:
            n_nolabel += 1
            continue
        # 同 stem 有多个标签时，选路径前缀与图像最接近的那个
        best = max(cands, key=lambda p: len(os.path.commonprefix([p, ap_])))
        cls = _infer_class(parts, args.default_class)
        # scene / seq 取图像所在目录的上两级；不足则补 default
        tail = [p for p in parts if p.lower() not in IMG_DIR_CANDIDATES_LOWER]
        seq = tail[-1] if len(tail) >= 1 else 'seq01'
        scene = tail[-2] if len(tail) >= 2 else 'scene01'
        plan.append((ap_, best, cls, scene, seq, stem, ext))
        cls_count[cls] = cls_count.get(cls, 0) + 1

    print('\n配对成功 %d 张，缺标签 %d 张' % (len(plan), n_nolabel))
    print('型号分布: %s' % cls_count)
    if not plan:
        return 1

    # 3) 打印计划
    print('\n目标布局: <dst>/<型号>/JPEGImages/<scene>/<seq>/<stem><ext>')
    print('          <dst>/<型号>/labels/<scene>/<seq>/<stem>.txt')
    print('\n前几条：')
    for ap_, lb, cls, scene, seq, stem, ext in plan[:5]:
        print('  %s' % os.path.relpath(ap_, src))
        print('    -> %s' % os.path.join(cls, 'JPEGImages', scene, seq, stem + ext))
    exts = sorted({e.lower() for _, _, _, _, _, _, e in plan})
    if exts != ['.jpg']:
        print('\n注意: 图像扩展名为 %s。dataset 里推标签名已改用 splitext，'
              '所以非 .jpg 也能跑；如需严格贴合官方说明可自行转成 jpg。' % exts)

    if not args.apply:
        print('\n[dry-run] 以上仅为计划。确认无误后加 --apply 执行。')
        return 0

    # 4) 执行
    import shutil
    op = shutil.move if args.mode == 'move' else shutil.copy2
    n = 0
    for ap_, lb, cls, scene, seq, stem, ext in plan:
        im_dir = os.path.join(dst, cls, 'JPEGImages', scene, seq)
        lb_dir = os.path.join(dst, cls, 'labels', scene, seq)
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(lb_dir, exist_ok=True)
        di = os.path.join(im_dir, stem + ext)
        dl = os.path.join(lb_dir, stem + '.txt')
        if not os.path.exists(di):
            op(ap_, di)
        if not os.path.exists(dl):
            op(lb, dl)
        n += 1
        if n % 2000 == 0:
            print('  已处理 %d/%d' % (n, len(plan)))

    # 官方自带的 train.txt / test.txt 放在 <型号>/ 下，而 dataset 读的是
    # <型号>/split/{train,test}.txt。存在就搬过去，省得再自己切分。
    # 官方 split 里是绝对路径，dataset 用 split('/')[-3:] 取 scene/seq/frame，
    # 所以原样复制即可。
    n_split = 0
    for dirpath, _dn, filenames in os.walk(src):
        for fn in filenames:
            if fn.lower() not in ('train.txt', 'test.txt', 'val.txt'):
                continue
            rel = os.path.relpath(dirpath, src).replace('\\', '/')
            cls = _infer_class([] if rel == '.' else rel.split('/'), args.default_class)
            out = os.path.join(dst, cls, 'split')
            os.makedirs(out, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(out, fn.lower()))
            print('  官方 split: %s -> %s' % (fn, os.path.join(cls, 'split', fn.lower())))
            n_split += 1

    print('\n完成，%d 帧已整理到 %s（另搬运 %d 个官方 split 文件）' % (n, dst, n_split))
    print('接下来：')
    print('  python tools/mav6d_prepare.py check --root %s' % dst)
    print('  python tools/mav6d_prepare.py split --root %s --train-ratio 0.8' % dst)
    return 0


IMG_DIR_CANDIDATES_LOWER = {c.lower() for c in IMG_DIR_CANDIDATES} | {'label', 'labels', 'mask', 'masks'}


def main():
    ap = argparse.ArgumentParser(description='MAV6D 数据整理 / 体检 / split 生成 / 可视化')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('organize', help='把官方下载解压后的目录整理成本仓库期望的布局')
    p.add_argument('--src', required=True, help='解压后的 MAV6D 目录')
    p.add_argument('--dst', required=True, help='整理到哪里，例如 F:/dataset/MAV6D')
    p.add_argument('--apply', action='store_true', help='不加则只打印计划(dry-run)')
    p.add_argument('--mode', choices=['copy', 'move'], default='copy')
    p.add_argument('--default-class', default='phantom4',
                   help='路径里认不出型号时用这个名字')
    p.set_defaults(func=cmd_organize)

    p = sub.add_parser('check', help='体检数据集结构与几何一致性')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('split', help='生成 train/test split')
    p.add_argument('--root', required=True)
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use-official', dest='use_official', action='store_true', default=True,
                   help='默认开：沿用 <型号>/split/ 下的官方 train/test，'
                        '只过滤掉磁盘上不存在的帧（保持与论文口径可比）')
    p.add_argument('--no-use-official', dest='use_official', action='store_false',
                   help='忽略官方划分，按序列重新随机切分')
    p.set_defaults(func=cmd_split)

    p = sub.add_parser('vis', help='把 GT 3D 框投影回图上')
    p.add_argument('--root', required=True)
    p.add_argument('--num', type=int, default=8)
    p.add_argument('--out', default='./mav6d_vis')
    p.set_defaults(func=cmd_vis)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
