# -*- coding: utf-8 -*-
"""wf21/strict 第 2 步修订：P0 不开热图时序背景后，纯仿真检测器在真实 train 上的最高分检测常落在静态杂物上
（种子 2–4：几乎每帧都过置信 0.5，但与本帧全严格伪标签位置一致的只有 25–33%；种子 0/1 只有 45% 的帧过 0.5）。
旧版（带时序背景）裁窗里约 2/3 与伪标签一致。为了不让真实适应段吃进大量"没有无人机"的裁窗，
真实训练裁窗改为：P0_s{j} 单帧最高分检测（置信 ≥0.5、投影宽 ∈ [6,170]）**且**与本帧全严格单帧伪标签 st2sfnb 的位置相差 < 0.5 m 的那些。
（判断只用本帧的伪标签，不读真值、不看其他帧。）直接从 extract_nb.py 的输出里挑行，不重读图像。
输出 selftrain/real_s{j}_nbm{rgb,scal}.npy 与 real_s{j}_nbmdp_sfnb.npy（全部有伪目标）。"""
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
for j in range(5):
    dp = np.load('%s/selftrain/real_s%d_nbdp_sfnb.npy' % (G, j))
    m = np.isfinite(dp)
    rgb = np.load('%s/selftrain/real_s%d_nbrgb.npy' % (G, j), mmap_mode='r')
    sc = np.load('%s/selftrain/real_s%d_nbscal.npy' % (G, j))
    assert len(rgb) == len(sc) == len(dp)
    np.save('%s/selftrain/real_s%d_nbmrgb.npy' % (G, j), np.ascontiguousarray(rgb[m]))
    np.save('%s/selftrain/real_s%d_nbmscal.npy' % (G, j), sc[m])
    np.save('%s/selftrain/real_s%d_nbmdp_sfnb.npy' % (G, j), dp[m])
    old = np.load('%s/selftrain/real_s%d_dpseudo.npy' % (G, j))
    print('j=%d：单帧裁窗 %d → 与伪标签一致 %d（旧带背景版裁窗 %d，其中有伪目标 %d）' % (j, len(dp), int(m.sum()), len(old), int(np.isfinite(old).sum())), flush=True)
