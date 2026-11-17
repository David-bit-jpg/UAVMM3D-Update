# [review1113 copy of srcx/lib/mkcfg.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s20/lib] sys.path -> redirected sx; code unchanged
# -*- coding: utf-8 -*-
"""[srcx] write the two per-combo detector configs (tools/cfgs/models/uavdet_3d/camnorm/srcx/):
  real_<combo>.yaml = target real detector config (same as outreal_r34_geo_full_tr_out / mav6d_r34_geo_full_tr) with the SOURCE detector's
                      target-standardisation stats (DEPTH/SIZE mean/std: the decoder must match the stage-1 training)
  st_<combo>.yaml   = Module 3 self-training recipe (= st_outreal_r34_geo_full_tr_out = indoor st2nl075wa: rot weight 0, LR 2e-4, 3 epochs,
                      hflip-only augmentation) on index_pseudo_srcx_<combo>.pkl"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sx
D = sx.ROOT + '/tools/' + os.path.dirname(sx.REAL_CFG); os.makedirs(D, exist_ok=True)
dm, ds = sx.S['depth']; sm, ss = sx.S['size']
open(sx.ROOT + '/tools/' + sx.REAL_CFG, 'w').write(
    "# [srcx] %s: target %s real detector config + decoder stats of source %s (written by srcx/lib/mkcfg.py)\n"
    "_BASE_CONFIG_: %s\nDATA_CONFIG:\n    DEPTH_MEAN: %r\n    DEPTH_STD: %r\n    SIZE_MEAN: %r\n    SIZE_STD: %r\n"
    % (sx.COMBO, sx.TGT, sx.SRC, sx.T['det_base'], dm, ds, list(sm), list(ss)))
open(sx.ROOT + '/tools/' + sx.ST_CFG, 'w').write(
    "# [srcx] %s: Module 3 detector self-training (recipe = st_outreal_r34_geo_full_tr_out = indoor st2nl075wa)\n"
    "_BASE_CONFIG_: %s\nDATA_CONFIG:\n    INDEX_FILE: 'index_pseudo_%s.pkl'\n"
    "    AUG: {hflip: 0.5, scale: [1.0, 1.0], photometric: false, noise: 0.0, blur: 0.0, blur_prob: 0.5}\n"
    "MODEL:\n    DENSE_HEAD_2D:\n      LOSS_WEIGHTS: {hm: 1.0, center_res: 2.0, center_dis: 8.0, dim: 2.0, rot: 0.0, size2d: 4.0}\n"
    "OPTIMIZATION:\n    LR: 0.0002\n    NUM_EPOCHS: 3\n" % (sx.COMBO, sx.REAL_CFG, sx.PSEUDO_TAG))
print('configs', sx.REAL_CFG, sx.ST_CFG)
