# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] sim + real-pseudo concatenation for OUR detector under the baselines' +ST / +PL protocol.
Baselines (D:/UAV3D_baselines_pack/scripts/run.sh): train(<out>, sim labels.pkl, pseudo.pkl) = one label list = plain concatenation,
uniform sampling, same epochs as their sim-only training. Here: torch ConcatDataset of
  part 0 = the sim-only recipe's own MMCache_Det_Dataset (DATA_CONFIG as given = sim cache, sim NORM, recipe AUG / VIEW_AUG, ROT_SYMMETRY ...)
  part 1 = the same DATA_CONFIG with only these keys replaced from DATA_CONFIG.MIX_REAL: DATA_PATH (real target cache copy), INDEX_FILE
           (pseudo-label index written by make_labels.py), NORM_MEAN / NORM_STD (the target's per-domain input normalisation = what the
           detector always uses on real images at test time), MIN_RGB_VIS 0 (no vis_score.npy for real caches; sim keeps its filter).
           The real caches are 512x288 npy stores -> VIEW_AUG crop/zoom does not apply to them (it needs the jpeg store); every other
           augmentation (AUG) is the recipe's, identical for both parts.
DataLoader shuffle=True over the concatenation = uniform sampling over sim + pseudo frames.
Evaluation (training=False) returns the plain sim dataset (train.py's --val_split sanity on sim test, as in stage1_det.sh).
Registered into uavdet3d.datasets.__all__ by ourdet_train.py (no repository file is modified)."""
import copy
import numpy as np
from torch.utils.data import Dataset
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset

NAME = 'OurdetConcat'


def _part_cfg(dataset_cfg, real):
    c = copy.deepcopy(dataset_cfg)
    c.DATASET = 'MMCache_Det_Dataset'
    c.pop('MIX_REAL', None)
    if real is not None:
        for k in ('DATA_PATH', 'INDEX_FILE', 'NORM_MEAN', 'NORM_STD'):
            c[k] = real[k]
        c.MIN_RGB_VIS = 0.0
    return c


class OurdetConcatDataset(Dataset):
    def __init__(self, sim, real):
        self.parts = [sim, real]
        self.cum = np.cumsum([len(sim), len(real)])

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, item):
        p = int(np.searchsorted(self.cum, item, side='right'))
        return self.parts[p][item - (0 if p == 0 else int(self.cum[p - 1]))]

    def collate_batch(self, batch_list, _unused=False):
        return self.parts[0].collate_batch(batch_list, _unused)

    def __getattr__(self, name):      # model building reads dataset_cfg / im_num / data_pre_processor ... -> the sim part (same recipe)
        if name == 'parts' or name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.parts[0], name)


def build(dataset_cfg, root_path=None, training=True, logger=None):
    sim = MMCache_Det_Dataset(dataset_cfg=_part_cfg(dataset_cfg, None), training=training, root_path=root_path, logger=logger)
    if not training:
        return sim
    real = MMCache_Det_Dataset(dataset_cfg=_part_cfg(dataset_cfg, dataset_cfg.MIX_REAL), training=True, root_path=None, logger=logger)
    for m in (real.metas[int(i)] for i in real.valid_idx):
        assert not m.get('pseudo', False), 'pseudo frames are written as plain labels (all heads supervised, like the baselines)'
    if logger is not None:
        logger.info('[ourdet_baseproto] concat: sim %d frames (%s) + real pseudo %d frames (%s/train/%s) = %d, uniform sampling'
                    % (len(sim), sim.split_dir, len(real), dataset_cfg.MIX_REAL.DATA_PATH, dataset_cfg.MIX_REAL.INDEX_FILE, len(sim) + len(real)))
    return OurdetConcatDataset(sim, real)
