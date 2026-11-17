import torch
import numpy as np
from torch.utils.data import DataLoader
# [UAV port] dataset imports made lazy (nuscenes/waymo deps are not installed; KITTI code untouched)


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4):
    # perpare dataset
    if cfg['type'] == 'UAV':  # [UAV port] unified cmpfull labels.pkl; no test loader (selection done by predict.py on sim_val)
        from lib.datasets.uav.uav_dataset import UAV_Dataset
        train_set = UAV_Dataset(split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set, batch_size=cfg['batch_size'], num_workers=workers,
                                  worker_init_fn=my_worker_init_fn, shuffle=True, pin_memory=False, drop_last=False,
                                  persistent_workers=workers > 0)
        return train_loader, None
    if cfg['type'] == 'KITTI':
        from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    elif cfg['type'] == 'NUSC':
        from lib.datasets.nuscenes.nusc_dataset import NUSC_Dataset
        train_set = NUSC_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = NUSC_Dataset(split=cfg['test_split'], cfg=cfg)
    elif cfg['type'] == 'WAYMO':
        from lib.datasets.waymo.waymo_dataset import Waymo_Dataset
        train_set = Waymo_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = Waymo_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
