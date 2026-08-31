
import cv2
from tools.api import Open3DUAVDet
import numpy as np
import os

#im_path = r'G:\dataset\MAV6D\mavic2\JPEGImages\01\0101\1649917993418355227.jpg'
im_path_root = r'../dataset/LAA3D_real_data/drone_swarm_1/0001'

all_frame = os.listdir(im_path_root)
all_frame = sorted(all_frame)

detector = Open3DUAVDet(config='cfgs/models/uavdet_3d/laa3d/centerdet.yaml',
                 mode = 'pose_estimation',
                 model_path=r"../output/models/uavdet_3d/laa3d/centerdet/default/ckpt/checkpoint_epoch_10.pth")


for k_i in all_frame:

    im_path = os.path.join(im_path_root, k_i)

    image = cv2.imread(im_path)

    pred_dict = detector(image)

    image = detector.plot_on_image(image, pred_dict['pred_box_9dof'])
    cv2.imshow('im', image)
    cv2.waitKey(0)
