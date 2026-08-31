#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3,4,5 nohup python -m torch.distributed.launch --nproc_per_node=3 --master_port=29512 train_laam6d.py --launcher pytorch > log.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 train.py --launcher pytorch
