# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/4/4 11:29 PM
# @brief      : flower config file
"""

import torch
from torchvision import transforms
from easydict import EasyDict

cfg = EasyDict()
cfg.dataset_dir = r'/User/zhuyue/Code/datasets/flowers102'
cfg.train_bs = 32
cfg.valid_bs = 64
cfg.workers = 0

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [30, 45]
cfg.max_epoch = 50

cfg.log_interval = 10

norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万图像统计得来
norm_std = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train = transforms.Compose([
    transforms.Resize((256)),  # (256, 256) 区别； （256） 最短边256
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normTransform,
])
cfg.transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])