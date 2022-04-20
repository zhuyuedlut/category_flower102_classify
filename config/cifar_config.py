# -*- coding: utf-8 -*-
"""
# @file name  : flower_config.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-04-23
# @brief      : cifar-10分类参数配置
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.transforms as transforms
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.pb = True  # 是否采用渐进式采样
cfg.mixup = False  # 是否采用mixup
cfg.mixup_alpha = 1.  # beta分布的参数. beta分布是一组定义在(0,1) 区间的连续概率分布。
cfg.label_smooth = False  # 是否采用标签平滑
cfg.label_smooth_eps = 0.01  # 标签平滑超参数 eps

cfg.train_bs = 128
cfg.valid_bs = 128
cfg.workers = 4

cfg.lr_init = 0.1
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [160, 180]
cfg.max_epoch = 200
 
cfg.log_interval = 20

# 数据预处理设置
cfg.norm_mean = [0.4914, 0.4822, 0.4465]    # cifar10 from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
cfg.norm_std = [0.2023, 0.1994, 0.2010]

normTransform = transforms.Normalize(cfg.norm_mean, cfg.norm_std)
cfg.transforms_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normTransform
])






