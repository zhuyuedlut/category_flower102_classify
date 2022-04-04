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