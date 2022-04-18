# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/4/16 21:49
# @brief      : mixup实现
"""
import numpy as np
import torch


def mixed_data(x, y, alpha=1.0, device=True):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


