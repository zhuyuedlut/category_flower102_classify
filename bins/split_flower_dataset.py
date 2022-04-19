# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/4/4 11:24 PM
# @brief      : 将flower数据集分为train valid test
"""
import os
import shutil
import random

from config import flower_config as cfg


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.mkdir(my_dir)


def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, data_dir)
    print("{} dataset, copy {} imgs to {}".format(setname, len(imgs), data_dir))


if __name__ == '__main__':
    random_seed = 9527
    train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
    root_dir = cfg.dataset_dir
    data_dir = os.path.join(root_dir, 'jpg')
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith('.jpg')]
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]

    random.seed(random_seed)
    random.shuffle(path_imgs)

    train_breakpoints = int(len(path_imgs) * train_ratio)
    valid_breakpoints = int(len(path_imgs) * valid_ratio)
    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    move_img(train_imgs, root_dir, "train")
    move_img(valid_imgs, root_dir, "valid")
    move_img(test_imgs, root_dir, "test")