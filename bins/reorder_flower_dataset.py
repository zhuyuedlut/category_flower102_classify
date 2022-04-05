# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/4/5 11:03 PM
# @brief      : 将flower数据集按类别排放，便于分析
"""

import os
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':
    root_dir = r""
    path_mat = r""
    reorder_dir = os.path.join(root_dir, "reorder")
    jpg_dir = os.path.join(root_dir, "jpg")

    from scipy.io import loadmat
    label_array = loadmat(path_mat)["labels"].squeeze()

    names = os.listdir(jpg_dir)
    names = [p for p in names if p.endswith(".jpg")]
    for name in names:
        idx = int(name[6:11])
        label = label_array[idx-1]-1
        out_dir = os.path.join(reorder_dir, str(label))
        path_src = os.path.join(jpg_dir, name)
        my_mkdir(out_dir)
        shutil.copy(path_src, out_dir)