# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/4/5 7:53 PM
# @brief      : flower 102数据集读取
"""
import os

from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset

from config.flower_config import cfg


class FlowerDataset(Dataset):
    cls_num = 102
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith(".jpg")]

        label_file = "imagelabels.mat"
        path_label_file = os.path.join(self.root_dir, "..", label_file)
        label_array = loadmat(path_label_file)['labels'].squeeze()
        self.label_array = label_array

        idx_imgs = [int(n[6:11]) for n in names_imgs]

        path_imgs = [os.path.join(self.root_dir, n) for n in names_imgs]
        self.img_info = [(p, int(label_array[idx-1]-1)) for p, idx in zip(path_imgs, idx_imgs)]

if __name__ == "__main__":
    root_dir = os.path.join(cfg.dataset_dir, 'test')

    test_dataset = FlowerDataset(root_dir)

    print(len(test_dataset))
    print(next(iter(test_dataset)))