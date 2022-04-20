import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torch
import numpy as np

from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tools.common_tools import check_data_dir
from datasets.cifar_longtail import CifarLIDataset
from config.cifar_config import cfg
import matplotlib.pyplot as plt


class ProgressiveSampler(object):
    def __init__(self, dataset, max_epoch):
        self.max_epoch = max_epoch
        self.dataset = dataset
        self.train_targets = [int(info) for info in dataset.img_info]
        self.nums_per_cls = dataset.nums_per_cls

    def _cal_class_prob(self, q):
        """
        根据q值计算每个类的采样概率，公式中的 p_j
        :param q: float , [0, 1]
        :return: list,
        """
        num_pow = list(map(lambda x: pow(x, q), self.nums_per_cls))
        sigma_num_pow = sum(num_pow)
        cls_prob = list(map(lambda x: x/sigma_num_pow, num_pow))
        return cls_prob

    def _cal_pb_prob(self, t):
        """
        progressively-balanced 概率计算
        :param t: 当前epoch数
        :return:
        """
        p_ib = self._cal_class_prob(q=1)
        p_cb = self._cal_class_prob(q=0)
        p_pb = (1 - t/self.max_epoch) * np.array(p_ib) + (t/self.max_epoch) * np.array(p_cb)

        p_pb /= np.array(self.nums_per_cls)
        return p_pb.tolist()

    def __call__(self, epoch):
        p_pb = self._cal_pb_prob(t=epoch)
        p_pb = torch.tensor(p_pb, dtype=torch.float)
        # 计算每个样本被采样的权重，这里是依据样本的类别来赋权，self.train_targets是标签
        samples_weights = p_pb[self.train_targets]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights))
        return sampler, p_pb

    def plot_line(self):
        for i in range(self.max_epoch):
            _, weights = self(i)
            if i % 20 == 19:
                x = range(len(weights))
                plt.plot(x, weights, label="t="+str(i))
        plt.legend()
        plt.title("max epoch="+str(self.max_epoch))
        plt.xlabel("class index sorted by numbers")
        plt.ylabel("weights")
        plt.show()


if __name__ == '__main__':
    # 设置路径
    train_dir = r""
    check_data_dir(train_dir)
    train_data = CifarLIDataset(root_dir=train_dir, transform=cfg.transforms_train, isTrain=True)

    max_epoch = 200
    sampler_generator = ProgressiveSampler(train_data, max_epoch)
    sampler_generator.plot_line()

    for epoch in range(max_epoch):
        if epoch % 20 != 19:
            continue

        sampler, _ = sampler_generator(epoch)
        train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False, num_workers=cfg.workers,
                                  sampler=sampler)
        labels = []
        for data in train_loader:
            _, label, _ = data
            labels.extend(label.tolist())
        print("Epoch:{}, Counter:{}".format(epoch, Counter(labels)))
