import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
# from RandAugment import RandAugment
# from RandAugment.augmentations import CutoutDefault

cifar100_mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
cifar100_std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)    # 训练集和测试集归一化参数相同???
])


# finetune train data load
def get_cifar_train(root, l_samples, transform_train = transform_train, download=False):
    base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
    train_idxs = train_split(base_dataset.targets, l_samples)
    train_dataset = CIFAR100_labeled(root, train_idxs, train=True, transform=transform_train)
    print(f"#Labeled: {len(train_idxs)}")

    return train_dataset

# finetune validate data load
def get_cifar_val(root, transform_val = transform_val, download=False):
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=download)

    return test_dataset

def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []

    for i in range(100):
        idxs = np.where(labels == i)[0]  # 返回属于当前类别的坐标索引 因为返回的是二维数组 需要的是 np[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

    return train_labeled_idxs


class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target