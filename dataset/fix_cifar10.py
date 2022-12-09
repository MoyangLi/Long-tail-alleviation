import numpy as np
from PIL import Image

import torchvision
import simsiam
from torchvision.transforms import transforms

# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

normalize = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

augmentation = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        # 改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    normalize
]

transform = simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# simsiam 预训练数据读取
def get_cifar(root, l_samples, transform = transform, download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_idxs = train_split(base_dataset.targets, l_samples)
    train_dataset = CIFAR10_labeled(root, train_idxs, train=True, transform=transform)
    print(f"#Labeled: {len(train_idxs)}")

    return train_dataset

# finetune train data load
def get_cifar_train(root, l_samples, transform_train = transform_train, download=False):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_idxs = train_split(base_dataset.targets, l_samples)
    train_dataset = CIFAR10_labeled(root, train_idxs, train=True, transform=transform_train)
    print(f"#Labeled: {len(train_idxs)}")

    return train_dataset

# finetune validate data load
def get_cifar_val(root, transform_val = transform_val, download=False):
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=download)

    return test_dataset

def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]  # 返回属于当前类别的坐标索引 因为返回的是二维数组 需要的是 np[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

    return train_labeled_idxs


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              # target_transform 是对标签进行处理
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