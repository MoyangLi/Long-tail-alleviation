# coding='utf-8'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets


import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data

# import models.wrn as models
import models.resnet_cifar as models

import simsiam.loader
import simsiam.builder

parser = argparse.ArgumentParser(description='PyTorch Simsiam Training')
# Optimization options
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='/mnt/ssd1/moyangli/Prior-LT-main/result/Final/cifar10_100/Mix/ckps/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
"""
parser.add_argument('--resume', default='./cifar10/100/Margin/model_200.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
"""

parser.add_argument('--out', default='',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA

use_cuda = torch.cuda.is_available()

if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    num_class = 10
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    num_class = 100

def main():
    set_seed(args.manualSeed)

    test_set = dataset.get_cifar_val('./data')
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)    # 测试集不需要 shuffle

    # Model
    print("==> creating Resnet-32")

    def create_model(ema=False):
        model = models.resnet32(num_classes = num_class)
        if use_cuda:
            model = model.cuda(args.gpu)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()

    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    features, targets = validate(test_loader, model, use_cuda)

    n_components = 2
    n_neighbor = 50

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=n_components, perplexity=n_neighbor, early_exaggeration=80, init='pca', random_state=0)

    Y = tsne.fit_transform(features)  # 转换后的输出
    plt.scatter(Y[:, 0], Y[:, 1], s=2, c=targets, cmap='tab10')    # camp 参数 控制颜色的类型
    plt.colorbar()
    plt.show()



def validate(valloader, model, use_cuda):
    # switch to evaluate mode
    model.eval()
    feature = []
    target = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):

            if use_cuda:
                inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu, non_blocking=True)
            # compute output
            _, features = model(inputs)
            if batch_idx == 0:
                feature = features.cpu().numpy()
                target = targets.cpu().numpy()
            else:
                feature = np.vstack((feature, features.cpu().numpy()))
                target = np.hstack((target, targets.cpu().numpy()))

    return feature, target


def set_seed(seed):
    # Random seed
    if seed is None:
        seed = random.randint(1, 10000)
    # 如果读取数据的过程采用了随机预处理(如RandomCrop、RandomHorizontalFlip等)，那么对python、numpy的随机数生成器也需要设置种子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("gpu cuda is available!")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("cuda is not available! cpu is available!")
        torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
