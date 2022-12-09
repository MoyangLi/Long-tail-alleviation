from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from core import RangeLossOp, compensated_loss
import simsiam.loader
import simsiam.builder

# import models.wrn as models
import models.resnet_cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch Simsiam Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--out', default='',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default=3, type=int,
                    help='GPU id to use.')

# Method options
parser.add_argument('--num_max', type=int, default=5000,
                        help='Number of samples in the maximal class')
parser.add_argument('--imb_ratio', type=int, default=100,    # head class 和 tail class 比例
                        help='Imbalance ratio for data')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')

# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')

# Hyperparameters for FixMatch
parser.add_argument('--ema-decay', default=0.999, type=float)

# Sampler options
parser.add_argument('--sampler', default='random', type=str, choices=['random', 'mean', 'reverse'],
                    help='Sampler for labeled data')
parser.add_argument('--semi-sampler', default='random', type=str, choices=['random', 'mean', 'reverse'],
                    help='Sampler for unlabeled data')
parser.add_argument('--class-num',type=int,help='the number of train data from different classes')

# Mix type options
parser.add_argument('--Mix', default='None', type=str, choices=['RangeLoss++', 'None', 'Mixup', 'UniMix', 'My_Mix', 'RandMix'],
                    help='Mix type')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA

use_cuda = torch.cuda.is_available()

if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    num_class = 10
    args.num_max = 5000
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    num_class = 100
    args.num_max = 500

best_acc = 0  # best test accuracy
args.out = os.path.join(str(args.imb_ratio), args.Mix)

def main():
    global best_acc
    set_seed(args.manualSeed)
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced ' + args.dataset)

    # labeled data
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio)    # num_class 代表类别个数

    train_labeled_set = dataset.get_cifar_train('./data', N_SAMPLES_PER_CLASS)
    test_set = dataset.get_cifar_val('./data')

    target = torch.tensor(train_labeled_set.targets)
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])    # 计算每个类别的数据量，bool 值输出为 1 代表为当前类别，求 bool 值之和即为类别个数
    args.class_num = class_sample_count/class_sample_count.sum()

    mean_weight = 1. / class_sample_count.float()    # 每个类别样本占的概率
    reverse_weight = mean_weight / class_sample_count.float()

    mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
    reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])

    mean_sampler = data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
    reverse_sampler = data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))
    if args.sampler == 'random':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.sampler == 'mean':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, sampler=mean_sampler, num_workers=4, drop_last=True)    # 默认 shuffle = False 因为本身 sampler 就是随机取的数据
    elif args.sampler == 'reverse':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, sampler=reverse_sampler, num_workers=4, drop_last=True)

    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)    # 测试集不需要 shuffle

    # Model
    print("==> creating Resnet-32")

    def create_model(ema=False):
        model = models.resnet32(num_classes=10)


        if use_cuda:
            model = model.cuda(args.gpu)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # 构建 Rangeloss 项
    RangeLoss = RangeLossOp(margin_inter=100, margin_intra=25, k=2, alpha=5e-5, beta=1e-2, gpu=args.gpu)
    RangeLoss = RangeLoss.cuda(args.gpu)
    train_criterion = MixLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'fix-cifar'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss M', 'Test Loss', 'Top-1 Acc', 'Top-5 Acc', 'Head Acc', 'Mid Acc', 'Tail Acc'])

    # 保存 Loss 和 Accuracy
    Top1 = []
    Top5 = []
    Acc = []
    path_res = os.path.join(args.out, 'Res')

    """
    path_res = os.path.join(args.out, 'Res.npz')
    with np.load(path_res) as Res1:
        Top1 = Res1['Top1'].tolist()
        Top5 = Res1['Top5'].tolist()
        Acc = Res1['Acc'].tolist()
    """

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Training part
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, model, optimizer, ema_optimizer,
                                                       train_criterion, use_cuda, RangeLoss)

        test_loss, top1, top5, test_cls= validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

        Top1.append(top1)
        Top5.append(top5)
        Acc.append(test_cls)

        np.savez(path_res, Top1=Top1, Top5=Top5, Acc=Acc)

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, top1, top5, test_cls[0], test_cls[1], test_cls[2]])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, is_best=(top1 > best_acc))

        if top1 > best_acc:
            best_acc = top1

    logger.close()
    np.savez(path_res, Top1=Top1, Top5=Top5, Acc=Acc, Mean=np.mean(Top1[-20:]), bAcc=best_acc)

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(Top1[-20:]))

    print('Best bAcc:')
    print(best_acc)

    print('Name of saved folder:')
    print(args.out)


def train(labeled_trainloader, model, optimizer, ema_optimizer, criterion, use_cuda, RangeLoss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        # iter 完一轮重新 iter
        try:    # 可能产生异常的代码块
            inputs_x, targets_x = labeled_train_iter.next()    # batch 大小为 64，所以两变量大小都是 64
        except:    # 处理异常的代码块
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)

        targets_x = targets_x.long()
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(args.gpu), targets_x.cuda(args.gpu, non_blocking=True)

        if args.Mix != 'None' and args.Mix != 'RangeLoss++':    # select the type used to augment data
            image_x, targets_a, targets_b, lam = mixup_data(inputs_x, targets_x, args, use_cuda)
            output2,_ = model(image_x)

        if args.Mix == 'My_Mix':
            output1,_ = model(inputs_x)    # the output of the original training data
            loss, Lx, Lm = criterion(output1, targets_a, output2, targets_b, lam)
            loss = Lx + Lm

        elif args.Mix == 'None':
            output1, feature = model(inputs_x)
            criterion = nn.CrossEntropyLoss()
            Lx = criterion(output1, targets_x)
            Lm = RangeLoss(features = feature, labels=targets_x)   # calculate the range loss in the feature space
            loss = Lx + Lm

        else:
            loss, Lx, Lm = criterion(output2, targets_a, output2, targets_b, lam)

        # 计算类间和类内距离

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_m.update(Lm.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_m.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_m.avg)

def mixup_data(images, target, args, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = images.size(0)
    # target = target.long()
    # 计算当前 batch 每个类别的数量
    class_sample_count = args.class_num

    if use_cuda:
        index = torch.randperm(batch_size).cuda(args.gpu)    # randperm() 函数将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
    else:
        index = torch.randperm(batch_size)
    # 表征倾向于生成更靠近哪类样本的 mixup 数据
    alpha = class_sample_count[target]
    beta = (class_sample_count[target])[index]
    alpha = 2 * alpha / (alpha + beta)
    beta = 2 - alpha

    y_a = target
    mixed_x = torch.zeros_like(images)
    # 将数据放到 gpu 上  double 精度变成 float 精度
    if args.Mix == 'Mixup':
        lam = torch.tensor(np.random.beta(1.0, 1.0)).float().cuda(args.gpu)
        mixed_x = lam * images +(1-lam) * images[index,:]
        y_b = target[index]

    elif args.Mix == 'My_Mix':
        # batch 内计算样本比例
        class_sample_count = torch.tensor(np.zeros(num_class))
        for t in torch.unique(target, sorted=True):
            class_sample_count[t.long()] = torch.tensor((target == t).sum())
        # 计算当前 batch 每个类别的概率
        class_sample_count = class_sample_count / batch_size
        # 表征倾向于生成更靠近哪类样本的 mixup 数据
        alpha = class_sample_count[target]
        beta = (class_sample_count[target])[index]
        alpha = 2 * alpha / (alpha + beta)
        beta = 2 - alpha
        lam = torch.tensor(np.random.beta(alpha, beta)).float().cuda(args.gpu)
        for i, para in enumerate(lam):  # 直接加 [] 得到的是 list
            mixed_x[i, :] = (para * (images[index])[i, :] + (1 - para) * images[i, :])
        y_b = target[index] * lam.round() + target * (1 - lam).round()

    elif args.Mix == 'RandMix':
        lam = torch.tensor(np.random.beta(alpha, beta)).float().cuda(args.gpu)
        for i, para in enumerate(lam):  # 直接加 [] 得到的是 list
            mixed_x[i, :] = (para * (images[index])[i, :] + (1 - para) * images[i, :])
        y_b = (torch.rand(batch_size).cuda(args.gpu) * (num_class - 1)).round()

    elif args.Mix == 'UniMix':
        lam = torch.tensor(np.ones(batch_size) * np.random.beta(0.8, 0.8)).float().cuda(args.gpu)
        for i, para in enumerate(lam):  # 直接加 [] 得到的是 list
            para = (para + beta[i] - 1) if torch.gt(para, alpha[i]) else (para + beta[i])
            mixed_x[i, :] = (para * (images[index])[i, :] + (1 - para) * images[i, :])  # !!!!!!! 必须先求 index 对应 tensor 再求 i 对应 tensor
        y_b = target[index]

    else:
        exit(0)
        print('Mix type not find!!!')

    return mixed_x, y_a, y_b.long(), lam


def validate(valloader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_correct = classwise_correct.cuda(args.gpu)
        classwise_num = classwise_num.cuda(args.gpu)
        section_acc = section_acc.cuda(args.gpu)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()

    if use_cuda:
        section_acc = section_acc.cpu()

    return (losses.avg, top1.avg, top5.avg, section_acc.numpy()*100)

def make_imb_data(max_num, class_num, gamma):    # head class 和 tail class 样本数量比为 gamma
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)

def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar', is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth.tar'))

class MixLoss(object):
    def __call__(self, output1, lab_1, output2, lab_2, lam):

        criterion = nn.CrossEntropyLoss(reduction = 'none')
        Loss = lam * criterion(output1, lab_1) + (1 - lam) * criterion(output2, lab_2)
        Loss = torch.mean(Loss)
        criterion_ = nn.CrossEntropyLoss()
        Lx = criterion_(output1, lab_1)
        Lm = criterion_(output2, lab_2)
        return Loss, Lx, Lm


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

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
