import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import choice

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os

class RangeLossOp(nn.Module):
    def __init__(self, margin_inter, margin_intra, k, alpha, beta, gpu):
        super(RangeLossOp, self).__init__()
        self.margin_inter = margin_inter
        self.margin_intra = margin_intra
        self.k = k    # 调和项数
        self.alpha = alpha
        self.beta = beta
        self.gpu = gpu

    def compute_min_dist(self, centers):
        Matrix = torch.norm(centers, p=2, dim=1)**2  # 计算每个 center tensor 的二范数
        # repeat()函数 在指定的维度上重复这个变量 permute() 将tensor的维度换位
        # 每一行都是 a^2 + 每一列都是 b^2 - 2a*b
        N = centers.shape[0]
        dis = Matrix.repeat(N,1).permute(1,0) + Matrix.repeat(N,1) - 2 * torch.mm(centers, centers.permute(1,0))
        dist_array = dis.reshape((1, -1))
        dist_array = dist_array[torch.where(dist_array > 0)]

        dis_sort = dist_array.sort().values
        return dis_sort[0]

    def compute_max_dist(self, features_l):
        # test
        center =  features_l.mean(dim=0)
        dists = torch.norm((features_l - center), p=2)**2
        dis_sort = dists.sort().values
        return dis_sort[-1:]

    def forward(self, features = None, labels = None):

        # 计算类别数目，以及不重复的标签
        unique_labels, counts = torch.unique(labels, return_counts=True)
        # 保存每一类的类别中心
        centers = torch.zeros((unique_labels.shape[0], features.shape[1])).cuda(self.gpu)
        # 保存每一类的最大 k 个样本类内距离
        # d = torch.zeros((unique_labels.shape[0], self.k)).cuda(gpu)
        d = torch.zeros(unique_labels.shape[0]).cuda(self.gpu)
        l_r = torch.zeros((unique_labels.shape[0])).cuda(self.gpu)

        for idx, l in enumerate(unique_labels):
            # 查找同一类的所有样本
            indices = np.where(labels.cpu().numpy() == l.item())    # tensor(CUDA)->np 先转化为cpu()
            features_l = features[indices, :]
            features_l = features_l.reshape(features_l.shape[1], -1)
            # 计算类别中心
            centers[idx, :] = torch.mean(features_l, dim=0)

        d_center = self.compute_min_dist(centers)
        # print(d_center)
        # print(l_intra)
        l_inter = max(self.margin_inter - d_center, 0)
        # loss = l_intra * self.alpha + l_inter * self.beta
        loss = l_inter * torch.tensor(self.beta)

        return loss   # 不能直接生成 torch.tensor(loss) 因为会没有梯度计算函数 叶子节点无

class compensated_loss(nn.Module):
    def __init__(self,
                 train_cls_num_list=None,
                 inf_lable_distrbution=None,    # 偏好样本分布，各类别
                 weight=None):
        super(compensated_loss, self).__init__()

        self.weight = weight

        self.train_cnl = train_cls_num_list
        self.prior = np.log(self.train_cnl / sum(self.train_cnl))
        self.prior = torch.from_numpy(self.prior).type(torch.cuda.FloatTensor)

        self.inf = inf_lable_distrbution
        self.inf = np.log(self.inf / sum(self.inf))
        self.inf = torch.from_numpy(self.inf).type(torch.cuda.FloatTensor)

    def forward(self, x, target, epoch, epoch_max):
        alpha = (epoch/epoch_max)**2    # 前期交叉熵 后期 bayais
        logits = x + self.prior + self.inf # 分到每个类别的概率 soft label + log(样本分布概率 num/total_num) - log(平均概率 1/class_num)
        loss1 = F.cross_entropy(logits,    # 高斯损失
                               target,
                               weight=self.weight,
                               reduction='none')
        loss2 = F.cross_entropy(x, target, weight=self.weight, reduction='none')    # 交叉熵损失
        loss = (1-alpha) * loss1 + alpha * loss2
        return loss


def unimix_sampler(batch_size, labels, cls_num_list, tau):
    idx = np.linspace(0, batch_size - 1, batch_size)
    cls_num = np.array(cls_num_list)
    idx_prob = cls_num[labels.cpu().numpy()]
    idx_prob = np.power(idx_prob, tau, dtype=float)
    idx_prob = idx_prob / np.sum(idx_prob)
    idx = choice(idx, batch_size, p=idx_prob)
    idx = torch.Tensor(idx).type(torch.LongTensor)
    return idx


def unimix_factor(labels_1, labels_2, cls_num_list, alpha):
    cls_num_list = np.array(cls_num_list)
    n_i = cls_num_list[labels_1.cpu().numpy()]
    n_j = cls_num_list[labels_2.cpu().numpy()]
    lam = n_j / (n_i + n_j)
    lam = [np.random.beta(alpha, alpha) + t for t in lam]
    lam = np.array([t - 1 if t > 1 else t for t in lam])
    return torch.Tensor(lam).cuda()

def my_mix_factor(labels_1, labels_2, cls_num_list, alpha):
    cls_num_list = np.array(cls_num_list)
    n_i = cls_num_list[labels_1.cpu().numpy()]
    n_j = cls_num_list[labels_2.cpu().numpy()]
    lam = n_j / (n_i + n_j)
    alpha = 2 * lam    # alpha 越小，生成 lam 越小 lam，均值为 n_j / (n_i + n_j)
    beta = 2 - alpha
    lam = np.random.beta(alpha, beta)
    return torch.Tensor(lam).cuda()


def unimix(images, labels, cls_num_list, alpha, tau):

    batch_size = images.size()[0]

    # index = unimix_sampler(batch_size, labels, cls_num_list, tau)
    # my_mix 去除 sampler
    index = torch.randperm(batch_size)
    images_1, images_2 = images, images[index, :]
    labels_1, labels_2 = labels, labels[index]

    # lam = unimix_factor(labels_1, labels_2, cls_num_list, alpha)
    lam = my_mix_factor(labels_1, labels_2, cls_num_list, alpha)

    mixed_images = torch.zeros_like(images)
    for i, s in enumerate(lam):
        mixed_images[i, :, :, :] = images_1[i, :, :, :] * s + images_2[
            i, :, :, :] * (1 - s)
    mixed_images = mixed_images[:batch_size].cuda()
    labels_2 = labels_1 * lam.round() + labels_2 * (1 - lam).round()

    # labels_1, labels_2 = labels_1, labels_2[:batch_size]

    return mixed_images, labels_1, labels_2.long(), lam

def mixup(images, labels):

    batch_size = images.size()[0]
    index = torch.randperm(batch_size)
    images_1, images_2 = images, images[index, :]
    labels_1, labels_2 = labels, labels[index]

    lam = np.random.beta(1.0, 1.0)

    mixed_images = lam * images_1 + (1 - lam) * images_2
    mixed_images = mixed_images[:batch_size].cuda()

    return mixed_images, labels_1, labels_2, lam

def SMOOTH_Margin(images, target, num_class):
    batch_size = images.size(0)
    index = torch.randperm(batch_size).cuda()    # randperm() 函数将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写

    class_sample_count = np.zeros(num_class).astype(int)
    for t in torch.unique(target, sorted=True):
        class_sample_count[t.item()] = (torch.tensor((target == t).sum())).item()

    resample_count = np.zeros(num_class).astype(int)
    id = np.where(class_sample_count > 0)
    id = id[0]
    id_ = np.flipud(id)
    resample_count[id] = class_sample_count[id_]

    list1 = []
    list2 = []
    for i, class_num in enumerate(class_sample_count):
        if class_num != 0:
            # 查找属于某个类别的元素坐标 返回 tuple 类型 要取第一个元素
            indices = np.where(target.cpu().numpy() == i)
            rand_id1 = np.random.randint(class_sample_count[i], size=(resample_count[i]))
            rand_id2 = np.random.randint(class_sample_count[i], size=(resample_count[i]))
            list1 = np.concatenate((list1, (indices[0])[rand_id1]), axis=0)
            list2 = np.concatenate((list2, (indices[0])[rand_id2]), axis=0)
    lam = torch.tensor(np.random.beta(1.0, 1.0)).float().cuda()
    mixed_x = lam * images[torch.tensor(list1).long().cuda(), :] + \
              (1 - lam) * images[torch.tensor(list2).long().cuda(), :]
    mixed_x = 0.7 * mixed_x + 0.3 * images[index, :]
    y_b = target[torch.tensor(list1).long().cuda()]
    return mixed_x, y_b.long()


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to train mode
    model.train()

    RangeLoss = RangeLossOp(margin_inter=100, margin_intra=25, k=2, alpha=5e-5, beta=1e-2, gpu=cfg.gpu)

    end = time.time()
    for (images, labels) in train_loader:

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(cfg.gpu, non_blocking=True)
            labels = labels.cuda(cfg.gpu, non_blocking=True)

        if cfg.mix_type != None and epoch < cfg.mix_stop_epoch:

            if cfg.mix_type == 'unimix':
                mix_images, lab_1, lab_2, lam = unimix(
                    images=images,
                    labels=labels,
                    cls_num_list=cfg.train_cls_num_list,    # 不是按照 batch 计算而是按照 train data 类别个数计算
                    alpha=cfg.unimix_alp,
                    tau=cfg.unimix_tau)

            if cfg.mix_type == 'Margin':
                mix_images, lab_2 = SMOOTH_Margin(
                    images=images,
                    target=labels,
                    num_class=cfg.train_cls_num_list.shape[0])

            elif cfg.mix_type == 'mixup':
                mix_images, lab_1, lab_2, lam = mixup(
                    images=images,
                    labels=labels)

            elif cfg.mix_type == 'my_mix':
                mix_images, lab_1, lab_2, lam = unimix(
                    images=images,
                    labels=labels,
                    cls_num_list=cfg.train_cls_num_list,
                    alpha=cfg.unimix_alp,
                    tau=cfg.unimix_tau)
            else:
                print('Should mixup training but no mix type is selected!')
                os._exit(0)

            output2, feature2 = model(mix_images)
            output1, feature1 = model(images)
            # loss = lam * criterion(output, lab_1) + \
            #         (1 - lam) * criterion(output, lab_2)

            # loss = criterion(outputs, labels) + criterion(output, lab_2)
            """
            loss = RangeLoss(features=torch.cat([feature1, feature2], dim=0),
                             labels=torch.cat([labels, lab_2], dim=0)) + \
                   criterion(torch.cat([output1, output2], dim=0),
                             torch.cat([labels, lab_2], dim=0))
            """
            loss = criterion(torch.cat([output1, output2], dim=0),
                             torch.cat([labels, lab_2], dim=0), epoch, cfg.epochs)
            # criterion2 = nn.CrossEntropyLoss()
            # loss = criterion(output1, labels) + criterion2(output2, lab_2)
        else:
            output = model(images)
            loss = criterion(output, labels)

        loss = torch.mean(loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output1, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(val_loader, model, criterion, epoch, logger, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(cfg.num_classes).cuda()
    correct = torch.zeros(cfg.num_classes).cuda()

    cfd = np.array([])
    pred_cls = np.array([])
    gt_cls = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(cfg.gpu, non_blocking=True)
                labels = labels.cuda(cfg.gpu, non_blocking=True)

            output,_ = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = torch.mean(criterion(output, labels))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            labels_one_hot = F.one_hot(labels, cfg.num_classes)
            predict_one_hot = F.one_hot(predicted, cfg.num_classes)
            class_num = class_num + labels_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (labels_one_hot + predict_one_hot
                                 == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            cfd_part, pred_cls_part = torch.max(prob, dim=1)
            cfd = np.append(cfd, cfd_part.cpu().numpy())
            pred_cls = np.append(pred_cls, pred_cls_part.cpu().numpy())
            gt_cls = np.append(gt_cls, labels.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        acc_cls = correct / class_num
        h_acc = acc_cls[cfg.h_class_idx[0]:cfg.h_class_idx[1]].mean() * 100
        m_acc = acc_cls[cfg.m_class_idx[0]:cfg.m_class_idx[1]].mean() * 100
        t_acc = acc_cls[cfg.t_class_idx[0]:cfg.t_class_idx[1]].mean() * 100
        cal = calibration(gt_cls, pred_cls, cfd, num_bins=15)

        if not cfg.debug:
            logger.info(f'Epoch [{epoch}]:\n')
            logger.info(
                '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'
                .format(top1=top1,
                        top5=top5,
                        head_acc=h_acc,
                        med_acc=m_acc,
                        tail_acc=t_acc))
            logger.info('* ECE   {ece:.3f}%.'.format(
                ece=cal['expected_calibration_error'] * 100))

    return top1.avg, cal['expected_calibration_error'] * 100
