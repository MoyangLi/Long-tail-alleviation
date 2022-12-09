# Class Imbalance Alleviation

## Dependencies

* `python3`
* `pytorch`
* `torchvision`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## How to train the model?
To train a model on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, random sampler for labeled data and random sampler for unlabeled data
```
python train.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler random --semi-sampler random --out cifar10_fix_100_2_random_random
```

To fine-tune a model (here the model trained with above command) on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, mean sampler for labeled data and mean sampler for unlabeled data
```
python3 fix_finetune.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler mean --semi-sampler mean --resume cifar10_fix_100_2_random_random/checkpoint.pth.tar --out cifar10_fix_100_2_random_random_stage2
```
## Method
### Bias Compensation
![img_6.png](img_6.png)
![img_7.png](img_7.png)

### Visualization of introduction of RangeLoss

![img_8.png](img_8.png)
![img_9.png](img_9.png)


### Experiment Results
![img_4.png](img_4.png)