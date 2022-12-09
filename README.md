# Class Imbalance Alleviation

## Dependencies

* `python3`
* `pytorch`
* `torchvision`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## How to run?
To train a model on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, random sampler for labeled data and random sampler for unlabeled data
```
python train.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler random --semi-sampler random --out cifar10_fix_100_2_random_random
```

To fine-tune a model (here the model trained with above command) on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, mean sampler for labeled data and mean sampler for unlabeled data
```
python fix_finetune.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler mean --semi-sampler mean --resume cifar10_fix_100_2_random_random/checkpoint.pth.tar --out cifar10_fix_100_2_random_random_stage2
```
## Method
### Bias Compensation
<img src=https://user-images.githubusercontent.com/119117070/206727528-099b7857-8b88-41c2-9656-18bd3fd8e003.png width=70% />
<img src=https://user-images.githubusercontent.com/119117070/206727710-e6eff004-9a08-4cdc-afa4-84188480f571.png width=80% />

### RangeLoss

<img src=https://user-images.githubusercontent.com/119117070/206729147-7e672302-062b-4fb4-9380-b029ce57b98e.png width=50% />
<img src=https://user-images.githubusercontent.com/119117070/206727306-f86d16a3-28a2-48f1-a843-01fc1f9b2f4b.png width=80% />

### Experiment Results
<img src=https://user-images.githubusercontent.com/119117070/206727637-8143c89c-e2cc-42c8-911f-0af7dab934ce.png width=100% />
