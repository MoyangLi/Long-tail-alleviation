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
Considering this problem in a simple binary classification setting, there was a bias between the optimal decision boundaries for the long tail and balanced distributions, and the bias increased the probability of decision error on testing data and favoured the majority class.

<div align=center> 
<img src=https://user-images.githubusercontent.com/119117070/206727528-099b7857-8b88-41c2-9656-18bd3fd8e003.png width=60% />
</div>

`Compensate for the bias caused by class imbalance by modifying the standard cross-entropy loss to obtain an unbiased decision surface.`

<div align=left> 
<img src=https://user-images.githubusercontent.com/119117070/206746514-6d1941ce-7594-4d88-84c5-96a78a8d554c.png width=48% />
</div>


### Range Loss

When visualizing the feature distribution, I found that the degradation of performance was mainly caused by the poor separability between tail classes and the dispersion of intra-class samples in the feature space rather than the expected clustering. 

<div align=center> 
<img src=https://user-images.githubusercontent.com/119117070/206746893-4f355d5a-ffaa-4781-8ee1-1904579eb635.png width=60% />
</div>

`Utilize range loss to reduce the intra-class variations and enlarge the inter-class distance to improve the separability.`

<div align=center> 
  <img src=https://user-images.githubusercontent.com/119117070/206727306-f86d16a3-28a2-48f1-a843-01fc1f9b2f4b.png width=80% />
</div>


<p align="center">(a) feature distribution with Range Loss  &nbsp; &emsp; &emsp; &emsp; &emsp; &emsp;  (b)feature distribution w/o Range Loss</p>

### Experiment Results

<img src=https://user-images.githubusercontent.com/119117070/206727637-8143c89c-e2cc-42c8-911f-0af7dab934ce.png width=100% />
