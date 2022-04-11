# ARRA: Absolute-Relative Ranking Attack against Image Retrieval

### Abstract
With the extensive application of deep learning, adversarial attacks especially query-based attacks receive more concern than ever before.
However, the scenarios assumed by existing query-based attacks against image retrieval are usually too simple to satisfy the attack demand.
In this paper, we propose a novel method termed *Absolute-Relative Ranking Attack* (ARRA) that considers a more practical attack scenario.
Specifically, we propose two compatible goals for the query-based attack, *i.e.*, *absolute ranking attack* and *relative ranking attack*, which aim to change the relative order of chosen candidates and assign the specific ranks to chosen candidates in retrieval list respectively.
We further devise the *Absolute Ranking Loss* (ARL) and *Relative Ranking Loss* (RRL) for the above goals and implement our ARRA by minimizing their combination with black-box optimizers and evaluate the attack performance by *attack success rate* and *normalized ranking correlation*.
Extensive experiments conducted on widely-used SOP and CUB-200 datasets demonstrate the superiority of the proposed approach over the baselines.
Moreover, the attack result on a real-world image retrieval system, *i.e.*, Huawei Cloud Image Search, also proves the practicability of our ARRA approach.

### About the paper

### Requirements
Our code is based on the following dependencies
- pytorch == 1.6.0
- torchvision == 0.7.0
- numpy == 1.19.5
- matplotlib == 2.0.0

The target models are trained with [Deep_Metric](https://github.com/bnu-wangxun/Deep_Metric/).

### Running the code
To reproduce the results in our paper, run:
```sh
python main.py --model BN-Inception --model_pth ${YOUR_MODEL_PTH} --dataset cub
  --attack nes --budget 2000 --epsilon 0.05 --loss arl
  --N 8 --k 1.0 --rc 0.5 --rt 1.0 --n_ex 1000 --batch_size 32 --gpu 0
  --alpha 2e-3 --momentum 0.5 --rb 0.5
```
```sh
 python arra.py --model BN-Inception --model_pth ${YOUR_MODEL_PTH} --dataset sop
  --attack nes --budget 2000 --epsilon 0.05
  --N 100 --k 0.1 --rc 0.5 --n_ex 1000 --batch_size 32 --gpu 0
  --alpha 2e-3 --momentum 0.5 --rb 0.5 --gamma 1.0
```
To launch an attack against image search service provided by [HCIS](https://www.huaweicloud.com/intl/en-us/product/imagesearch.html),
you need to install the dependencies first:
```sh
pip install huaweicloundsdkcore
pip install huaweicloudsdkimagesearch
```
then run the following command:
```sh
python real.py --dataset deep_fashion --query_root ${YOUR_QUERY_ROOT}
  --attack sa --budget 2000 --epsilon 0.05
  --N 10 --k 1.0 --gamma 3 --rb 0.5
```
