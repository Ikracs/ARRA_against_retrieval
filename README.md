# ARRA: Absolute-Relative Ranking Attack against Image Retrieval

### Abstract

### About the paper

### Requirements
Our code is based on the following dependencies
- pytorch == 1.6.0
- torchvision == 0.7.0
- numpy == 1.19.5
- matplotlib == 2.0.0

The target model is trained with [Deep_Metric](https://github.com/bnu-wangxun/Deep_Metric/).

### Running the code
To reproduce the results in our paper, run:
```sh
python main.py --model BN-INception --model_pth ${YOUR_MODEL_PTH} --dataset cub
  --attack nes --budget 2000 --epsilon 0.05 --loss arl
  --N 8 --k 1.0 --rc 0.5 --rt 1.0 --n_ex 1000 --batch_size 32 --gpu 0
  --alpha 2e-3 --momentum 0.5 --rb 0.5
```
```sh
 python arra.py --model BN-INception --model_pth ${YOUR_MODEL_PTH} --dataset sop
  --attack nes --budget 2000 --epsilon 0.05
  --N 100 --k 0.1 --rc 0.5 --n_ex 1000 --batch_size 32 --gpu 0
  --alpha 2e-3 --momentum 0.5 --rb 0.5 --gamma 1.0
```
If you want to lanuch an attack against image search service provided by [HCIS](https://www.huaweicloud.com/product/imagesearch.html),
you need to install the dependencies first:
```sh
pip install huaweicloundsdkcore
pip install huaweicloudsdkimagesearch
```
Then run the following commands:
```sh
python real.py --dataset deep_fashion --query_root ${YOUR_QUERY_ROOT}
  --attack sa --budget 2000 --epsilon 0.05
  --N 10 --k 1.0 --gamma 3 --rb 0.5
```
