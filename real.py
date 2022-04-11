# WARNNING: Before runing this code, you need to re-implement a black-box optimizer
# (SignHunter and SquareAttack are recommended) with numpy

import os
import sys
import time
import json
import requests
import argparse
import torch
import numpy as np
from PIL import Image

import myutils
from adv_loss import RRL, ARL
from your_black_box import SquareAttack as YOUR_SA
from rsystems import BingSearch, HuaweiCloudSearch
from rsystems import SUPPORT_FORMAT

from huaweicloudsdkcore.exceptions import exceptions

def select_cands(ids):
    return [ids[1], ids[9]], [ids[9]]

def gen_rank_t(N):
    rank_r = torch.tensor([[1, 0]]).long()
    rank_a = torch.tensor([[1]]).long()
    return rank_r, rank_a

def id2rank(cids, ids, N):
    rank = []
    for cid in cids:
        r = np.where(cid == ids[0])[0]
        if r.shape[0] == 0: rank.append(N)
        else: rank.append(r.item())
    return torch.tensor([rank])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARRA against Huawei Cloud Image Search')
    parser.add_argument('--dataset', type=str, default='deep_fashion')
    parser.add_argument('--data_root', type=str, default='rdata/DeepFashion/img')
    parser.add_argument('--query_root', type=str, default='queries')
    parser.add_argument('--save_root', type=str, default='saves/real')
    parser.add_argument('--budget', type=int, default=200, help='query budget for black-box attack')
    parser.add_argument('--epsilon', type=float, default=0.05, help='maximum lp norm of perturbation')
    parser.add_argument('--N', type=int, default=10, help='len of visible ranking list')
    parser.add_argument('--gpu', type=str, default='0', help='Available GPU id')
    parser.add_argument('--log_freq', type=int, default=1, help='log frequency')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')

    # hyperparameters for black-box attacks or loss functions
    parser.add_argument('--k', type=float, default=1, help='hyperp of RRL/ARL/ARRL')
    parser.add_argument('--gamma', type=float, default=3, help='balance factor for ARRA')
    parser.add_argument('--rb', type=float, default=0.5, help='ratio of bases in ARL')
    
    cfg = vars(parser.parse_args())
    for k in cfg.keys(): print(k + ' ' + str(cfg[k]))

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu']

    search = HuaweiCloudSearch(cfg)
    rrl, arl = RRL(cfg), ARL(cfg)
    attacker = YOUR_SA(cfg)

    adv_root = os.path.join(cfg['save_root'], 'advs')
    os.makedirs(adv_root, exist_ok=True)

    print('Start Creating img indices...')
    # search.delete_instance(cfg['dataset'])
    search.create_instance(cfg['dataset'])
    categories = os.listdir(cfg['data_root'])
    np.random.shuffle(categories)

    gallery = {}
    category_num, img_num_per_category = 100, 10
    for category in categories[:category_num]:
        croot = os.path.join(cfg['data_root'], category)
        if os.path.isdir(croot):
            gallery[category] = []; count = 0
            for fname in os.listdir(croot):
                if count >= img_num_per_category: break
                ext = os.path.splitext(fname)[-1].lower()
                if ext in SUPPORT_FORMAT:
                    print('Adding img {:s} in {:s}'.format(fname, category))
                    try:
                        img_pth = os.path.join(croot, fname)
                        search.add_img(img_pth)
                        gallery[category].append(fname); count += 1
                    except exceptions.ClientRequestException as e:
                        print('Error Code [{}], '.format(e.error_code), end='')
                        print('Message: {}'.format(e.error_msg))
    
    gal_pth = os.path.join(cfg['save_root'], 'gallery.json')
    myutils.save_json(gallery, gal_pth)

    start_time = time.time()
    print('Start attacking Huawei Cloud Image Search...')
    for fname in os.listdir(cfg['query_root']):
        ext = os.path.splitext(fname)[-1].lower()
        if ext in SUPPORT_FORMAT:
            try:
                print('Querying img {:s}...'.format(fname))
                img_pth = os.path.join(cfg['query_root'], fname)
                adv_pth = os.path.join(adv_root, fname)
                ids = search.retrieval(img_pth)

                print('Init ranking list: ', ids)

                c_r, c_a = select_cands(ids)
                rank_r, rank_a = gen_rank_t(cfg['N'])

                def model(img):
                    myutils.save_img(img.squeeze(0), adv_pth)
                    return np.array([search.retrieval(adv_pth)])

                def criterion(ids):
                    metrics = {}
                    cur_rank_r = myutils.abs2rel(id2rank(c_r, ids, cfg['N']))
                    cur_rank_a = id2rank(c_a, ids, cfg['N'])
                    metrics['RRL'] = rrl(cur_rank_r, rank_r)
                    metrics['ARL'] = arl(cur_rank_a, rank_a)
                    metrics['Loss'] = metrics['RRL'] + cfg['gamma'] * metrics['ARL']
                    return metrics

                print('Attacking start...')
                img = myutils.load_img(img_pth).unsqueeze(0)
                adv_img = attacker.run(model, criterion, img)[0]
                myutils.save_img(adv_img.squeeze(0), adv_pth)
                adv_ids = search.retrieval(adv_pth)
                print('Final ranking list: ', adv_ids)
            
            except exceptions.ClientRequestException as e:
                print('Error Code [{}], '.format(e.error_code), end='')
                print('Message: {}'.format(e.error_msg))
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Elapsed Time: {:.0f} h {:.0f} m'.format(hours, minutes))
