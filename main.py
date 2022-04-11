import os
import sys
import time
from datetime import datetime
import argparse
import torch
import numpy as np
import myutils
from victim_model import Model
from data import load_cub, load_sop
from adv_loss import RL, SRC
from adv_loss import RRL, ARL
from black_box import ZOO, NES
from black_box import SignHunter as SH
from black_box import SquareAttack as SA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Black-box Attack against Image Retrieval Model')
    parser.add_argument('--model', type=str, default='BN-Inception', choices=['BN-Inception'])
    parser.add_argument('--model_pth', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub', 'sop'])
    parser.add_argument('--attack', type=str, default='zoo', choices=['zoo', 'nes', 'sh', 'sa'])
    parser.add_argument('--budget', type=int, default=2000, help='query budget for black-box attack')
    parser.add_argument('--epsilon', type=float, default=0.05, help='maximum lp norm of perturbation')
    parser.add_argument('--loss', type=str, default='rrl', choices=['rl', 'src', 'rrl', 'arl'])
    parser.add_argument('--N', type=int, default=8, help='len of visible ranking list')
    parser.add_argument('--rc', type=float, default=0.5, help='ratio of candidates in visible ranking list')
    parser.add_argument('--rt', type=float, default=1.0, help='ratio of targeted attacks in candidates')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of images to attack')
    parser.add_argument('--batch_size', type=int, default=64, help='num of images attacked in an iter')
    parser.add_argument('--gpu', type=str, default='0', help='Available GPU id')
    parser.add_argument('--log_freq', type=int, default=500, help='log frequency')
    parser.add_argument('--save_root', type=str, default=None, help='save root of adv images')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')

    # hyperparameters for black-box attacks or loss functions
    parser.add_argument('--alpha', type=float, default=1e-3, help='learning rate of ZOO/NES')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum of ZOO/NES')
    parser.add_argument('--n_samples', type=int, default=20, help='sampling num of ZOO/NES')
    parser.add_argument('--k', type=float, default=1, help='hyperp of RRL/ARL/ARRL')
    parser.add_argument('--rb', type=float, default=0.5, help='ratio of bases in ARL')
    
    cfg = vars(parser.parse_args())
    for k in cfg.keys(): print(k + ' ' + str(cfg[k]))
    
    # general settings
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu']
    
    # load victim model and dataset
    print('init victim model and load dataset...')
    vmodel = Model(model_type=cfg['model'], model_pth=cfg['model_pth'])
    test_set = load_cub() if cfg['dataset'] == 'cub' else load_sop()
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=3)
    vmodel.extract_gallery_feats(test_loader)
    
    # initialize loss function and black-box attacker
    loss = eval(cfg['loss'].upper())(cfg)
    attacker = eval(cfg['attack'].upper())(cfg)
    
    # run black-box attack batch by batch
    metrics_all = {
        'Loss': torch.FloatTensor(),
        'L2 norm': torch.FloatTensor()
    }
    mtype = 'ASR' if cfg['loss'].lower() in ['rl', 'arl'] else 'NRC'
    metrics_all[mtype] = torch.FloatTensor()

    num_attacked = 0
    start_time = time.time()
    with torch.no_grad():
        for bid, (imgs, _) in enumerate(test_loader):
            print('Attacking Batch {:d}:'.format(bid))
            idxs = vmodel.query(imgs)
            cidxs = myutils.select_cands(idxs, cfg['N'], cfg['rc'])
            rank_s = myutils.idx2rank(idxs).gather(dim=-1, index=cidxs)
            rank_t = myutils.gen_rank_t(*cidxs.shape, cfg['N'], cfg['rt'])

            model = lambda x: vmodel.query(x)

            def criterion(idxs):
                rank = myutils.idx2rank(idxs).clamp(0, cfg['N'])
                rank = rank.gather(dim=-1, index=cidxs)
                if cfg['loss'].lower() == 'rl':
                    metrics = myutils.evaluate(rank, rank_t, 'ASR')
                    metrics['Loss'] = loss(rank, rank_s)
                elif cfg['loss'].lower() in ['src', 'rrl']:
                    rank = myutils.abs2rel(rank)
                    rank_r = myutils.abs2rel(rank_t)
                    metrics = myutils.evaluate(rank, rank_r, 'NRC')
                    metrics['Loss'] = loss(rank, rank_r)
                else:   # cfg['loss'].lower() == 'arl'
                    metrics = myutils.evaluate(rank, rank_t, 'ASR')
                    metrics['Loss'] = loss(rank, rank_t)
                return metrics
            
            adv_imgs, metrics = attacker.run(model, criterion, imgs)
            
            for k in metrics.keys():
                metrics_all[k] = torch.cat((metrics_all[k], metrics[k]))
            norms = (adv_imgs - imgs).norm(p=2, dim=(1, 2, -1))
            metrics_all['L2 norm'] = torch.cat((metrics_all['L2 norm'], norms))
            
            if cfg['save_root']:
                print('Saving imgs and adv imgs...')
                save_pth = os.path.join(cfg['save_root'], 'batch_{:d}'.format(bid))
                myutils.save_imgs(imgs, adv_imgs, save_pth)
            
            num_attacked += imgs.shape[0]
            if num_attacked >= cfg['n_ex']: break
    
    print('Final Results ({0}/{0}): '.format(cfg['n_ex']), end='')
    metrics_all = {k: metrics_all[k][: cfg['n_ex']] for k in metrics_all.keys()}
    myutils.display_evaluations(metrics_all)
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Elapsed Time: {:.0f} h {:.0f} m'.format(hours, minutes))
