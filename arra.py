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

def select_cands(idxs, N, ratio):
	C = int(N * ratio); S = int(C * 0.5)
	c_r, c_a = [], []
	for idx in idxs:
		c_r.append(np.random.permutation(idx[: N])[: S])
		c_a.append(np.random.permutation(idx[: N])[: S])
	c_r = torch.from_numpy(np.array(c_r))
	c_a = torch.from_numpy(np.array(c_a))
	return c_r, c_a

def gen_rank_t(c_r, c_a, N):
	c_r, c_a = c_r.numpy(), c_a.numpy()
	rank_r, rank_a = [], []
	for i in range(c_r.shape[0]):
		cidxs = np.union1d(c_r[i], c_a[i])
		ranks = np.random.permutation(N)[: cidxs.shape[0]]
		rank_r.append([ranks[np.where(c == cidxs)[0]].item() for c in c_r[i]])
		rank_a.append([ranks[np.where(c == cidxs)[0]].item() for c in c_a[i]])
	rank_r = myutils.abs2rel(torch.from_numpy(np.array(rank_r)))
	rank_a = torch.from_numpy(np.array(rank_a))
	return rank_r, rank_a

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Absoluate-Relative Ranking Attack against Image Retrieval Model')
	parser.add_argument('--model', type=str, default='BN-Inception', choices=['BN-Inception'])
	parser.add_argument('--model_pth', type=str, required=True)
	parser.add_argument('--dataset', type=str, default='cub', choices=['cub', 'sop'])
	parser.add_argument('--attack', type=str, default='zoo', choices=['zoo', 'nes', 'sh', 'sa'])
	parser.add_argument('--budget', type=int, default=2000, help='query budget for black-box attack')
	parser.add_argument('--epsilon', type=float, default=0.05, help='maximum lp norm of perturbation')
	parser.add_argument('--N', type=int, default=8, help='len of visible ranking list')
	parser.add_argument('--rc', type=float, default=0.5, help='ratio of candidates in visible ranking list')
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
	parser.add_argument('--gamma', type=float, default=1, help='balance factor for ARRA')
	
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
	
	# initialize RRL, ARL and black-box attacker
	rrl, arl = RRL(cfg), ARL(cfg)
	attacker = eval(cfg['attack'].upper())(cfg)
	
	# run black-box attack batch by batch
	metrics_all = {
		'NRC'    : torch.FloatTensor(),
		'ASR'    : torch.FloatTensor(),
		'RRL'    : torch.FloatTensor(),
		'ARL'    : torch.FloatTensor(),
		'Loss'   : torch.FloatTensor(),
		'L2 norm': torch.FloatTensor(),
		'ASR'    : torch.FloatTensor(),
		'NRC'    : torch.FloatTensor()
	}

	num_attacked = 0
	start_time = time.time()
	with torch.no_grad():
		for bid, (imgs, _) in enumerate(test_loader):
			print('Attacking Batch {:d}:'.format(bid))
			idxs = vmodel.query(imgs)
			c_r, c_a = select_cands(idxs, cfg['N'], cfg['rc'])
			rank_r, rank_a = gen_rank_t(c_r, c_a, cfg['N'])
			
			model = lambda x: vmodel.query(x)

			def criterion(idxs):
				rank = myutils.idx2rank(idxs).clamp(0, cfg['N'])
				cur_rank_r = myutils.abs2rel(rank.gather(dim=-1, index=c_r))
				cur_rank_a = rank.gather(dim=-1, index=c_a)
				metrics = myutils.evaluate(cur_rank_r, rank_r, mtype='NRC')
				metrics.update(myutils.evaluate(cur_rank_a, rank_a, mtype='ASR'))
				metrics['RRL'] = rrl(cur_rank_r, rank_r)
				metrics['ARL'] = arl(cur_rank_a, rank_a)
				metrics['Loss'] = metrics['RRL'] + cfg['gamma'] * metrics['ARL']
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
