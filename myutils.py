import os
import torch
import numpy as np
from PIL import Image

def select_cands(B, N, ratio=0.5):
    '''
    select candidates randomly from visible ranking list.
    Args:
        B: size of the img batch
        N: len of visible ranking list
        ratio: ratio of candidates in visible ranking list
    Return:
        LongTensor of [B, ratio * N], candidates idxs
    '''
    cands = np.array([np.random.permutation(N)[: int(N * ratio)] for _ in range(B)])
    return torch.from_numpy(cands)

def gen_rank_t(B, N, rc=0.5, rt=1.0):
    '''
    generate target ranking list randomly for candidates.
    Args:
        B: size of the img batch
        N: len of visible ranking list
        rc: ratio of candidates in visible ranking list
        rt: ratio of targeted attacks in candidates
    Return:
        LongTensor of [B, N * rc], target ranking list of candidates
    '''
    C, T = int(N * rc), int(N * rc * rt)
    t_ranks = np.array([np.random.permutation(N)[: T] for _ in range(B)])
    t_ranks = torch.from_numpy(t_ranks)
    t_idxs = np.array([np.random.permutation(C)[: T] for _ in range(B)])
    t_idxs = torch.from_numpy(t_idxs)
    # 0 ~ (N - 1) represent targeted attack, N represents untargeted attack
    rank_t = torch.ones((B, C)).long() * N
    rank_t.scatter_(dim=-1, index=t_idxs, src=t_ranks)
    return rank_t

def idx2rank(idxs):
    B, N = idxs.shape
    ranks = torch.zeros_like(idxs).long()
    bidx = torch.arange(B).reshape(-1, 1)
    ranks[bidx, idxs] = torch.arange(N).repeat(B, 1)
    return ranks

def abs2rel(abs_rank):
    return idx2rank(abs_rank.argsort())

def evaluate(x, y, mtype):
    '''
    check whether an image is successfully attacked.
    Args:
        x: LongTensor of [B, C], idx of each candidate in current ranking list
        y: LongTensor of [B, C], idx of each candidate in target ranking list
        mtype: Normalized Ranking Correlation or Attack Success Rate
    '''
    metric = {}
    if mtype.lower() == 'all' or mtype.lower() == 'nrc':
        rela_rank_x = (x.unsqueeze(-1) - x.unsqueeze(1)).sign()
        rela_rank_y = (y.unsqueeze(-1) - y.unsqueeze(1)).sign()
        corr_mat = torch.triu(rela_rank_x == rela_rank_y, diagonal=1)
        pnum = torch.triu(torch.ones(corr_mat.shape[1:]), diagonal=1).sum()
        metric['NRC'] = corr_mat.sum(dim=(1, -1)).float() / pnum
    if mtype.lower() == 'all' or mtype.lower() == 'asr':
        metric['ASR'] = (x == y).float().mean(dim=-1)
    return metric

def display(metric):
    for k in metric.keys():
        print(' {:s}: {:.3f}'.format(k, metric[k].mean()), end='')
    print('')   # start an new line

def save_img(img, save_pth):
    img = 255 * img.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    b, g, r = img.split(); img = Image.merge('RGB', (r, g, b))
    img.save(save_pth)

def save_imgs(imgs, adv_imgs, save_root):
    '''save clean imgs and their counterparts.''' 
    img_root = os.path.join(save_root, 'imgs')
    adv_root = os.path.join(save_root, 'advs')
    if not os.path.exists(img_root): os.makedirs(img_root)
    if not os.path.exists(adv_root): os.makedirs(adv_root)
    for i, img in enumerate(imgs):
        save_img(img, os.path.join(img_root, '{:d}.jpg'.format(i)))
    for i, img in enumerate(adv_imgs):
        save_img(img, os.path.join(adv_root, '{:d}.jpg'.format(i)))

def save_retrievals(query, model, data_loader, K, save_root):
    '''
    save query img and the first-K imgs in the ranking list.
    WARNING: shuffle=False is needed in data_loader
    '''
    ret_root = os.path.join(save_root, 'retrieval')
    if not os.path.exists(ret_root): os.makedirs(ret_root)
    ridx = model(query.unsqueeze(0))[0, :K]
    for k in range(K):
        ret_img = data_loader.dataset[ridx[k]][0]
        save_img(ret_img, os.path.join(ret_root, 'rank_{}.jpg'.format(k)))
    save_img(query, os.path.join(ret_root, 'query.jpg'))
