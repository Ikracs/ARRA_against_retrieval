import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def select_cands(idxs, N, ratio=0.5):
    '''
    select candidates randomly from visible ranking list.
    Args:
        idxs: ranked idx sequence returned by retrieval model
        N: len of visible ranking list
        ratio: ratio of candidates in visible ranking list
    Return:
        LongTensor of [B, ratio * N], candidates idxs
    '''
    B = idxs.shape[0]
    index = np.array([np.random.permutation(N)[: int(N * ratio)] for _ in range(B)])
    return idxs.gather(dim=-1, index=torch.from_numpy(index))

def gen_rank_t(B, C, N, ratio=1.0):
    '''
    generate target ranking list randomly for candidates.
    Args:
        B: size of image batch
        C: num of candidates
        N: len of visible ranking list
        ratio: ratio of targeted attacks in candidates
    Return:
        LongTensor of [B, C], target ranking list of candidates
    '''
    T = int(C * ratio)
    t_ranks = np.array([np.random.permutation(N)[: T] for _ in range(B)])
    t_idxs = np.array([np.random.permutation(C)[: T] for _ in range(B)])
    t_ranks = torch.from_numpy(t_ranks); t_idxs = torch.from_numpy(t_idxs)
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
    return idx2rank(abs_rank.argsort(dim=-1))

def evaluate(x, y, mtype):
    '''
    check whether an image is successfully attacked.
    Args:
        x: LongTensor of [B, C], idx of each candidate in current ranking list
        y: LongTensor of [B, C], idx of each candidate in target ranking list
        mtype: metric type in str, i.e., NRC | ASR | ASR@K
    '''
    metrics = {}
    if mtype == 'NRC':
        rela_rank_x = (x.unsqueeze(-1) - x.unsqueeze(1)).sign()
        rela_rank_y = (y.unsqueeze(-1) - y.unsqueeze(1)).sign()
        corr_mat = torch.triu(rela_rank_x == rela_rank_y, diagonal=1)
        pnum = torch.triu(torch.ones(corr_mat.shape[1:]), diagonal=1).sum()
        metrics['NRC'] = corr_mat.sum(dim=(1, -1)).float() / pnum
    elif mtype == 'ASR':
        metrics['ASR'] = (x == y).float().mean(dim=-1)
    elif mtype.startswith('ASR@'):
        K = int(mtype.strip('ASR@'))
        metrics[mtype.upper()] = ((x - y).abs() <= K).float().mean(dim=-1)
    else:
        raise NotImplementedError
    return metrics

def display_evaluations(metrics):
    for key in sorted(metrics.keys()):
        value = metrics[key].mean()
        print('{:s}: {:.3f}'.format(key, value), end=' ')
    print('')   # start an new line

def save_json(obj, save_pth):
    js = json.dumps(obj, sort_keys=True, indent=4)
    with open(save_pth, 'w') as f:
        f.write(js)

def load_img(load_pth):
    with open(load_pth, 'rb') as f:
        img = Image.open(f).convert('RGB')
        r, g, b = img.split()
        img = Image.merge('RGB', (b, g, r))
        return transforms.ToTensor()(img)

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
    WARNING: 'shuffle' of data_loader must be set to False
    '''
    ret_root = os.path.join(save_root, 'retrieval')
    if not os.path.exists(ret_root): os.makedirs(ret_root)
    ridx = model(query.unsqueeze(0))[0, :K]
    for k in range(K):
        ret_img = data_loader.dataset[ridx[k]][0]
        save_img(ret_img, os.path.join(ret_root, 'rank_{}.jpg'.format(k)))
    save_img(query, os.path.join(ret_root, 'query.jpg'))
