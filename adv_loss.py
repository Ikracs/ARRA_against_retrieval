import torch
import myutils


class RL(object):
    '''
    Implementation of Relevance-based Loss
    proposed in 'QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval'
    '''
    def __init__(self, cfg):
        self.N = cfg['N']  # len of visible ranking list

    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate the Relevance-based Loss for untargeted attack.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in the ranking list of clean imgs
        '''
        # calculate prior and conditional prob for each candidate
        logit = torch.arange(self.N, 0, -1)
        logit = 2 ** logit - 1 if self.N <= 16 else logit
        prob = logit.float() / logit.sum()
        prob = torch.cat((prob, torch.zeros(1)))
        prior_prob, cond_prob = prob[rank_s], prob[rank_c]
        # calculate loss according to total prob formula
        return (prior_prob * cond_prob).sum(dim=-1)
    
    def __str__(self):
        return 'rl'


class TRL(object):
    '''
    Implementation of Targeted Relevance-based Loss
    devised for the incapacity of RL to assign candidates to specified ranks other than N + 1
    '''
    def __init__(self, cfg):
        self.N = cfg['N']  # len of visible ranking list
        self.W = cfg.get('W', self.N // 4)  # window size

    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate the Relevance-based Loss for untargeted attack.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in the ranking list of clean imgs
            z: LongTensor of [B, C], idx of each candidate in target ranking list 
        '''
        logit = (self.W - (rank_s - rank_t).abs()).clamp(min=0)
        logit = 2 ** logit - 1 if self.W <= 16 else logit
        prior_prob = logit.float() / (logit.sum(dim=-1, keepdim=True) + 1e-5)
        logit = (rank_c - rank_t).abs()
        logit = 2 ** logit - 1 if self.W <= 16 else logit
        cond_prob = logit.float() / logit.sum(dim=-1, keepdim=True)
        return (prior_prob * cond_prob).sum(dim=-1)
    
    def __str__(self):
        return 'trl'


class SRC(object):
    '''
    Implementation of Short-range Ranking Correlation
    proposed in 'Practical Relative Order Attack in Deep Ranking'.
    '''
    def __init__(self, cfg):
        self.N = cfg['N']  # len of visible ranking list

    def kendall_corr(self, rank_c, rank_t):
        rank_c, rank_t = rank_c.float(), rank_t.float()
        rela_rank_c = (rank_c.unsqueeze(-1) - rank_c.unsqueeze(1)).sign()
        rela_rank_t = (rank_t.unsqueeze(-1) - rank_t.unsqueeze(1)).sign()
        # punish the candidates which are out-of-range
        pidx = (rank_c == self.N).type(torch.float32)
        punish = torch.matmul(pidx.unsqueeze(-1), pidx.unsqueeze(1))
        corr_mat = rela_rank_c * rela_rank_t * (1 - punish) - punish
        corr = torch.triu(corr_mat, diagonal=1).sum(dim=(1, -1))
        pnum = torch.triu(torch.ones(corr_mat.shape[1:]), diagonal=1).sum()
        return corr / pnum

    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate the negative Short-range Ranking Correlation for targeted attack.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        rank_c = myutils.abs2rel(rank_c)
        rank_t = myutils.abs2rel(rank_t)
        return -self.kendall_corr(rank_c, rank_t)
    
    def __str__(self):
        return 'src'


class COMB(object):
    def __init__(self, cfg):
        self.g = cfg.get('gamma', 8) # balance factor

        self.src_loss = SRC(cfg)
        self.trl_loss = TRL(cfg)
    
    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate RARL as 'gamma * TRL + SRC'.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in target ranking list
            z: LongTensor of [B, C], idx of each candidate in target ranking list 
        '''
        l_trl = self.trl_loss(rank_c, rank_s, rank_t)
        l_src = self.src_loss(rank_c, rank_s, rank_t)
        return self.g * l_trl + l_src
    
    def __str__(self):
        return 'comb'


class RRL(object):
    '''Implementation of our Relative Ranking Loss.'''
    def __init__(self, cfg):
        self.N = cfg['N']   # len of visible ranking list
        
        if self.N == 8 or self.N == 10:
            self.k = cfg.get('k', 1)
        elif self.N == 32:
            self.k = cfg.get('k', 0.2)
        elif self.N == 100:
            self.k = cfg.get('k', 0.1)
        else:
            self.k = cfg['k']
    
    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate the Relative Ranking Loss.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        rank_c = myutils.abs2rel(rank_c).float()
        rank_t = myutils.abs2rel(rank_t).float()
        rela_rank_c = rank_c.unsqueeze(-1) - rank_c.unsqueeze(1)
        rela_rank_t = rank_t.unsqueeze(-1) - rank_t.unsqueeze(1)
        prob_c = 1 / (1 + torch.exp(-self.k * rela_rank_c))
        prob_t = 1 / (1 + torch.exp(-self.k * rela_rank_t))
        kl_mat = prob_t * torch.log(prob_t / prob_c)
        # delete the pairs of same candidates
        kl_mat -= kl_mat.diagonal(dim1=1, dim2=-1).diag_embed(dim1=1, dim2=-1)
        pnum = torch.triu(torch.ones(kl_mat.shape[1:]), diagonal=1).sum()
        return kl_mat.sum(dim=(1, -1)) / pnum
    
    def __str__(self):
        return 'rrl'


class ARL(object):
    '''Implementation of our Absoluate Ranking Loss.'''
    def __init__(self, cfg):
        self.N = cfg['N']   # len of visible ranking list
        
        if self.N == 8 or self.N == 10:
            self.k = cfg.get('k', 1)
        elif self.N == 32:
            self.k = cfg.get('k', 0.2)
        elif self.N == 100:
            self.k = cfg.get('k', 0.1)
        else:
            self.k = cfg['k']
    
    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate the Absoluate Ranking Loss.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        rank_c, rank_t = rank_c.float(), rank_t.float()
        prob_c = 1 / (1 + torch.exp(-self.k * (rank_c - self.N)))
        prob_t = 1 / (1 + torch.exp(-self.k * (rank_t - self.N)))
        kl_pairs = prob_t * torch.log(prob_t / prob_c)
        kl_pairs += (1 - prob_t) * torch.log((1 - prob_t) / (1 - prob_c))
        return kl_pairs.sum(dim=-1) / rank_c.shape[1]
    
    def __str__(self):
        return 'arl'


class ARRL(object):
    '''
    Combination of ARL and RRL.
    This loss can be used for both targeted attack and untargeted attack.
    '''
    def __init__(self, cfg):
        self.g = cfg.get('gamma', 8) # balance factor

        self.rel_loss = RRL(cfg)
        self.abs_loss = ARL(cfg)
    
    def __call__(self, rank_c, rank_s, rank_t):
        '''
        calculate RARL as 'RRL + gamma * ARL'.
        Args:
            x: LongTensor of [B, C], idx of each candidate in current ranking list
            y: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        l_rel = self.rel_loss(rank_c, rank_s, rank_t)
        l_abs = self.abs_loss(rank_c, rank_s, rank_t)
        return l_rel + self.g * l_abs
    
    def __str__(self):
        return 'arrl'
