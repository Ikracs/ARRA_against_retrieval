import torch


class RL(object):
    '''
    Implementation of Relevance-based Loss
    proposed in 'QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval'
    '''
    def __init__(self, cfg):
        self.N = cfg['N']  # len of visible ranking list

    def __call__(self, rank_c, rank_s):
        '''
        calculate the Relevance-based Loss for untargeted attack.
        Args:
            rank_c: LongTensor of [B, C], idx of each candidate in current ranking list
            rank_s: LongTensor of [B, C], idx of each candidate in the ranking list of clean imgs
        '''
        # calculate prior and conditional prob for each candidate
        logit = torch.arange(self.N, 0, -1)
        logit = 2 ** logit - 1 if self.N <= 16 else logit
        prob = logit.float() / logit.sum()
        prob = torch.cat((prob, torch.zeros(1)))
        prior_prob, cond_prob = prob[rank_s], prob[rank_c]
        # calculate loss according to total prob formula
        return (prior_prob * cond_prob).sum(dim=-1)


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

    def __call__(self, rank_c, rank_t):
        '''
        calculate the negative Short-range Ranking Correlation for targeted attack.
        Args:
            rank_c: LongTensor of [B, C], idx of each candidate in current ranking list
            rank_t: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        return -self.kendall_corr(rank_c, rank_t)


class RRL(object):
    '''Implementation of our Relative Ranking Loss.'''
    def __init__(self, cfg):
        self.N = cfg['N']   # len of visible ranking list
        self.k = cfg['k']   # hyperp of sigmoid
    
    def __call__(self, rank_c, rank_t):
        '''
        calculate the Relative Ranking Loss.
        Args:
            rank_c: LongTensor of [B, C], idx of each candidate in current ranking list
            rank_t: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        rank_c, rank_t = rank_c.float(), rank_t.float()
        rela_rank_c = rank_c.unsqueeze(-1) - rank_c.unsqueeze(1)
        rela_rank_t = rank_t.unsqueeze(-1) - rank_t.unsqueeze(1)
        prob_c = 1 / (1 + torch.exp(-self.k * rela_rank_c))
        prob_t = 1 / (1 + torch.exp(-self.k * rela_rank_t))
        kl_mat = prob_t * torch.log(prob_t / prob_c)
        # delete the pairs of same candidates
        kl_mat -= kl_mat.diagonal(dim1=1, dim2=-1).diag_embed(dim1=1, dim2=-1)
        pnum = torch.triu(torch.ones(kl_mat.shape[1:]), diagonal=1).sum()
        return kl_mat.sum(dim=(1, -1)) / pnum


class ARL(object):
    '''Implementation of our Absoluate Ranking Loss.'''
    def __init__(self, cfg):
        self.N  = cfg['N']   # len of visible ranking list
        self.k  = cfg['k']   # hyperp of sigmoid
        self.rb = cfg['rb']  # ratio of bases
        
        self.bases = self._select_bases()
        
    def _select_bases(self):
        BN = int(self.N * self.rb)
        if BN <= 1:  return torch.tensor([self.N], dtype=torch.long)
        else: return torch.linspace(0, self.N, BN, dtype=torch.long)
    
    def __call__(self, rank_c, rank_t):
        '''
        calculate the Absoluate Ranking Loss.
        Args:
            rank_c: LongTensor of [B, C], idx of each candidate in current ranking list
            rank_t: LongTensor of [B, C], idx of each candidate in target ranking list
        '''
        rank_c, rank_t = rank_c.float(), rank_t.float()
        bases = self.bases.unsqueeze(0).repeat(rank_c.shape[0], 1)
        rela_rank_c = rank_c.unsqueeze(-1) - bases.unsqueeze(1)
        rela_rank_t = rank_t.unsqueeze(-1) - bases.unsqueeze(1)
        prob_c = 1 / (1 + torch.exp(-self.k * rela_rank_c))
        prob_t = 1 / (1 + torch.exp(-self.k * rela_rank_t))
        kl_mat = prob_t * torch.log(prob_t / prob_c)
        kl_mat += (1 - prob_t) * torch.log((1 - prob_t) / (1 - prob_c))
        return kl_mat.mean(dim=(1, -1))
