import math
import torch
import myutils


class ZOO(object):
    def __init__(self, cfg):
        self.budget    = cfg['budget']   # query budget
        self.N         = cfg['N']        # len of visible list
        self.epsilon   = cfg['epsilon']  # max perb norm
        self.log_freq  = cfg['log_freq'] # log frequency

        self.delta     = cfg.get('delta', 4)
        self.alpha     = cfg.get('alpha', 1e-3)
        self.n_samples = cfg.get('n_samples', 20)
        self.momentum  = cfg.get('momentum', 0.5)

    def normalize(self, x):
        return x / x.norm(p=2, dim=-1, keepdim=True)

    def project(self, o, x):
        x = o + torch.clamp(x - o, -self.epsilon, self.epsilon)
        return x.clamp(0.0, 1.0)

    def run(self, model, criterion, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        adv_x = x.clone()
        grad = torch.zeros_like(x)

        idxs = model(x.reshape(B, C, H, W))
        
        max_iter = self.budget // (self.n_samples + 1)
        for i_iter in range(max_iter):
            metrics = criterion(idxs)
            
            if i_iter % self.log_freq == 0 or i_iter == max_iter - 1:
                print('[Iter {:0>4d}]'.format(i_iter), end='')
                myutils.display_evaluations(metrics)

            grad_new = torch.zeros_like(x)
            for _ in range(self.n_samples):
                basis = self.normalize(torch.randn_like(x))
                bias_adv = (adv_x + self.delta * basis).clamp(0.0, 1.0)
                rank_new = model(bias_adv.reshape(B, C, H, W))
                loss_new = criterion(rank_new)['Loss']
                diff = (loss_new - metrics['Loss']) / self.delta
                grad_new += diff.reshape(-1, 1) * basis
            grad = self.momentum * grad + grad_new
            adv_x = self.project(x, adv_x - self.alpha * grad.sign())
            idxs = model(adv_x.reshape(B, C, H, W))
        
        return adv_x.reshape(B, C, H, W), metrics


class NES(object):
    def __init__(self, cfg):
        self.budget    = cfg['budget']   # query budget
        self.N         = cfg['N']        # len of visible list
        self.epsilon   = cfg['epsilon']  # max perb norm
        self.log_freq  = cfg['log_freq'] # log frequency

        self.sigma     = cfg.get('sigma', 1e-2)
        self.alpha     = cfg.get('alpha', 1e-3)
        self.n_samples = cfg.get('n_samples', 20)
        self.momentum  = cfg.get('momentum', 0.5)

    def project(self, o, x):
        x = o + torch.clamp(x - o, -self.epsilon, self.epsilon)
        return x.clamp(0.0, 1.0)

    def run(self, model, criterion, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        adv_x = x.clone()
        grad = torch.zeros_like(x)

        idxs = model(x.reshape(B, C, H, W))

        max_iter = self.budget // self.n_samples
        for i_iter in range(max_iter):
            metrics = criterion(idxs)
            
            if i_iter % self.log_freq == 0 or i_iter == max_iter - 1:
                print('[Iter {:0>4d}]'.format(i_iter), end='')
                myutils.display_evaluations(metrics)

            grad_new = torch.zeros_like(x)
            for _ in range(self.n_samples // 2):
                noise = torch.randn_like(x)
                lx = (adv_x - self.sigma * noise).clamp(0.0, 1.0)
                rx = (adv_x + self.sigma * noise).clamp(0.0, 1.0)
                l_loss = criterion(model(lx.reshape(B, C, H, W)))['Loss']
                r_loss = criterion(model(rx.reshape(B, C, H, W)))['Loss']
                grad_new += (r_loss - l_loss).reshape(-1, 1) * noise
            grad = self.momentum * grad + grad_new
            adv_x = self.project(x, adv_x - self.alpha * grad.sign())
            idxs = model(adv_x.reshape(B, C, H, W))

        return adv_x.reshape(B, C, H, W), metrics


class SignHunter(object):
    def __init__(self, cfg):
        self.budget   = cfg['budget']   # query budget
        self.N        = cfg['N']        # len of visible list
        self.epsilon  = cfg['epsilon']  # max perb norm
        self.log_freq = cfg['log_freq'] # log frequency

    def run(self, model, criterion, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        perb = torch.ones_like(x)

        adv_x = (x + self.epsilon * perb).clamp(0.0, 1.0)
        idxs = model(adv_x.reshape(B, C, H, W))

        node_i, tree_h = 0, 0
        rs = math.ceil((C * H * W) / (2 ** tree_h))
        bidx = torch.arange(B).view(-1, 1)
        
        for i_iter in range(self.budget):
            metrics = criterion(idxs)
            
            if i_iter % self.log_freq == 0 or i_iter == self.budget - 1:
                print('[Iter {:0>4d}]'.format(i_iter), end='')
                myutils.display_evaluations(metrics)

            # construct new perturbation and calculate loss
            perb_new = perb.clone()
            perb_new[bidx, node_i * rs: (node_i + 1) * rs] *= -1
            adv_x = (x + self.epsilon * perb_new).clamp(0.0, 1.0)
            idxs_new = model(adv_x.reshape(B, C, H, W))
            metrics_new = criterion(idxs_new)
            
            # update if loss is decreased
            improved = metrics_new['Loss'] < metrics['Loss']
            perb[improved] = perb_new[improved]
            idxs[improved] = idxs_new[improved]
            
            node_i += 1
            if node_i == 2 ** tree_h:
                node_i = 0; tree_h += 1
                rs = math.ceil((C * H * W) / (2 ** tree_h))
        
        adv_x = (x + self.epsilon * perb).clamp(0.0, 1.0)
        return adv_x.reshape(B, C, H, W), metrics


class SquareAttack(object):
    def __init__(self, cfg):
        self.budget   = cfg['budget']            # query budget
        self.N        = cfg['N']                 # len of visible list
        self.epsilon  = cfg['epsilon']           # max perb norm
        self.log_freq = cfg['log_freq']          # log frequency
        self.initp    = cfg.get('p', 0.5)        # frac of changed pixs

    def select_p(self, i_iter, max_iter):
        p = self.initp
        i = int(i_iter / max_iter * 2000)
        if i > 10:   p *= 0.9
        if i > 20:   p *= 0.9
        if i > 50:   p *= 0.9
        if i > 100:  p *= 0.8
        if i > 200:  p *= 0.8
        if i > 800:  p *= 0.8
        if i > 1000: p *= 0.7
        if i > 1500: p *= 0.7
        return p

    def square_linf(self, model, criterion, x):
        B, C, H, W = x.shape
        dim = C * H * W
        
        # generate random stripped perturbation
        perb = torch.randint(2, (B, C, 1, W)).repeat(1, 1, H, 1)
        perb = (2 * perb.float() - 1) * self.epsilon
        idxs = model((x + perb).clamp(0.0, 1.0))

        for i_iter in range(self.budget):
            metrics = criterion(idxs)

            if i_iter % self.log_freq == 0 or i_iter == self.budget - 1:
                print('[Iter {:0>4d}]'.format(i_iter), end=' ')
                myutils.display_evaluations(metrics)

            # compute window size S
            p = self.select_p(i_iter, self.budget)
            S = int(round(math.sqrt(p * dim / C)))
            S = min(max(S, 1), H -1)
            # construct new perturbation and calculate loss
            ch = torch.randint(H - S, (B,))
            cw = torch.randint(W - S, (B,))
            bidx = torch.arange(B).reshape(-1, 1, 1, 1)
            cidx = torch.arange(C).reshape(1, -1, 1, 1).repeat(B, 1, 1, 1)
            hidx = torch.arange(S).reshape(1, 1, -1, 1).repeat(B, C, 1, S)
            hidx += ch.reshape(-1, 1, 1, 1).repeat(1, 1, S, S)
            widx = torch.arange(S).reshape(1, 1, 1, -1).repeat(B, C, S, 1)
            widx += cw.reshape(-1, 1, 1, 1).repeat(1, 1, S, S)

            perb_new = perb.clone()
            r_window = (2 * torch.randint(2, (B, C, 1, 1)).float() - 1) * self.epsilon
            perb_new[bidx, cidx, hidx, widx] = r_window
            idxs_new = model((x + perb_new).clamp(0.0, 1.0))
            metrics_new = criterion(idxs_new)

            # update if loss is decreased
            improved = metrics_new['Loss'] < metrics['Loss']
            perb[improved] = perb_new[improved]
            idxs[improved] = idxs_new[improved]
        
        adv_x = (x + perb).clamp(0.0, 1.0)
        return adv_x, metrics

    def run(self, x, c, y):
        return self.square_linf(x, c, y)