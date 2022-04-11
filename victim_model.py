import os
import time
import torch
from models import BN_Inception
from evaluations import extract_features

import myutils


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, x):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (x - mean) / std


class Model(object):
    def __init__(self, model_type, model_pth):
        '''load parameters for a trained image retrieval model.'''
        if model_type == 'BN-Inception':
            if not os.path.exists(model_pth):
                raise FileNotFoundError
            model = BN_Inception(model_path=model_pth)
        else:
            raise NotImplementedError
        # normlization for CUB and SOP
        norm_layer = Normalize(
            mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
            std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
        self.model = torch.nn.Sequential(norm_layer, model).cuda().eval()

    def extract_gallery_feats(self, data_loader):
        '''extract features for imgs in the gallery.'''
        self.feats = extract_features(self.model, data_loader)[0].cuda()

    def query(self, img_batch):
        '''take a batch of imgs as input and return the ranking list.'''
        output = self.model(img_batch.cuda())
        sim_mat = torch.matmul(output, self.feats.t())
        return sim_mat.argsort(dim=-1, descending=True).cpu()
