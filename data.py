import os
import torch
from DataSet import transforms
from DataSet import MyData

def load_cub():
    DATA_ROOT = 'rdata/CUB/CUB_200_2011'
    label_file = os.path.join(DATA_ROOT, 'test.txt')
    trans = transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize((256)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
    ])
    return MyData(DATA_ROOT, label_file, trans)

def load_sop():
    DATA_ROOT = 'rdata/SOP/Products'
    label_file = os.path.join(DATA_ROOT, 'test.txt')
    trans = transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return MyData(DATA_ROOT, label_file, trans)
