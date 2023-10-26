import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import sys

sys.path.append("evaluation/emotionmatch")
from dataset import MapDataset

import pdb



class VAPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1 = self.projector(in_dim=2000, out_dim=1024, use_bias=False, use_bn=False, relu=True)
        self.FC2 = self.projector(in_dim=1024, out_dim=512, use_bias=False, use_bn=False, relu=True)
        self.FC3 = self.projector(in_dim=512, out_dim=2, use_bias=False, use_bn=False, relu=False)



    def forward(self, muse):
        muse = self.FC1(muse)
        muse = self.FC2(muse)
        pre = self.FC3(muse)
        return pre


    def projector(self, in_dim, out_dim, use_bias=True, use_bn=False, relu=False):
        net = nn.Sequential()
        net.add_module("FC1", nn.Linear(in_dim, out_dim, bias=use_bias))
        if use_bn:
            net.add_module("BN", nn.BatchNorm1d(out_dim))
        if relu:
            net.add_module("ReLU", nn.ReLU())
        return net


