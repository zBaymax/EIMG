import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from dataset import MapDataset
from losses import SupConLoss

import config
import pdb



class CLloss(nn.Module):
    def __init__(self, img_dim, mus_dim, tau=0.1):
        super().__init__()
        self.imgpro1 = self.projector(in_dim=img_dim, out_dim=config.MID_DIM, use_bias=False, use_bn=True, relu=True)
        self.imgpro2 = self.projector(in_dim=config.MID_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        self.imgpro3 = self.projector(in_dim=config.CL_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        self.muspro = self.projector(in_dim=mus_dim, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        self.demuspro = self.projector(in_dim=config.CL_DIM, out_dim=mus_dim, use_bias=False, use_bn=True, relu=True)
        self.loss_func = SupConLoss(temperature=tau)
        self.rec_loss = nn.MSELoss()
        self.last = None

    def forward(self, pos_img, neg_img, pos_muse, neg_muse, training=True):

        pos_img, neg_img, pos_muse, neg_muse = pos_img.squeeze(), neg_img.squeeze(), pos_muse.squeeze(), neg_muse.squeeze()
        pos_img_emb1 = self.imgpro1(pos_img)
        pos_img_emb2 = self.imgpro2(pos_img_emb1)
        pos_img_emb3 = self.imgpro3(pos_img_emb2)
        neg_img_emb1 = self.imgpro1(neg_img)
        neg_img_emb2 = self.imgpro2(neg_img_emb1)
        neg_img_emb3 = self.imgpro3(neg_img_emb2)

        pos_muse_emb = self.muspro(pos_muse)
        neg_muse_emb = self.muspro(neg_muse)

        # inter-model: 
        intra_feature = torch.cat([pos_img_emb3, pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        intra_feature = F.normalize(intra_feature, dim=2)
        # print(intra_feature.shape)
        intra_label = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        intra_loss = self.loss_func(intra_feature, intra_label)
        # print(intra_loss)

        # img intra-model
        img_feature = torch.cat([pos_img_emb3, neg_img_emb3], dim=0).unsqueeze(dim=1)
        img_feature = F.normalize(img_feature, dim=2)
        img_label = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
        img_loss = self.loss_func(img_feature, img_label)

        # muse intra-model
        muse_feature = torch.cat([pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        muse_feature = F.normalize(muse_feature, dim=2)
        muse_label = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
        muse_loss = self.loss_func(muse_feature, muse_label)

        # reconstruction
        true_muse = torch.cat([pos_muse.squeeze(), neg_muse.squeeze()], dim=0)
        muse = torch.cat([pos_muse_emb, neg_muse_emb], dim=0)

        re_muse = self.demuspro(muse)
        re_loss = self.rec_loss(true_muse, re_muse)


        if training == False:
            pro_muse = self.demuspro(pos_img_emb3)[0]
            return pro_muse

        return intra_loss, img_loss, muse_loss, re_loss



    def projector(self, in_dim, out_dim, use_bias=True, use_bn=False, relu=False):
        net = nn.Sequential()
        net.add_module("FC1", nn.Linear(in_dim, out_dim, bias=use_bias))
        if use_bn:
            net.add_module("BN", nn.BatchNorm1d(out_dim))
        if relu:
            net.add_module("ReLU", nn.ReLU())
        return net

