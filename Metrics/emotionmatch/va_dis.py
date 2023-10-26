import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from decimal import Decimal
import sys

sys.path.append("evaluation/")
from emotionmatch.va_dataset import MapDataset
from emotionmatch.model import VAPredictor
import io


##
# 给定音乐，输出va距离

##

def train(muse_path, save_path):

    # init model
    vap = VAPredictor()

    state = torch.load(save_path)
    vap.load_state_dict(state['model_state_dict'])

    vap.cuda()
    vap.eval()

    # optimizers

    # dataset
    dataset = MapDataset(muse_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # loss
    # mse = nn.MSELoss()
    ed = nn.PairwiseDistance(p=2)

    # all_loss = 0
    losses = []
    cnt = dict()
    for i in range(0, 10, 2):
        cnt[str(i)] = 0
    for i in range(10, 110, 10):
        cnt[str(i)] = 0
    # print(cnt)
    for idx, (rep, va) in enumerate(loader):
        
        if torch.equal(rep.float(), torch.zeros(1, 1, 2000)):
            # print(rep)
            continue
        rep = rep.float()
        rep, va = rep.cuda(), va.cuda()

        # run
        pre = vap.forward(rep)


        pre = pre.squeeze().unsqueeze(dim=0)
        va = va.squeeze().unsqueeze(dim=0)

        loss = ed(pre, va)

        if loss < 10:
            label = int(loss) if int(loss) % 2 == 0 else (int(loss) - 1) 
            cnt[str(label)] += 1

        losses.append(loss)
        
    losses = torch.Tensor(losses)
    mean_loss = torch.sum(losses) / losses.shape[0]
    mean_loss_max = (torch.sum(losses) - torch.max(losses) - torch.mean(losses)) / (losses.shape[0] - 2)
    mid = torch.median(losses)

    return mean_loss, mean_loss_max, mid, cnt, dataset.err

