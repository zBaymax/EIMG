import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from decimal import Decimal

from dataset import MapDataset
from model import VAPredictor
import io


def train():

    # init model
    vap = VAPredictor()
    vap.cuda()
    vap.train()

    # optimizers
    optimizer = optim.Adam(vap.parameters(), lr = 0.0001)

    # dataset
    dataset = MapDataset("dataset/piano_version")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # loss
    mse = nn.MSELoss()

    EPOCH = 300
    # train
    for epoch in range(0, EPOCH):
        all_loss = 0

        for idx, (rep, va) in enumerate(loader):
           

            rep = rep.float()
            rep, va = rep.cuda(), va.cuda()

            # run
            pre = vap.forward(rep)
            loss = mse(pre, va)
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("batch loss | {}/{} | {}".format(idx, len(loader), loss.item()))

            all_loss += loss


        all_loss = all_loss / len(loader)


        print("epoch loss | {}/{} | {}".format(epoch, EPOCH, all_loss))

        save_epoch_path = 'DMusicVAPredictor/{}.pt'.format("params" + "_" + str(epoch) + "_" + str(round(all_loss.item(), 3)))
        print("Saving model to... ", save_epoch_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': vap.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': all_loss
        }, save_epoch_path)



if __name__ == "__main__":
    train()