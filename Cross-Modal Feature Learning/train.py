from importlib.resources import path
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import pickle
import numpy as np
from decimal import Decimal

from dataset import MapDataset
from model import CLloss
import config
import io


def train():

    # init model
    clnet = CLloss(img_dim=128, mus_dim=280, tau=0.1)
    clnet.cuda()
    clnet.train()

    # optimizers
    optimizer = optim.Adam(clnet.parameters(), lr = config.init_lr)

    # dataset
    dataset = MapDataset(mode="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # train
    for epoch in range(0, config.EPOCH):
        # intra_loss, img_loss, muse_loss, re_loss
        all_loss = 0
        all_intra_loss = 0
        all_img_loss = 0
        all_muse_loss = 0
        all_re_loss = 0

        for idx, (img_file, pos_img, neg_img, pos_muse, neg_muse) in enumerate(loader):
           
            optimizer.zero_grad()

            pos_img, neg_img, pos_muse, neg_muse = pos_img.cuda(), neg_img.cuda(), pos_muse.cuda(), neg_muse.cuda()

            # run
            intra_loss, img_loss, muse_loss, re_loss = clnet.forward(pos_img, neg_img, pos_muse, neg_muse)

            loss = intra_loss + img_loss + muse_loss + 10 * re_loss

            # update
            loss.backward()
            optimizer.step()

            print("batch loss | {}/{} | {} {} {} {} {}".format(idx, len(loader), loss.item(), intra_loss.item(), img_loss.item(), muse_loss.item(), re_loss.item()))

            all_loss += loss
            all_intra_loss += intra_loss
            all_img_loss += img_loss
            all_muse_loss += muse_loss
            all_re_loss += re_loss

        all_loss = all_loss / len(loader)
        all_intra_loss = all_intra_loss / len(loader)
        all_img_loss = all_img_loss / len(loader)
        all_muse_loss = all_muse_loss / len(loader)
        all_re_loss = all_re_loss / len(loader)
        print("epoch loss | {}/{} | {} {} {} {} {}".format(epoch, config.EPOCH, all_loss, all_intra_loss, all_img_loss, all_muse_loss, all_re_loss))

        if epoch % 5 == 0:

            save_epoch_path = 'params/{}.pt'.format("params" + "_" + str(epoch) + "_" + str(round(all_loss.item(), 3)))
            print("Saving model to... ", save_epoch_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': clnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_loss
            }, save_epoch_path)


if __name__ == "__main__":
    train()