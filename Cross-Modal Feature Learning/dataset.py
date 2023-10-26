import numpy as np
import config
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, mode="train"):

        # alae
        # img = np.load(config.IMAGE_DATA_ROOT, allow_pickle=True).item()
        # self.img = img
        # vqvae/bvae
        img = torch.load(config.IMAGE_DATA_ROOT)
        self.img = img

        muse = np.load(config.MUSE_ROOT,allow_pickle=True)
        self.muse = torch.from_numpy(muse)

        if mode == "train":
            img_pos_pairs = open(config.IMG_POS_ROOT,"r",encoding='utf-16')
            self.img_pos_pairs = img_pos_pairs.readlines()

            img_neg_pairs = open(config.IMG_NEG_ROOT,"r",encoding='utf-16')
            self.img_neg_pairs = img_neg_pairs.readlines()


            muse_pos_pairs = open(config.MUSE_POS_ROOT,"r",encoding='utf-16')
            self.muse_pos_pairs = muse_pos_pairs.readlines()

            muse_neg_pairs = open(config.MUSE_NEG_ROOT, "r",encoding='utf-16')
            self.muse_neg_pairs = muse_neg_pairs.readlines()

        elif mode == "test":
            img_pos_pairs = open(config.TEST_IMG_POS_ROOT,"r",encoding='utf-16')
            self.img_pos_pairs = img_pos_pairs.readlines()

            img_neg_pairs = open(config.TEST_IMG_NEG_ROOT,"r",encoding='utf-16')
            self.img_neg_pairs = img_neg_pairs.readlines()


            muse_pos_pairs = open(config.TEST_MUSE_POS_ROOT,"r",encoding='utf-16')
            self.muse_pos_pairs = muse_pos_pairs.readlines()

            muse_neg_pairs = open(config.TEST_MUSE_NEG_ROOT, "r",encoding='utf-16')
            self.muse_neg_pairs = muse_neg_pairs.readlines()


    def __len__(self):
        return len(self.muse_pos_pairs)


    def __getitem__(self, index):

        # file name get
        img_num_int = self.img_pos_pairs[index].strip().split(" ")[0]           # 1
        img_file = str((int(img_num_int) + 1)).zfill(4) + ".jpg"    # 2.jpg
        # print(img_num_int)
        # print(img_file)
        
        # intra-img
        # print(self.img_pos_pairs[index])

        # alae
        # pos_img = []
        # for indx, i in enumerate(self.img_pos_pairs[index].strip().split(" ")):
        #     if indx == 0:
        #         continue
        #     i = str(int(i) + 1).zfill(4)
        #     x = self.img.get(i)
        #     pos_img.append(x)
        
        # pos_img = torch.tensor(pos_img).squeeze(dim=1)
        # # print(pos_img.shape)

        # # print(self.img_neg_pairs[index])
        # neg_img = []
        # for indx, i in enumerate(self.img_neg_pairs[index].strip().split(" ")):
        #     if indx == 0:
        #         continue
        #     i = str(int(i) + 1).zfill(4)
        #     x = self.img.get(i)
        #     neg_img.append(x)
        
        # neg_img = torch.tensor(neg_img).squeeze(dim=1)
        # print(neg_img.shape)

        # vqvae/beta-vae
        pos_img_num = self.img_pos_pairs[index].strip().split(" ")[1:]
        pos_img = torch.cat([self.img[int(i)].unsqueeze(0) for i in pos_img_num], dim=0)

        neg_img_num = self.img_neg_pairs[index].strip().split(" ")[1:]
        neg_img = torch.cat([self.img[int(i)].unsqueeze(0) for i in neg_img_num], dim=0)

        # inter-model
        pos_muse_num = self.muse_pos_pairs[index].strip().split(" ")[1:]
        pos_muse = torch.cat([self.muse[int(i)].unsqueeze(0) for i in pos_muse_num], dim=0)
        # print(pos_muse.shape)

        neg_muse_num = self.muse_neg_pairs[index].strip().split(" ")[1:]
        neg_muse = torch.cat([self.muse[int(i)].unsqueeze(0) for i in neg_muse_num], dim=0)
        # print(neg_muse.shape)


        return img_file, pos_img, neg_img, pos_muse, neg_muse


if __name__ == "__main__":
    dataset = MapDataset(mode="train")
    # print(len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)



    for i, (img_file, pos_img, neg_img, pos_muse, neg_muse) in enumerate(loader):
        print(pos_img.shape)
        print(neg_img.shape)
        print(pos_muse.shape)
        print(neg_muse.shape)
        break
        # a = 1
