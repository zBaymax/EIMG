import numpy as np
import os
import muspy
from torch.utils.data import Dataset, DataLoader
import xlrd
import torch



class MapDataset(Dataset):
    def __init__(self, midis_path):
        self.midis_path = midis_path
        self.midis = os.listdir(midis_path)
        self.midis.sort()
        valabel = xlrd.open_workbook("emotionmatch/all.xlsx").sheets()[0]
        self.vlabel = valabel.col_values(19)[2:]
        self.alabel = valabel.col_values(20)[2:]


    def __len__(self):
        return len(self.midis)


    def __getitem__(self, index):
        midi = self.midis[index]
        muse_path = os.path.join(self.midis_path, midi)
        music = muspy.read_midi(muse_path)
        rep = muspy.to_note_representation(music)
        rep = np.pad(rep, ((0, 500-rep.shape[0]), (0, 0)), 'constant', constant_values=1)
        # print(rep)
        rep = rep.reshape(1, 2000)
        # print(rep)

        num = int(midi[:-4])
        v = self.vlabel[num - 1]
        a = self.alabel[num - 1]
        va = torch.tensor([[v, a]])
        return rep, va
