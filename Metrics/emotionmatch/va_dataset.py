import numpy as np
import os
import muspy
from torch.utils.data import Dataset, DataLoader
import xlrd
import torch



class MapDataset(Dataset):
    def __init__(self, midis_path):

        def dataread(fileName, sheetNum, col_v, col_a, del_num):
            wb = xlrd.open_workbook(fileName)
            sh = wb.sheets()[sheetNum]
            v_list = sh.col_values(col_v)
            a_list = sh.col_values(col_a)
            del v_list[0:del_num]
            del a_list[0:del_num]
            mat = torch.stack([torch.Tensor(v_list), torch.Tensor(a_list)], dim=0)
            return torch.transpose(mat, 0, 1)

        def IAPSread(fileName):
            cnt = 0
            v = []
            a = []
            
            for line in open(fileName):
                seg = line.split("\t")
                v.append(float(seg[2]))
                a.append(float(seg[4]))

            mat = torch.stack([torch.Tensor(v), torch.Tensor(a)], dim=0)
            return torch.transpose(mat, 0, 1)  

        self.midis_path = midis_path
        self.midis = os.listdir(midis_path)
        self.midis.sort()
        NAPS = dataread("evaluation/emotionmatch/NAPS.xls", 0, 17, 21, 3)
        IAPS = IAPSread("evaluation/emotionmatch/data_new.txt")
        self.vlabel = torch.cat([NAPS, IAPS], 0)[:,0]
        self.alabel = torch.cat([NAPS, IAPS], 0)[:,1]
        # print(self.vlabel.shape)
        self.err = 0


    def __len__(self):
        return len(self.midis)


    def __getitem__(self, index):
        midi = self.midis[index]
        muse_path = os.path.join(self.midis_path, midi)

        music = muspy.read_midi(muse_path)
        rep = muspy.to_note_representation(music)
   
        rep = np.pad(rep, ((0, 500-rep.shape[0]), (0, 0)), 'constant', constant_values=1)

        rep = rep.reshape(1, 2000)

        num = int(midi[:-4])
        v = self.vlabel[num - 1]
        a = self.alabel[num - 1]
        va = torch.tensor([[v, a]])
        return rep, va


