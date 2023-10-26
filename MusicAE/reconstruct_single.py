import json
import torch
from model_v2 import *
import os
from sklearn.model_selection import train_test_split
from dataset_train import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pretty_midi
# from IPython.display import Audio
from tqdm import tqdm
from polyphonic_event_based_v2 import *
from collections import Counter
import matplotlib.pyplot as plt
from polyphonic_event_based_v2 import parse_pretty_midi


import pdb


def m2m(rp_path, muse_path):
    # Initialization
    with open('MUSICAE/model_config_v2.json') as f:
        args = json.load(f)
    if not os.path.isdir('log'):
        os.mkdir('log')
    if not os.path.isdir('params'):
        os.mkdir('params')

    # Model dimensions
    EVENT_DIMS = 342
    RHYTHM_DIMS = 3
    NOTE_DIMS = 16
    CHROMA_DIMS = 24

    # Load model
    model = MusicAttrSingleVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                            chroma_dims=CHROMA_DIMS,
                            hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                            n_step=args['time_step'])


    save_path = 'MUSICAE/params/music_attr_vae_singlevae_8.pt_190.pt'

    state = torch.load(save_path)
    model.load_state_dict(state['model_state_dict'])
    print("Loading params/music_attr_vae_singlevae.pt...")
        

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')

    step, pre_epoch = 0, 0
    batch_size = args["batch_size"]
    is_shuffle = False


    #########
    def convert_to_one_hot(input, dims):
        if len(input.shape) > 1:
            input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
            input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
        else:
            input_oh = torch.zeros((input.shape[0], dims)).cuda()
            input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
        return input_oh

    def clean_output(out):
        recon = np.trim_zeros(torch.argmax(out, dim=-1).cpu().detach().numpy().squeeze())
        if 1 in recon:
            last_idx = np.argwhere(recon == 1)[0][0]
            recon[recon == 1] = 0
            recon = recon[:last_idx]
        return recon

    def repar(mu, stddev, sigma=1):
        eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
        z = mu + stddev * eps  # reparameterization trick
        return z


    list_files = os.listdir(rp_path)
    list_files.sort()
    for cnt, file in enumerate(list_files):
 
        model.eval()

        try:   

            z = torch.load(rp_path + "/" + file).cuda()
            out = model.global_decoder(z, steps=300)

            pm = magenta_decode_midi(notes=clean_output(out), file_path= muse_path + "/"+ file[:-3] +".mid")

        except ValueError as e:
            print("error! ", file)


if __name__=="__main__":
    m2m("rep/", "midi/")
