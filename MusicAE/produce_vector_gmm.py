import json
import torch
from torch.autograd import grad
from gmm_model import *
import os
from sklearn.model_selection import train_test_split
from dataset_process import *
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

# Initialization
with open('gmm_model_config.json') as f:
    args = json.load(f)

# Model dimensions
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24

# Load model
model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=2)  

save_path = 'params/music_attr_vae_reg_gmm_long_v_100.pt'
state = torch.load(save_path)
model.load_state_dict(state['model_state_dict'])
print("Loading params/music_attr_vae_reg_gmm.pt...")
    

if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
is_shuffle = False

# ================ In this example, we will load only the examples from VGMIDI dataset ========== #
print("Loading VGMIDI...")
data_lst, rhythm_lst, note_density_lst, chroma_lst, arousal_lst, valence_lst = get_vgmidi()
vgm_train_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst)
vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=1, shuffle=False, num_workers=0)


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



al = [0, 0, 0, 0, 0, 0, 0, 0]
muse = []
for j, x in enumerate(vgm_train_dl_dist):
    al = x

    d, r, n, a, v, c, r_density, n_density = al
    c = torch.Tensor(c).cuda().unsqueeze(0)

    # Print the encoded event tokens
    eos_index = np.where(d==1)[1][0]
    d_t = d.int().numpy()[0][:eos_index]
    # print("Input tokens:", d_t)

    # Decode it into MIDI and listen the segment
    # Note: you need to pre-install fluidsynth (using apt-get on linux) and pyfluidsynth (using pip)
    d_oh = convert_to_one_hot(torch.Tensor(d).cuda().long(), EVENT_DIMS)
    # pdb.set_trace()


    ##### reconstruct
    model.eval()
    # dis_r, dis_n = model.encode(d_oh.unsqueeze(0))
    with torch.no_grad():
        dis_r, dis_n = model.encode(d_oh)
    z_r = dis_r.rsample()
    z_n = dis_n.rsample()


    c = torch.ones(z_r.shape[0], 24).cuda()
    z = torch.cat([z_r, z_n, c], dim=1)        


    print("{} finished.".format(j))

    muse.append(z.squeeze(0))



muse = torch.tensor( [item.cpu().detach().numpy() for item in muse] ).numpy()
np.save("my_muse_data/muse.npy", muse)

print(muse.shape)
