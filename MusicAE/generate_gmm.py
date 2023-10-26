import json
import torch
from gmm_model import *
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

# Initialization
with open('gmm_model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('params'):
    os.mkdir('params')

# Model dimensions
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 32
CHROMA_DIMS = 24

# Load model
model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=args['num_clusters'])  


save_path = 'params/music_attr_vae_reg_gmm_long_v_150.pt'
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


print("Loading VGMIDI...")
data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst = get_vgmidi()
vgm_train_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, mode="train")
vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=1, shuffle=is_shuffle, num_workers=0, drop_last=True)
vgm_val_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, mode="val")
vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=1, shuffle=is_shuffle, num_workers=0, drop_last=True)


#############################
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


##########Obtain shifting vectors

mu_r_lst = []
var_r_lst = []
mu_n_lst = []
var_n_lst = []
for k_i in torch.arange(0, 2):
    mu_k = model.mu_r_lookup(k_i.cuda())
    mu_r_lst.append(mu_k.cpu().detach())
    
    var_k = model.logvar_r_lookup(k_i.cuda()).exp_()
    var_r_lst.append(var_k.cpu().detach())
    
    mu_k = model.mu_n_lookup(k_i.cuda())
    mu_n_lst.append(mu_k.cpu().detach())
    
    var_k = model.logvar_n_lookup(k_i.cuda()).exp_()
    var_n_lst.append(var_k.cpu().detach())

r_low_to_high = mu_r_lst[1] - mu_r_lst[0]
r_high_to_low = mu_r_lst[0] - mu_r_lst[1]
n_low_to_high = mu_n_lst[1] - mu_n_lst[0]
n_high_to_low = mu_n_lst[0] - mu_n_lst[1]

#######Load Base Music
# Here, we use a melody segment from the VGMIDI test set
# Choose any number between 0 - 51 for `idx` variable
# Alternatively, you can also encode your desired melody segment using `magenta_encode_midi` in `ptb_v2.py`
# and use the token sequence as `d` here
idx = 3
for cnt in range(0, 5):
    al = [0, 0, 0, 0, 0, 0, 0, 0]
    for j, x in enumerate(vgm_train_dl_dist):
        if j == idx:
            al = x
            break

    d, r, n, a, v, c, r_density, n_density = al
    c = torch.Tensor(c).cuda().unsqueeze(0)

    # Print the encoded event tokens
    eos_index = np.where(d==1)[1][0]
    d_t = d.int().numpy()[0][:eos_index]
    # print("Input tokens:", d_t)

    # Decode it into MIDI and listen the segment
    # Note: you need to pre-install fluidsynth (using apt-get on linux) and pyfluidsynth (using pip)
    d_oh = convert_to_one_hot(torch.Tensor(d).cuda().long(), EVENT_DIMS)

    pm = magenta_decode_midi(notes=d_t, file_path="midi/origin_"+ str(idx) +".mid")  #!

    print("origin "+str(idx), ": ", pm) #!



    ##### reconstruct
    model.eval()

    dis_r, dis_n = model.encode(d_oh)
    z_r = dis_r.rsample()
    z_n = dis_n.rsample()

    # lmbda is a parameter for you to control `how much` is the extent of transfer
    # if you think the transferred arousal of output is not high enough, increase lmbda (and vice versa)
    lmbda = 0.5

    z_r_new = z_r + lmbda*torch.Tensor(r_low_to_high).cuda()
    z_n_new = z_n + lmbda*torch.Tensor(n_low_to_high).cuda()

    c = torch.ones(z_r_new.shape[0], 24).cuda()
    z = torch.cat([z_r_new, z_n_new, c], dim=1)   
 
    out = model.global_decoder(z, steps=300)
    # print("Tokens:", clean_output(out))

    # Listen to the transferred output
    pm = magenta_decode_midi(notes=clean_output(out), file_path="midi/reconstruction_"+ str(idx) +".mid")
    print("reconstruct "+str(idx), ": ", pm)

    idx += 2

