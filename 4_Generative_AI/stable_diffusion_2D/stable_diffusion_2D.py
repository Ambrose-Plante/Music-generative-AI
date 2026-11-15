from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app/midi_ML_data/stable_diffusion_chorus

! pip install denoising_diffusion_pytorch

# ! sed -i '826d' /usr/local/lib/python3.10/dist-packages/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# ! sed -i '825d' /usr/local/lib/python3.10/dist-packages/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# ! sed -i '824d' /usr/local/lib/python3.10/dist-packages/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
! cp ../denoising_diffusion_pytorch.py /usr/local/lib/python3.10/dist-packages/denoising_diffusion_pytorch/

import torch
# from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import os
import glob
from torch import from_numpy, Tensor
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from torchvision.transforms import ToTensor
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
import glob

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

torch.autograd.set_detect_anomaly(True)

plt.ioff() #turn off matplotlib interactive mode

import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt

cmajor_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'E_minor', 'F_major', 'G_major', 'A_minor', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

def scramble_array(vocals_array, note_change_fraction):
    midi_list = vocals_array_to_list(vocals_array)
    num_notes_to_change = round(len(midi_list)*note_change_fraction)
    for change_index in np.random.choice(np.arange(len(midi_list)-1), num_notes_to_change, replace=False):
        len_note1 = midi_list[change_index][1] - midi_list[change_index][0]
        pitch_note1 = midi_list[change_index][2]

        len_note2 = midi_list[change_index+1][1] - midi_list[change_index+1][0]
        pitch_note2 = midi_list[change_index+1][2]

        midi_list[change_index][1] = midi_list[change_index][0] + len_note2
        midi_list[change_index][2] = pitch_note2

        midi_list[change_index+1][0] = midi_list[change_index][1] + 1
        midi_list[change_index+1][1] = midi_list[change_index][1] + len_note1
        midi_list[change_index+1][2] = pitch_note1

    new_array = np.zeros((vocals_array.shape[0], vocals_array.shape[1]))
    for z in midi_list:
        new_array[z[0],z[2]]=1

    return new_array

def vocals_array_to_list(vocals):
  midi_list = [[i,x] for i,x in enumerate(np.squeeze(vocals).argmax(axis=1)) if x!=17]
  midi_list2 = []
  for i in range(len(midi_list)):
      if i<len(midi_list)-1:
        length = min(24, midi_list[i+1][0]-midi_list[i][0])
      else:
        length = min(24, 543 - midi_list[i][0])
      midi_list2.append([midi_list[i][0], midi_list[i][0] + length, midi_list[i][1]])
  return midi_list2

def chords_array_to_list(music_array):

    arr_shape = music_array.shape
    vocals_array = np.zeros((arr_shape[0], arr_shape[1]+1))
    vocals_array[:,:-1] = music_array
    for i,x in enumerate(vocals_array.max(axis=1)):
      if x<0.2:
        vocals_array[i,:-1] = 0
        vocals_array[i,-1] = 1

    vocals_array = np.asarray([[float(x) for x in q] for q in vocals_array])
    vocals_array = vocals_array.argmax(axis=1)

    silence = len(c_major_chords)+1
    for i,x in enumerate(vocals_array):
      if x!=silence:
        note_start=i
        break

    # print(note_start)

    current_note = vocals_array[note_start]
    j=note_start-1 # allows usage of 'j+=1' at the top of loop
    midi_list=[]
    for i,x in enumerate(vocals_array[note_start:]):
      j+=1
      if current_note==silence:
        if x==silence:
          continue # wait until next note
        else: # a new note
          current_note=x
          note_start=j
          continue # nothing more to do
      elif x==current_note: # note is still going, don't do anything
        continue
      else: # note has ended
          midi_list.append([note_start, j, current_note])
          if x==0:
            current_note=silence # set to zero
            continue # wait until next note starts
          else: # directly into new note
            current_note=x
            note_start=j

    if current_note!=silence: # last note was not added
      midi_list.append([note_start, j, current_note])

    midi_list = [[x[0], x[1], x[2]] for x in midi_list]
    midi_list = [x for x in midi_list if x[2]!=silence] # remove rests

    return midi_list

# section_ref = ['verse1', 'pre_chorus', 'chorus', 'post_chorus', 'verse2']
# max_num_notes = 200
# augment=False
# from PIL import Image
# n_files = len(glob.glob('../Train_Cmajor_half_step/*.npz'))

# for file_ind,data_loc in enumerate(glob.glob('../Train_Cmajor_half_step/*.npz')):
#     if 'post_chorus' in data_loc:
#       continue
#     if 'pre_chorus' in data_loc:
#       continue
#     if not 'chorus' in data_loc:
#       continue



#     print(file_ind, '/', n_files)
#     data = np.load(data_loc)
#     vocals_array = data['vocals_array']
#     chords_array = data['chords_array']


#     if 'post_chorus' in data_loc:
#       section = [0,0,0,1,0]
#     elif 'pre_chorus' in data_loc:
#       section = [0,1,0,0,0]
#     elif 'chorus' in data_loc:
#       section = [0,0,1,0,0]
#     elif 'verse1' in data_loc or 'verse_1' in data_loc:
#       section = [1,0,0,0,0]
#     elif 'verse2' in data_loc or 'verse_2' in data_loc:
#       section = [0,0,0,0,1]
#     else:
#       raise Exception('no section found')

#     if data['bpm_array'].argmax()<100:
#       bpm_array = [1,0,0] # slow
#     elif 100<=data['bpm_array'].argmax()<140:
#       bpm_array = [0,1,0] # medium
#     else:
#       bpm_array = [0,0,1] # fast

#     vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
#     vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
#     vocals_array2[:,:-1] = vocals_array
#     vocals_array = vocals_array2.copy()

#     ############################DATA AUGMENT ###################################
#     if augment==True:
#       if np.random.rand()<1.1: # %100 chance to engage in augmentation
#           # print('augmenting...')
#           for i in range(68):
#             if chords_array[67-i].max()!=0:
#               notes_end_chords=67-i
#               break

#           if notes_end_chords>30: ## Make a random cut from the end
#             cut_off_chords = np.random.choice(np.arange(20,notes_end_chords))
#             cut_off_vocals = cut_off_chords*8
#             vocals_array[cut_off_vocals:,:] = 0
#             vocals_array[cut_off_vocals:, -1] = 1
#             chords_array[cut_off_chords:,:] = 0
#             for i in range(68): # recompute final note
#               if chords_array[67-i].max()!=0:
#                 notes_end_chords=67-i
#                 break

#           if notes_end_chords<58:  ## Add a random string of measures to the end
#               measures = (4)*np.arange(17)
#               measures_vocals = (8*4)*np.arange(17)
#               last_measure = np.argmin(np.abs(measures - notes_end_chords))
#               last_possible_measure = min(2*last_measure-1, 17)
#               end_measure = np.random.choice(np.arange(last_measure+1,last_possible_measure))
#               copy_begin_measure = np.random.choice(np.arange(1,last_measure - (end_measure-last_measure)))
#               copy_end_measure = copy_begin_measure + (end_measure-last_measure)
#               chords_array[measures[last_measure]:measures[end_measure],:] = chords_array[measures[copy_begin_measure]:measures[copy_end_measure],:]
#               vocals_array[measures_vocals[last_measure]:measures_vocals[end_measure],:] = vocals_array[measures_vocals[copy_begin_measure]:measures_vocals[copy_end_measure],:]

#           vocals_array = scramble_array(vocals_array, note_change_fraction=0.1) ## Scramble individual notes with those next to them
#     ###############################################################

#     num_notes_array = np.zeros(max_num_notes)
#     num_notes_array[int(vocals_array[:,:17].sum())] = 1

#     #start chords wrap##############################################
#     chords_544 = np.zeros((544,17))
#     for start,end,pitch in [x for x in chords_array_to_list(chords_array) if x[2]!=7]:
#         chords_544[int(start*8):int(end*8),pitch] = 1

#     wrapped_chords_array = np.zeros((128,128))
#     for tele in range(5):
#         wrapped_chords_array[:102,17*tele:(17*tele+17)] = chords_544[102*tele:(102*tele+102),:]
#     tele=5
#     wrapped_chords_array[:34,17*tele:(17*tele+17)] = chords_544[510:,:]
#     #end chords wrap##############################################

#     vocals_array = vocals_array[:,:17]

#     wrapped_vocals_array = np.zeros((128,128))
#     for i in range(5):
#         wrapped_vocals_array[:102,17*i:(17*i+17)] = vocals_array[102*i:(102*i+102),:]
#     i=5
#     wrapped_vocals_array[:34,17*i:(17*i+17)] = vocals_array[510:,:]

#     file_stem = data_loc.split('/')[-1].split('.npz')[0]

#     # im_chords = Image.fromarray(255*wrapped_chords_array.astype(np.uint8))
#     im_voc = Image.fromarray(255*wrapped_vocals_array.astype(np.uint8))

#     # image_concat = Image.new(mode='L', size=(256,128))
#     # image_concat.paste(im_chords, (0,0))
#     # image_concat.paste(im_voc, (128,0))

#     print(data_loc)
#     im_voc.save('data/train/' + str(file_ind).zfill(6) + '.png')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    channels = 1,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    './data/train/',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,             # whether to calculate fid during training
    save_and_sample_every = 1000,
    num_fid_samples = 500
)

trainer.load('35')
# trainer.train()

from accelerate import Accelerator

accelerator = Accelerator(split_batches = True, mixed_precision = 'no')
diffusion = accelerator.prepare(diffusion)
device = accelerator.device
data = torch.load('results/model-17.pt', map_location=device)

diffusion = accelerator.unwrap_model(diffusion)
diffusion.load_state_dict(data['model'])

fake = []

with torch.no_grad():
  for i in range(100):
    print(i)
    fake += [x for x in diffusion.sample(batch_size=100)[:,0].cpu().numpy()]
    torch.cuda.empty_cache()

np.save('outputs/chorus', np.asarray(fake))

def unwrap_vocals_array(wrapped_vocals_array):
    vocals_array = np.zeros((544,17))
    for i in range(5):
        vocals_array[102*i:(102*i+102),:] = wrapped_vocals_array[:102,17*i:(17*i+17)]
    i=5
    vocals_array[510:,:] = wrapped_vocals_array[:34,17*i:(17*i+17)]

    return vocals_array


def array_to_list(vocals_array):
    array = unwrap_vocals_array(vocals_array)
    array[array>=0.5] = 1
    array[array<0.5] = 0

    array_list = []
    for i,col in enumerate(array):
      if sum(col>0.1):
        array_list.append([i*3, col.argmax()])

    if len(array_list)<2:
      return [1,2]

    final_list = []
    for i in range(len(array_list)-1):
      note_len = min(40, array_list[i+1][0]-array_list[i][0])
      final_list.append([array_list[i][0], array_list[i][1]])

    i = len(array_list)-1
    final_list.append([array_list[i][0], array_list[i][1]])

    return final_list

fake = np.load('outputs/chorus.npy')

import imageio.v3 as iio

real_vox = []
for filename in glob.glob('data/train/*'):
  base = filename.split('.png')[0]
  real_vox.append(iio.imread(filename))

real_vox = np.asarray(real_vox)

real_vox[real_vox>=0.5] = 1
real_vox[real_vox<0.5] = 0

fake[fake>=0.5] = 1
fake[fake<0.5] = 0

def num_note_overlap(test_fake, im_stack):
  return np.logical_and(test_fake == 1, im_stack == 1).sum(axis=1).sum(axis=1)

filtered_chorus = np.stack((fake[1], fake[2]))
real_note_lengths = (real_vox==1).sum(axis=1).sum(axis=1)

num_overfit = 0

for i, temp in enumerate(fake):
  print(i, '/', len(fake))
  temp_list = array_to_list(temp)
  if len(temp_list)==2: # array_to_list broken
    print('broken')
    continue
  num_bars = temp_list[-1][0]/96 -1
  if len(temp_list)/num_bars > 2: # at least 2 notes per bar on average
    banana = False

    if (max(num_note_overlap(temp, real_vox) / real_note_lengths) > 0.7) or (max(num_note_overlap(temp, real_vox)) / temp.sum() > 0.7): ## more than %70 overlap with real or vice versa
      print('overfit')
      num_overfit += 1
      banana = True # don't add to the final

    if max(num_note_overlap(temp, filtered_chorus) / (filtered_chorus==1).sum(axis=1).sum(axis=1)) > 0.7: ## more than %70 overlap with real
      print('duplicate')
      banana = True # don't add to the final

    if not banana:
      print('good')
      filtered_chorus = np.vstack((filtered_chorus, temp.reshape(1,128,128)))

np.save('outputs/chorus_filtered', np.asarray(filtered_chorus))

filtered_chorus.shape

dpi = matplotlib.rcParams['figure.dpi']
height, width = fake[0].T.shape
height *=10
width *= 10
# What size does the figure need to be in inches to fit the image?
figsize = width / float(dpi), height / float(dpi)
# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
im = ax.imshow(wrapped_vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')

ax.set_yticks(np.arange(6*len(cmajor_small_ref)), 6*cmajor_small_ref);
plt.show()

# dpi = matplotlib.rcParams['figure.dpi']
# height, width = wrapped_vocals_array.T.shape
# height *=10
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(wrapped_vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(6*len(cmajor_small_ref)), 6*cmajor_small_ref);
# plt.show()

dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array.T.shape
height *=100
width *= 10
# What size does the figure need to be in inches to fit the image?
figsize = width / float(dpi), height / float(dpi)
# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
im = ax.imshow(vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')

ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);
plt.show()

# print(data[0][0])

# dpi = matplotlib.rcParams['figure.dpi']
# height, width = vocals.T.shape
# height *=100
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(vocals.T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(len(cmajor_small_ref+['sil'])), cmajor_small_ref + ['sil'])
# plt.show()

# print(data[0][0])

# dpi = matplotlib.rcParams['figure.dpi']
# height, width = chords[0].T.shape
# height *=100
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(chords[0].T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(len(c_major_chords)), c_major_chords)
# plt.show()

# # assumes shape of input is (BATCH X channels X feature_dim)
# class Channelwise_Linear(nn.Module):
#   def __init__(self, in_channels, out_channels, in_feature_dim, out_feature_dim, drop_rate):
#       super(Channelwise_Linear, self).__init__()
#       self.in_channels = in_channels
#       self.out_channels = out_channels
#       self.in_feature_dim = in_feature_dim
#       self.out_feature_dim = out_feature_dim

#       self.linear_layers = []
#       self.bn_layers = []
#       self.activation_layers = []
#       self.do_layers = []
#       for out_channel in range(out_channels):
#           temp_linear = nn.ModuleList()
#           temp_bn = nn.ModuleList()
#           temp_AL = nn.ModuleList()
#           temp_do = nn.ModuleList()
#           for in_channel in range(in_channels):
#             temp_linear.append(nn.Linear(in_feature_dim, out_feature_dim))
#             temp_bn.append(nn.BatchNorm1d(out_feature_dim))
#             temp_AL.append(nn.LeakyReLU())
#             temp_do.append(nn.Dropout1d(drop_rate))
#           self.linear_layers.append(temp_linear)
#           self.bn_layers.append(temp_bn)
#           self.activation_layers.append(temp_AL)
#           self.do_layers.append(temp_do)

#   def forward(self, x):

#       x = torch.transpose(x,0,1) # now channel is first

#       output_list = []
#       for out_channel in range(self.out_channels):
#           temp = []
#           for in_channel in range(self.in_channels):
#             intermediate = self.linear_layers[out_channel][in_channel](x[in_channel])
#             intermediate = self.bn_layers[out_channel][in_channel](intermediate)
#             intermediate = self.activation_layers[out_channel][in_channel](intermediate)
#             intermediate = self.do_layers[out_channel][in_channel](intermediate)
#             temp.append(intermediate)
#           temp = torch.sum(torch.stack(temp), axis=0) # each out_channel is the sum of the output of the nn.Linear(in_channels), like Conv2D

#           output_list.append(temp)

#       return torch.transpose(torch.stack(output_list), 0, 1)

import torch.nn.functional as F
import math

class Channelwise_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, in_feature_dim, out_feature_dim, drop_rate=False, bn=True, leaky_relu=True):
        super(Channelwise_Linear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.drop_rate=drop_rate
        self.bn = bn
        self.leaky_relu=leaky_relu

        # Weights and biases for linear layers for all in-channels and out-channels.
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, out_feature_dim, in_feature_dim))
        self.bias = nn.Parameter(torch.Tensor(out_channels, in_channels, out_feature_dim))

        self.batch_norm = nn.BatchNorm1d(out_channels * in_channels)
        self.dropout = nn.Dropout(drop_rate)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reshape x and weight to apply linear layer to each pair of out_channel and in_channel
        out = torch.matmul(self.weight, x.view(x.size(0), 1, self.in_channels, -1).unsqueeze(-1)).squeeze(-1) + self.bias

        # Reshape out for batch normalization and dropout
        out = out.view(-1, self.out_channels * self.in_channels, out.size(-1))
        if self.bn:
          out = self.batch_norm(out)
        if self.leaky_relu:
          out = F.leaky_relu(out)
        if self.drop_rate:
          out = self.dropout(out)


        # Reshape out back to original and sum along in_channel dimension
        out = out.view(x.size(0), self.out_channels, self.in_channels, -1)
        out = out.sum(2)
        return out

# Generator Code
dropout_rate = 0.5

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.section_net = nn.Sequential(
            nn.Linear(5,544),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate)
        )

        self.bpm_net = nn.Sequential(
            nn.Linear(3,544),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate)
        )

        self.num_net = nn.Sequential(
            nn.Linear(200,544),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate)
        )

        self.final = nn.Sequential(
            # input state BATCH x (3) x 544
            Channelwise_Linear(in_channels=3, out_channels=32, in_feature_dim=544, out_feature_dim=256, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=32, out_channels=16, in_feature_dim=256, out_feature_dim=128, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=16, out_channels=8, in_feature_dim=128, out_feature_dim=64, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=8, out_channels=4, in_feature_dim=64, out_feature_dim=32, drop_rate=dropout_rate),

            Channelwise_Linear(in_channels=4, out_channels=3, in_feature_dim=32, out_feature_dim=64, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=3, out_channels=2, in_feature_dim=64, out_feature_dim=128, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=2, out_channels=1, in_feature_dim=128, out_feature_dim=256, drop_rate=dropout_rate),
            Channelwise_Linear(in_channels=1, out_channels=1, in_feature_dim=256, out_feature_dim=544, drop_rate=False, bn=False, leaky_relu=False),
        )
        # end state BATCH x 544

    def forward(self, noise, chords_array, section, bpm_class, num_notes):
        intermediate = torch.cat((chords_array.sum(axis=2).unsqueeze(1), noise.view(-1,544).unsqueeze(1), self.num_net(num_notes).unsqueeze(1)), dim=1)
        return torch.squeeze(self.final(intermediate))

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
# netG.apply(weights_init)

# Print the model
print(netG)

# data = next(iter(trainloader))
# real_vocals = data[1].to(device)
# real_chords = data[2].to(device)
# real_section = data[3].to(device)
# real_bpm = data[4].to(device)
# real_num_notes = data[5].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# fake = netG(noise, real_chords, real_section, real_bpm, real_num_notes)
# fake.shape

from torch.nn.modules.activation import Softmax
dis_dropout = 0.5

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.dis_net = nn.Sequential(
            # input state BATCH x (2) x 544 (the chords and vocals concatenated)
            Channelwise_Linear(in_channels=2, out_channels=4, in_feature_dim=544, out_feature_dim=256, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=4, out_channels=8, in_feature_dim=256, out_feature_dim=128, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=8, out_channels=16, in_feature_dim=128, out_feature_dim=64, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=16, out_channels=32, in_feature_dim=64, out_feature_dim=32, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=32, out_channels=64, in_feature_dim=32, out_feature_dim=16, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=64, out_channels=128, in_feature_dim=16, out_feature_dim=8, drop_rate=dis_dropout),
            Channelwise_Linear(in_channels=128, out_channels=256, in_feature_dim=8, out_feature_dim=4, drop_rate=dis_dropout),
        )

        self.real_fake = nn.Sequential(
            Channelwise_Linear(in_channels=256, out_channels=1, in_feature_dim=4, out_feature_dim=1, drop_rate=False, bn=False, leaky_relu=False),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            Channelwise_Linear(in_channels=256, out_channels=1, in_feature_dim=4, out_feature_dim=5, drop_rate=False, bn=False, leaky_relu=False),
            nn.Softmax(dim=2)
        )

        self.bpm_net = nn.Sequential(
            Channelwise_Linear(in_channels=256, out_channels=1, in_feature_dim=4, out_feature_dim=3, drop_rate=False, bn=False, leaky_relu=False),
            nn.Softmax(dim=2)
        )

        self.num_notes_net = nn.Sequential(
            Channelwise_Linear(in_channels=256, out_channels=1, in_feature_dim=4, out_feature_dim=200, drop_rate=False, bn=False, leaky_relu=False),
            nn.Softmax(dim=2)
        )

    def forward(self, vocals_array, chords_array):
        intermediate = torch.cat((vocals_array.unsqueeze(1), chords_array.sum(axis=2).unsqueeze(1)), dim=1)
        intermediate = self.dis_net(intermediate)

        return self.real_fake(intermediate), torch.squeeze(self.section_net(intermediate)), torch.squeeze(self.bpm_net(intermediate)), torch.squeeze(self.num_notes_net(intermediate))

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netD.apply(weights_init)

# Print the model
print(netD)

# data = next(iter(trainloader))
# real_vocals = torch.unsqueeze(data[1],1).to(device)
# real_chords = data[2].to(device)
# real_section = data[3].to(device)
# real_bpm = data[4].to(device)
# real_num_notes = data[5].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# fake = netG(noise, real_chords, real_section, real_bpm, real_num_notes)

# output = netD(fake, real_chords)
# output

# lr_sched_gen = np.concatenate((np.arange(0,0.01, 0.0004), 0.01*np.ones(300), 0.001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_gen)

# lr_sched_dis = np.concatenate((np.arange(0,0.001, 0.00004), 0.001*np.ones(300), 0.0001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_dis)

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Initialize BCELoss function
criterion = nn.BCELoss()
criterion_sec = nn.CrossEntropyLoss()
criterion_bpm = nn.CrossEntropyLoss()
criterion_num = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

##good combos####
##############
# optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.0005, betas=(beta1, 0.999))
##############

# optimizerD = optim.Adam(netD.parameters(), lr=0.003, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

### SGD
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.01, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.005, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

def fake_num_notes_loss(fake, real_num_notes):
    return torch.mean(torch.abs(torch.squeeze(fake).sum(axis=1)-real_num_notes.argmax(axis=1)))

# epoch = 510
# checkpoint = torch.load(SAVE_DIR + 'weights/' + str(epoch).zfill(6) + '.pt')

# netG.load_state_dict(checkpoint['netG_state_dict'])
# netD.load_state_dict(checkpoint['netD_state_dict'])
# optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
# errD_real = checkpoint['errD_real']
# errD_fake = checkpoint['errD_fake']
# errG = checkpoint['errG']

lr_dict = {}
for i in np.arange(150, 250):
  lr_dict[str(i)] = (100-(i-150))

torch.mean(torch.abs(real_num_notes.argmax(axis=1) - output[3].argmax(axis=1)))

# Training Loop

num_epochs=10000
starting_epoch = 0


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
valD_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    np.save('blah', np.array((1,1,1,1)))
    epoch += starting_epoch

    try:
      lr_scaler = lr_dict[str(epoch)]
    except:
      lr_scaler = 1

    print('lr_scaler: ', lr_scaler)

    # For each batch in the dataloader
    for i, data in enumerate(trainloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netG.zero_grad()
        netD.zero_grad()
        # Format batch
        real_vocals = data[1].to(device)
        real_chords = data[2].to(device)
        real_section = data[3].to(device)
        real_bpm = data[4].to(device)
        real_num_notes = data[5].to(device)
        real_num_notes = real_num_notes.requires_grad_()
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_vocals, real_chords)
        # Calculate section label accuracy for real examples
        real_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # Calculate bpm label accuracy for real examples
        real_bpm_accu = (torch.sum(real_bpm.argmax(axis=1)==output[2].argmax(axis=1))/real_bpm.shape[0]).item()
        # Calculate mean num difference
        Discrim_real_num_accu = torch.mean(1.0*torch.abs(real_num_notes.argmax(axis=1) - output[3].argmax(axis=1))).item()
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) + Discrim_real_num_accu # + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output[0].mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise, real_chords, real_section, real_bpm, real_num_notes)
        torch.cuda.empty_cache()
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), real_chords.detach())
        # Calculate section label accuracy for fake examples
        fake_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # Calculate section label accuracy for fake examples
        fake_bpm_accu = (torch.sum(real_bpm.argmax(axis=1)==output[2].argmax(axis=1))/real_bpm.shape[0]).item()
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(torch.squeeze(output[0]), label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output[0].mean().item()
        # Compute error of D as sum over the fake and the real batches

        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, real_chords)
        fake_num_accu = fake_num_notes_loss(fake, real_num_notes)

        Discrim_fake_num_accu = torch.mean(1.0*torch.abs(real_num_notes.argmax(axis=1) - output[3].argmax(axis=1))).item()
        # Calculate G's loss based on this output
        errG = criterion(torch.squeeze(output[0]), label) # + fake_num_accu + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm) + fake_num_accu
        if Discrim_real_num_accu<5:
          errG += Discrim_fake_num_accu
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output[0].mean().item()
        # Update G
        optimizerG.step()

        ind = [ind for ind,x in enumerate(data[0]) if 'Maroon 5_Payphone_chorus_0' in x]
        if ind:
          img_list.append(torch.squeeze(fake[ind].detach().cpu()))

        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  sec_acc:%.2f/%.2f, bpm_acc:%.2f/%.2f, fake_num_loss:%.2f, Discrim_real_num_accu:%.2f/%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_sec_accu, fake_sec_accu, real_bpm_accu, fake_bpm_accu, fake_num_accu.item(), Discrim_real_num_accu, Discrim_fake_num_accu))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # LR update
        # if (epoch > 1) and (epoch % 10 == 0) and (i == len(trainloader)-1):
        #   if D_G_z1<0.1 and D_x>0.9: # D is good, but overwhelming G
        #     optimizerG.param_groups[0]['lr'] *= 1.5
        #     optimizerD.param_groups[0]['lr'] /= 10
        #     print('new opt LRs: ', optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr'])


    if epoch%10==0 and epoch>1:
      torch.save({
                  'epoch': epoch,
                  'netG_state_dict': netG.state_dict(),
                  'netD_state_dict': netD.state_dict(),
                  'optimizerG_state_dict': optimizerG.state_dict(),
                  'optimizerD_state_dict': optimizerD.state_dict(),
                  'G_losses': G_losses,
                  'D_losses': D_losses,
                  'valD_losses': valD_losses,
                  'errD_real': errD_real,
                  'errD_fake': errD_fake,
                  'errG': errG,
                  }, SAVE_DIR + 'weights/' + str(epoch).zfill(6) + '.pt')

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = data[1].to(device)
            val_chords = data[2].to(device)
            val_section = data[3].to(device)
            val_bpm = data[4].to(device)
            val_num = data[5].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals, val_chords)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section, val_bpm, val_num)
            torch.cuda.empty_cache()
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake.detach(), val_chords.detach())
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real)
            + criterion(torch.squeeze(output_fake[0].detach()), labels_fake)
            # + criterion_sec(torch.squeeze(output_real[1].detach()), val_section)
            # + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section)
            # + criterion_bpm(torch.squeeze(output_real[2].detach()), val_bpm)
            # + criterion_bpm(torch.squeeze(output_fake[2].detach()), val_bpm)
            )

            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)
            print('accuracy_fake_bpm:', (output_fake[2].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_bpm:', (output_real[2].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size)
            print('fake_num_loss:', fake_num_notes_loss(fake, val_num).item())
            print('Discrim_real_num_accu:', torch.mean(1.0*torch.abs(val_num.argmax(axis=1) - output_real[3].argmax(axis=1))).item())
            print('Discrim_fake_num_accu:', torch.mean(1.0*torch.abs(val_num.argmax(axis=1) - output_fake[3].argmax(axis=1))).item())

            val_im = torch.squeeze(fake_to_array(fake.detach().cpu())[0]).T
            dpi = matplotlib.rcParams['figure.dpi']
            height, width = val_im.shape
            height *=100
            width *= 10
            # What size does the figure need to be in inches to fit the image?
            figsize = width / float(dpi), height / float(dpi)
            # Create a figure of the right size with one axes that takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])

            # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
            im = ax.imshow(val_im, interpolation='nearest', aspect='auto', origin='lower')

            ax.set_yticks(np.arange(len(cmajor_small_ref+['sil'])), cmajor_small_ref + ['sil'])
            plt.savefig(SAVE_DIR + 'figures/' +str(epoch).zfill(4) + '.png')

        iters += 1

 torch.squeeze(fake_to_array(fake.detach().cpu())[0])[:,10]

epoch = 510
checkpoint = torch.load(SAVE_DIR + 'weights/' + str(epoch).zfill(6) + '.pt')

netG.load_state_dict(checkpoint['netG_state_dict'])
netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

for param_group in optimizerG.param_groups:
    param_group['lr'] = 0.0001

for param_group in optimizerD.param_groups:
    param_group['lr'] = 0.0001

netG.eval()
for v, data in enumerate(valloader, 0):
  if v==0:
    data_tot = data
  else:
    data_tot[0] += data[0]
    for i in range(len(data_tot)):
      if i>0:
        data_tot[i] = torch.concat((data_tot[i],data[i]), dim=0)
  with torch.no_grad():

      # Real Batch
      val_vocals = torch.unsqueeze(data[1],1).to(device)
      val_chords = torch.unsqueeze(data[2],1).to(device)
      val_section = data[3].to(device)
      val_bpm = data[4].to(device)
      val_num = data[5].to(device)

      b_size = val_vocals.size(0)
      print('b_size:', b_size)
      labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

      # Fake Batch
      noise = torch.randn(b_size, nz, device=device)
      if v==0:
        fake = netG(noise, val_chords, val_section, val_bpm, val_num)
      else:
        fake = torch.concat((fake, netG(noise, val_chords, val_section, val_bpm, val_num)), dim=0)

np.savez(SAVE_DIR + 'outputs/validation_output' + str(epoch).zfill(4) + '.npz', names=data_tot[0], fake=fake.cpu(), val_vocals=data_tot[1].cpu(), val_chords=data_tot[2].cpu(), val_section=data_tot[3].cpu(), val_bpm=data_tot[4].cpu(), val_num_notes=data_tot[5].cpu())

fake.backward()

q=10
print(data_tot[0][q])
print(data_tot[4][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data_tot[0][q])['chords_array'].sum(axis=2).T, interpolation='nearest', aspect='auto', origin='lower')

print(data_tot[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
im = np.squeeze(fake[q].detach().cpu()).T
# im[im<0.5]=0
plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')
plt.savefig('midi_ML_data/figures/'+str(epoch).zfill(4)+'.png')

print(data_tot[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data_tot[0][q])['vocals_array'].T, interpolation='nearest', aspect='auto', origin='lower')

!mkdir /content/temp

for q in range(30):
  filename = data[0][q].split('./data/Train/')[1]
  print(filename)
  np.savez_compressed('/content/temp/temp.npz', vocals_array=torch.squeeze(fake.cpu().detach())[q])
  upload_file(filename, '/content/temp/temp.npz', '1FOvSNxUxNv04rr96bNLgnL-y-EkOwYRl')

torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'valD_losses': valD_losses,
            }, 'Models/dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/model_0005_005_batch32_' + str(epoch).zfill(6) + '.pt')

plt.figure(figsize=(10,5))
plt.title("Discriminator Val Loss During Training")
plt.plot(valD_losses,label="D-val")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
# plt.ylim(0,100)
plt.legend()
plt.show()

#%%capture
fig = plt.figure(figsize=(6,6))
plt.axis("off")
ims = [[plt.imshow(i.T, interpolation='nearest', aspect='auto', origin='lower', animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=10, blit=True)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)

a = HTML(ani.to_jshtml())

html = a.data
with open('Maroon 5_Payphone_chorus_0.html', 'w') as f:
    f.write(html)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
              ave_grads.append(p.grad.abs().mean())
              max_grads.append(p.grad.abs().max())
            except:
              pass
    print(len(max_grads))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.show()

plot_grad_flow(netG.cpu().named_parameters())

fake
