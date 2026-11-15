from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import os
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

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

torch.autograd.set_detect_anomaly(True)

plt.ioff() #turn off matplotlib interactive mode

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

    vocals_array2 = np.zeros((new_array.shape[0], new_array.shape[1]+1))
    vocals_array2[:,-1] = (1-new_array.sum(axis=1))
    vocals_array2[:,:-1] = new_array

    return vocals_array2

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

    silence = len(cmajor_small_ref)+1
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
    midi_list = [x for x in midi_list if x[2]!=17] # remove rests

    return midi_list

section_ref = ['verse1', 'pre_chorus', 'chorus', 'post_chorus', 'verse2']
max_num_notes = 200

# Define your own class LoadFromFolder
class LoadFromFolder(Dataset):
    def __init__(self, main_dir, augment):

        # Set the loading directory
        self.main_dir = main_dir
        self.augment = augment

        # List all images in folder and count them
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        # Return the previously computed number of images
        return len(self.total_imgs)

    def __getitem__(self, idx):
        data_loc = os.path.join(self.main_dir, self.total_imgs[idx])

        data = np.load(data_loc)
        vocals_array = data['vocals_array']
        chords_array = data['chords_array']

        if 'post_chorus' in data_loc:
          section = [0,0,0,1,0]
        elif 'pre_chorus' in data_loc:
          section = [0,1,0,0,0]
        elif 'chorus' in data_loc:
          section = [0,0,1,0,0]
        elif 'verse1' in data_loc or 'verse_1' in data_loc:
          section = [1,0,0,0,0]
        elif 'verse2' in data_loc or 'verse_2' in data_loc:
          section = [0,0,0,0,1]
        else:
          raise Exception('no section found')

        if data['bpm_array'].argmax()<100:
          bpm_array = [1,0,0] # slow
        elif 100<=data['bpm_array'].argmax()<140:
          bpm_array = [0,1,0] # medium
        else:
          bpm_array = [0,0,1] # fast

        vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
        vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
        vocals_array2[:,:-1] = vocals_array
        vocals_array = vocals_array2.copy()

        ############################DATA AUGMENT ###################################
        if self.augment==True:
          if np.random.rand()<0.9: # %90 chance to engage in augmentation
              for i in range(68):
                if chords_array[67-i].max()!=0:
                  notes_end_chords=67-i
                  break

              if notes_end_chords>30: ## Make a random cut from the end
                cut_off_chords = np.random.choice(np.arange(20,notes_end_chords))
                cut_off_vocals = cut_off_chords*8
                vocals_array[cut_off_vocals:,:] = 0
                vocals_array[cut_off_vocals:, -1] = 1
                chords_array[cut_off_chords:,:] = 0
                for i in range(68): # recompute final note
                  if chords_array[67-i].max()!=0:
                    notes_end_chords=67-i
                    break

              if notes_end_chords<58:  ## Add a random string of measures to the end
                  measures = (4)*np.arange(17)
                  measures_vocals = (8*4)*np.arange(17)
                  last_measure = np.argmin(np.abs(measures - notes_end_chords))
                  last_possible_measure = min(2*last_measure-1, 17)
                  end_measure = np.random.choice(np.arange(last_measure+1,last_possible_measure))
                  copy_begin_measure = np.random.choice(np.arange(1,last_measure - (end_measure-last_measure)))
                  copy_end_measure = copy_begin_measure + (end_measure-last_measure)
                  chords_array[measures[last_measure]:measures[end_measure],:] = chords_array[measures[copy_begin_measure]:measures[copy_end_measure],:]
                  vocals_array[measures_vocals[last_measure]:measures_vocals[end_measure],:] = vocals_array[measures_vocals[copy_begin_measure]:measures_vocals[copy_end_measure],:]

              new_array = scramble_array(vocals_array, note_change_fraction=0.1) ## Scramble individual notes with those next to them
        ###############################################################

        num_notes_array = np.zeros(max_num_notes)
        num_notes_array[:int(vocals_array[:,:17].sum())] = 1

        temp = np.asarray([[i,x] for i,x in enumerate(np.squeeze(vocals_array).argmax(axis=1)) if x!=17])

        if len(temp) == 0:
          raise Exception('no notes in vocals_array:', data_loc, vocals_array)

        vocals_array = np.zeros((200,2))
        num_notes = temp.shape[0]

        vocals_array[:num_notes, :] = temp

        return data_loc, from_numpy(vocals_array).float(), from_numpy(chords_array).float(), torch.FloatTensor(section), torch.FloatTensor(bpm_array), torch.FloatTensor(num_notes_array)

! nproc --all

BATCH_SIZE = 450
trainset = LoadFromFolder(main_dir="./midi_ML_data/Train_Cmajor_half_step/", augment=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

valset = LoadFromFolder(main_dir="./midi_ML_data/Validation_Cmajor_half_step/", augment=False)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

SAVE_DIR = './midi_ML_data/Crazy_idea_Cmajor_strict_B/'

# os.mkdir(SAVE_DIR)
# os.mkdir(SAVE_DIR + 'audio_outputs')
# os.mkdir(SAVE_DIR + 'outputs')
# os.mkdir(SAVE_DIR + 'figures')
# os.mkdir(SAVE_DIR + 'weights')
# os.mkdir(SAVE_DIR + 'val_figures')

# ! rm midi_ML_data/Crazy_idea_Cmajor_strict_B/weights/*
# ! rm midi_ML_data/Crazy_idea_Cmajor_strict_B/figures/*

# Size of z latent vector (i.e. size of generator input)
nz = 68*7

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# data = next(iter(trainloader))
# vocals = data[1].to(device)
# chords = data[2].to(device)
# section = data[3].to(device)
# bpm = data[4].to(device)
# num_notes = data[5].to(device)

# dpi = matplotlib.rcParams['figure.dpi']
# height, width = vocals[0].T.shape
# height *=100
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(vocals[0].T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(len(cmajor_small_ref+['sil'])), cmajor_small_ref + ['sil']);
# plt.show()

# dpi = matplotlib.rcParams['figure.dpi']
# height, width = chords.T.shape
# height *=100
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(chords.T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(len(c_major_chords)), c_major_chords);
# plt.show()

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
# height, width = chords.T.shape
# height *=100
# width *= 10
# # What size does the figure need to be in inches to fit the image?
# figsize = width / float(dpi), height / float(dpi)
# # Create a figure of the right size with one axes that takes up the full figure
# fig = plt.figure(figsize=figsize)
# ax = fig.add_axes([0, 0, 1, 1])

# # im = ax.imshow(data[data ind]['tiff' ll:10000,:,:l, origin='lower' )
# im = ax.imshow(chords.T, interpolation='nearest', aspect='auto', origin='lower')

# ax.set_yticks(np.arange(len(c_major_chords)), c_major_chords)
# plt.show()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Dense_Block(nn.Module):
  def __init__(self, in_channels, num_layers, kernal_size, padding, dropout):
      super(Dense_Block, self).__init__()
      self.num_layers = num_layers
      self.leakyrelu = nn.LeakyReLU(inplace = True)
      self.drop_first = nn.Dropout2d(dropout)

      self.conv_first = spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = kernal_size, padding = padding))
      self.bn_first = nn.BatchNorm2d(32)

      self.conv_layers = nn.ModuleList()
      self.bn_layers = nn.ModuleList()
      self.do_layers = nn.ModuleList()
      channels = 32+in_channels
      for i in range(num_layers):
        self.conv_layers.append(spectral_norm(nn.Conv2d(in_channels = channels, out_channels = 32, kernel_size = kernal_size, padding = padding)))
        self.bn_layers.append(nn.BatchNorm2d(32))
        self.do_layers.append(nn.Dropout2d(dropout))
        channels += 32

  def forward(self, x):
      intermediate = torch.cat((self.drop_first(self.leakyrelu(self.bn_first(self.conv_first(x)))), x), 1)

      for i in range(self.num_layers):
        intermediate = torch.cat([intermediate, self.do_layers[i](self.leakyrelu(self.bn_layers[i](self.conv_layers[i](intermediate))))], 1)

      return intermediate

# class PositionalEncoding_BF(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 200):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, embedding_dim]
#         """
#         x = x+self.pe[:,:x.size(1),:]
#         return self.dropout(x)

class PositionalEncoding_simple(nn.Module):

    def __init__(self, max_len: int = 200):
        super().__init__()

        pe = torch.arange(max_len)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        sequence = self.pe[:seq_len].repeat(x.shape[0],1)/seq_len
        x = torch.cat((x, sequence.unsqueeze(2)), dim=2)
        return x

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(8,3,1,1)
# conv_size(7,2,0,1)

# Generator Code
dropout_rate = 0.5

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.section_net = nn.Sequential(
            nn.Linear(5,200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate)
        )

        self.bpm_net = nn.Sequential(
            nn.Linear(3,200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate)
        )


        self.chords_net = nn.Sequential(
            # input state (1) x 68 x 7
            nn.Conv2d(1, 2, (4,2), padding=(0,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state (10) x 65 x 6

            nn.Conv2d(2, 4, (4,2), padding=(0,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (20) x 62 x 5

            nn.Conv2d(4, 8, (4,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (40) x 31 x 4

            nn.Conv2d(8, 16, (3,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (80) x 16 x 3

            nn.Conv2d(16, 32, (4,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (160) x 8 x 2

            nn.Conv2d(32, 200, (3,2), padding=(1,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (200) x 8 x 1

            nn.Flatten(start_dim=2),
            # input state BATCH x (200) x 8

        )

        self.positional_encoding = PositionalEncoding_simple(max_len=200)

        self.trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=12, nhead=6, dim_feedforward=1024, batch_first=True), num_layers=3)

        self.final = nn.Sequential(
            nn.Linear(12,2),
            nn.Tanh()
        )

    def forward(self, noise, chords_array, section, bpm_class, num_notes):

        intermediate = self.chords_net(chords_array)
        intermediate = torch.cat((intermediate, self.section_net(section).unsqueeze(-1), self.bpm_net(bpm_class).unsqueeze(-1), num_notes.unsqueeze(-1)), dim=2)
        intermediate = self.final(self.trans_encoder(self.positional_encoding(intermediate)))

        temp1 = (543/2)*torch.ones((intermediate.shape[0], intermediate.shape[1], 1))
        temp2 = (16/2)*torch.ones((intermediate.shape[0], intermediate.shape[1], 1))
        temp3 = torch.cat((temp1,temp2), dim=2).to(device)

        return torch.mul(intermediate, temp3) + temp3

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)

# data = next(iter(trainloader))
# real_vocals = torch.unsqueeze(data[1],1).to(device)
# real_chords = torch.unsqueeze(data[2],1).to(device)
# real_section = data[3].to(device)
# real_bpm = data[4].to(device)
# real_num_notes = data[5].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# fake = netG(noise, real_chords, real_section, real_bpm, real_num_notes)
# fake.shape

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(16,4,2,2)
# conv_size(5,1,0,2)

from torch.nn.modules.activation import Softmax
dis_dropout = 0.5

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input state (1) x 68 x 7
            nn.Conv2d(1, 4, (4,2), padding=(0,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state (2) x 65 x 6

            nn.Conv2d(4, 8, (4,2), padding=(0,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (4) x 62 x 5

            nn.Conv2d(8, 16, (4,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (8) x 31 x 4

            nn.Conv2d(16, 32, (3,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (16) x 16 x 3

            nn.Conv2d(32, 64, (4,2), padding=(2,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (2) x 9 x 2

            nn.Conv2d(64, 200, (3,2), padding=(1,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # input state (200) x 9

        )

        self.positional_encoding = PositionalEncoding_simple(max_len=200)

        self.trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=12, nhead=3, dim_feedforward=1024, batch_first=True), num_layers=3)


        self.real_fake = nn.Sequential(
            spectral_norm(nn.Linear(200*12, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            spectral_norm(nn.Linear(200*12, 5))
        )

        self.bpm_net = nn.Sequential(
            spectral_norm(nn.Linear(200*12, 3))
        )

        self.num_notes_net = nn.Sequential(
            spectral_norm(nn.Linear(200*12, 1)),
            nn.ReLU()
        )

    def forward(self, vocals_array, chords_array):
        intermediate = torch.cat((torch.squeeze(vocals_array), torch.squeeze(self.chords_net(chords_array))), dim=2)
        intermediate = self.trans_encoder(self.positional_encoding(intermediate))

        return self.real_fake(intermediate.flatten(start_dim=1)), self.section_net(intermediate.flatten(start_dim=1)), self.bpm_net(intermediate.flatten(start_dim=1)), torch.squeeze(self.num_notes_net(intermediate.flatten(start_dim=1)))

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# data = next(iter(trainloader))
# real_vocals = torch.unsqueeze(data[1],1).to(device)
# real_chords = torch.unsqueeze(data[2],1).to(device)
# real_section = data[3].to(device)
# real_bpm = data[4].to(device)
# real_num_notes = data[5].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# fake = netG(noise, real_chords, real_section, real_bpm, real_num_notes)

# output = netD(fake, real_chords)
# output.shape

# lr_sched_gen = np.concatenate((np.arange(0,0.01, 0.0004), 0.01*np.ones(300), 0.001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_gen)

# lr_sched_dis = np.concatenate((np.arange(0,0.001, 0.00004), 0.001*np.ones(300), 0.0001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_dis)

def notes_and_pitches_to_array(note_starts, pitches, real_num_notes):
    combined_tensor = note_starts.unsqueeze(3) * pitches.unsqueeze(2)

    # Expand dimensions of num_notes to match combined_tensor
    expanded_num_notes = real_num_notes.unsqueeze(-1).unsqueeze(-1)

    # Apply filter by multiplying the tensors element-wise
    filtered_combined_tensor = combined_tensor * expanded_num_notes

    summed_tensor = torch.sum(filtered_combined_tensor, dim=1)
    return summed_tensor.unsqueeze(1)

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
optimizerD = optim.Adam(netD.parameters(), lr=0.002, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, 0.999))
##############

# optimizerD = optim.Adam(netD.parameters(), lr=0.003, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

def take_first_n_notes(fake, real_num_notes):
  fake = torch.squeeze(fake)
  return torch.cat((torch.mul(fake[:,:,0], real_num_notes).unsqueeze(-1),torch.mul(fake[:,:,1],real_num_notes).unsqueeze(-1)), dim=2)

def fake_num_notes_loss(fake, real_num_notes):
    temp = (1-real_num_notes)
    temp2 = torch.cat((temp.unsqueeze(-1), temp.unsqueeze(-1)), dim=-1)
    return torch.mean(torch.mul(fake, temp2))

def notes_to_array(vocals):
  temp = torch.zeros(544,17)
  for note_start, pitch in torch.squeeze(torch.round(vocals)):
      if note_start<0:
        note_start = 0
      if note_start>543:
        note_start = 543
      if pitch<0:
        pitch = 0
      if pitch>16:
        pitch = 16

      if pitch>0 and note_start>0:
        temp[int(note_start), int(pitch)] += 1
  return temp

epoch = 520
checkpoint = torch.load(SAVE_DIR + 'weights/' + str(epoch).zfill(6) + '.pt')

netG.load_state_dict(checkpoint['netG_state_dict'])
netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
errD_real = checkpoint['errD_real']
errD_fake = checkpoint['errD_fake']
errG = checkpoint['errG']

lr_dict = {}
for i in np.arange(150, 250):
  lr_dict[str(i)] = (100-(i-150))

# Training Loop

num_epochs=10000
starting_epoch = 521


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
        real_vocals = torch.unsqueeze(data[1],1).to(device)
        real_chords = torch.unsqueeze(data[2],1).to(device)
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
        Discrim_real_num_accu = torch.mean(torch.abs(torch.sum(real_num_notes, axis=1) - output[3])).item()
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) # + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm) + criterion_num(torch.sum(real_num_notes, axis=1), output[3])
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
        # Calculate G's loss based on this output
        errG = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm) + fake_num_accu
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
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  sec_acc:%.2f/%.2f, bpm_acc:%.2f/%.2f, fake_num_loss:%.2f, real_num_loss:%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_sec_accu, fake_sec_accu, real_bpm_accu, fake_bpm_accu, fake_num_accu.item(), Discrim_real_num_accu))

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
            val_vocals = torch.unsqueeze(data[1],1).to(device)
            val_chords = torch.unsqueeze(data[2],1).to(device)
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
            + criterion_sec(torch.squeeze(output_real[1].detach()), val_section)
            + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section)
            + criterion_bpm(torch.squeeze(output_real[2].detach()), val_bpm)
            + criterion_bpm(torch.squeeze(output_fake[2].detach()), val_bpm))

            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)
            print('accuracy_fake_bpm:', (output_fake[2].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_bpm:', (output_real[2].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size)
            print('fake_num_loss:', fake_num_notes_loss(fake, val_num).item())
            print('real_num_loss:', torch.mean(torch.abs(torch.sum(val_num, axis=1) - output_real[3])).item())
            print('real_num_loss_fake:', torch.mean(torch.abs(torch.sum(val_num, axis=1) - output_fake[3])).item())

            val_im = torch.squeeze(notes_to_array(fake[0]).detach().cpu()).T
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

real_vocals.shape

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
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.show()

plot_grad_flow(netD.cpu().named_parameters())

fake
