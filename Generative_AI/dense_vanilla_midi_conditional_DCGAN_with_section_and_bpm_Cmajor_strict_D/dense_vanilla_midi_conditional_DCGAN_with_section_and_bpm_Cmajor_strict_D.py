from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import os
from torch import from_numpy
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

def array_to_list(music_array):

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

def scramble_array(vocals_array, note_change_fraction):
    midi_list = array_to_list(vocals_array)
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
        new_array[z[0]:z[1],z[2]]=1

    return new_array

section_ref = ['verse1', 'pre_chorus', 'chorus', 'post_chorus', 'verse2']
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

        ############################DATA AUGMENT ###################################
        if self.augment==True:
          if np.random.rand()<0.9: # %90 chance to engage in augmentation
              for i in range(1632):
                if chords_array[1631-i].max()!=0:
                  notes_end=1631-i
                  break

              if notes_end>800: ## Make a random cut from the end
                cut_off = np.random.choice(np.arange(400,notes_end))
                vocals_array[cut_off:,:] = 0
                chords_array[cut_off:,:] = 0
                for i in range(1632): ## recomupute new notes_end
                  if chords_array[1631-i].max()!=0:
                    notes_end=1631-i
                    break

              if notes_end<1400:  ## Add a random string of measures to the end
                  measures = (24*4)*np.arange(17)
                  last_measure = np.argmin(np.abs(measures - notes_end))
                  last_possible_measure = min(2*last_measure-1, 17)
                  end_measure = np.random.choice(np.arange(last_measure+1,last_possible_measure))
                  copy_begin_measure = np.random.choice(np.arange(1,last_measure - (end_measure-last_measure)))
                  copy_end_measure = copy_begin_measure + (end_measure-last_measure)
                  chords_array[measures[last_measure]:measures[end_measure],:] = chords_array[measures[copy_begin_measure]:measures[copy_end_measure],:]
                  vocals_array[measures[last_measure]:measures[end_measure],:] = vocals_array[measures[copy_begin_measure]:measures[copy_end_measure],:]

              new_array = scramble_array(vocals_array, note_change_fraction=0.1) ## Scramble individual notes with those next to them
        ###############################################################

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

        return data_loc, from_numpy(vocals_array).float(), from_numpy(chords_array).float(), torch.FloatTensor(section), torch.FloatTensor(bpm_array)

! nproc --all

BATCH_SIZE = 64
trainset = LoadFromFolder(main_dir="./midi_ML_data/Train_Cmajor/", augment=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

valset = LoadFromFolder(main_dir="./midi_ML_data/Validation_Cmajor/", augment=False)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

SAVE_DIR = 'midi_ML_data/dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_D/'

# os.mkdir(SAVE_DIR + 'audio_outputs')
# os.mkdir(SAVE_DIR + 'outputs')
# os.mkdir(SAVE_DIR + 'figures')
# os.mkdir(SAVE_DIR + 'weights')
# os.mkdir(SAVE_DIR + 'val_figures')


# for v, data in enumerate(trainloader, 0):
#   if v>10:
#     break

#   if v==0:
#     data_tot = data
#   else:
#     data_tot[0] += data[0]
#     for i in range(len(data_tot)):
#       if i>0:
#         data_tot[i] = torch.concat((data_tot[i],data[i]), dim=0)
#   with torch.no_grad():

#       # Real Batch
#       val_vocals = torch.unsqueeze(data[1],1).to(device)
#       val_chords = torch.unsqueeze(data[2],1).to(device)
#       val_section = data[3].to(device)
#       val_bpm = data[4].to(device)

#       b_size = val_vocals.size(0)
#       print('b_size:', b_size)

#       # Fake Batch
#       noise = torch.randn(b_size, nz, device=device)
#       if v==0:
#         fake = torch.unsqueeze(data[1],1).to(device)
#       else:
#         fake = torch.concat((fake, torch.unsqueeze(data[1],1).to(device)), dim=0)

# np.savez(SAVE_DIR + 'outputs/test_augmentation3.npz', names=data_tot[0], fake=fake.cpu(), val_vocals=data_tot[1].cpu(), val_chords=data_tot[2].cpu(), val_section=data_tot[3].cpu(), val_bpm=data_tot[4].cpu())

# data = next(iter(trainloader))
# print(data[0])
# print(data[1].shape)
# print(data[2].shape)
# print(data[3].shape)
# print(data[4].shape)

# temp=[]
# for data_loc in glob.glob('data/Train/*.npz'):
#     if 'post_chorus' in data_loc:
#       temp.append([0,0,0,1,0])
#     elif 'pre_chorus' in data_loc:
#       temp.append([0,1,0,0,0])
#     elif 'chorus' in data_loc:
#       temp.append([0,0,1,0,0])
#     elif 'verse1' in data_loc:
#       temp.append([1,0,0,0,0])
#     elif 'verse2' in data_loc:
#       temp.append([0,0,0,0,1])
#     else:
#       raise Exception('no section found')

# temp = np.asarray(temp)
# plt.hist(temp.argmax(axis=1))
# plt.xticks(np.arange(5), section_ref);

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(17,3,1,1)
# conv_size(6,4,1,2)

# Generator Code
dropout_rate = 0.5

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input is (1) x 1632 x 7
            nn.Conv2d(1,2, (4,3), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 2 x 816 x 5

            nn.Conv2d(2,8,(4,3), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (8) x 408 x 3

            nn.Conv2d(8,16,(4,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (16) x 204 x 2

            nn.Conv2d(16,32,(4,2), padding=(1,0), stride=(2,1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (32) x 102 x 1
        )

        self.section_net = nn.Sequential(
            nn.Linear(5,51)
        )

        self.bpm_net = nn.Sequential(
            nn.Linear(3,51)
        )

        self.latent = nn.Sequential(
            nn.Linear(nz, 102*1*(1024-32-1)),
        )

        self.gen_net_t1 = nn.Sequential(
            # input state (1024) x 102 x 1
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (512) x 204 x 2
        )

        self.dense_block_t1 = Dense_Block(in_channels=512, num_layers=1, kernal_size=(3,1), padding=(1,0), dropout=dropout_rate)
        # state size. BATCH x (576) x 204 x 2

        self.gen_net_t2 = nn.Sequential(
            nn.ConvTranspose2d(576, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (256) x 408 x 4
        )

        self.dense_block_t2 = Dense_Block(in_channels=256, num_layers=1, kernal_size=(5,3), padding=(2,1), dropout=dropout_rate)
        # state size. BATCH x (320) x 408 x 4

        self.gen_net_t3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (128) x 816 x 8
        )

        self.dense_block_t3 = Dense_Block(in_channels=128, num_layers=1, kernal_size=(7,3), padding=(3,1), dropout=dropout_rate)
        # state size. BATCH x (192) x 816 x 8

        self.gen_net_t4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=(2,3), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (64) x 1632 x 17
        )

        self.dense_block_t4 = Dense_Block(in_channels=64, num_layers=1, kernal_size=(5,3), padding=(2,1), dropout=0.1)
        # state size. BATCH x (128) x 1632 x 17

        self.final = nn.Sequential(
            nn.Conv2d(128,1, (5,3), padding=(2,1), stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            # state size. BATCH x (1) x 1632 x 17
        )

    def forward(self, noise, chords_array, section, bpm_class):
        bpm_section_array = torch.cat((self.section_net(section).view(-1,1,51,1), self.bpm_net(bpm_class).view(-1,1,51,1)), dim=2)
        intermediate = torch.cat((self.chords_net(chords_array), self.latent(noise).view(-1, (1024-32-1), 102, 1), bpm_section_array), dim=1)

        intermediate = self.dense_block_t1(self.gen_net_t1(intermediate))
        intermediate = self.dense_block_t2(self.gen_net_t2(intermediate))
        intermediate = self.dense_block_t3(self.gen_net_t3(intermediate))
        intermediate = self.dense_block_t4(self.gen_net_t4(intermediate))
        return self.final(intermediate)

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

data = next(iter(trainloader))
real_vocals = torch.unsqueeze(data[1],1).to(device)
real_chords = torch.unsqueeze(data[2],1).to(device)
real_section = data[3].to(device)
real_bpm = data[4].to(device)

noise = torch.randn(BATCH_SIZE, nz, device=device)
netG(noise, real_chords, real_section, real_bpm).shape

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(1632,4,1,2)
# conv_size(6,4,1,2)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input state BATCH_SIZE x (1) x 1632 x 7
            spectral_norm(nn.Linear(7, 17)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.4)
            # state BATCH_SIZE x (1) x 1632 x 17
        )

        # concatted input is (2) x 1632 x 17
        self.dense_block_t4 = Dense_Block(in_channels=2, num_layers=1, kernal_size=(5,3), padding=(2,1), dropout=0.4)
        # state size. BATCH x (66) x 1632 x 17

        self.dis_net_t4 = nn.Sequential(
            spectral_norm(nn.Conv2d(66,128, (4,5), padding=(1,1), stride=2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            # state size. (128) x 816 x 8
        )

        self.dense_block_t3 = Dense_Block(in_channels=128, num_layers=1, kernal_size=(7,3), padding=(3,1), dropout=0.4)
        # state size. BATCH x (192) x 816 x 8

        self.dis_net_t3 = nn.Sequential(
            spectral_norm(nn.Conv2d(192,256,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            # state size. (256) x 408 x 4
        )

        self.dense_block_t2 = Dense_Block(in_channels=256, num_layers=1, kernal_size=(5,3), padding=(2,1), dropout=0.4)
        # state size. BATCH x (320) x 408 x 4

        self.dis_net_t2 = nn.Sequential(
            spectral_norm(nn.Conv2d(320,512,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            # state size. (512) x 204 x 2
        )

        self.dense_block_t1 = Dense_Block(in_channels=512, num_layers=1, kernal_size=(3,1), padding=(1,0), dropout=0.4)
        # state size. BATCH x (576) x 204 x 2

        self.dis_net_t1 = nn.Sequential(
            spectral_norm(nn.Conv2d(576,1024,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            # state size. (1024) x 102 x 1

            nn.Flatten(),
        )

        self.real_fake = nn.Sequential(
            spectral_norm(nn.Linear(1024*102, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            spectral_norm(nn.Linear(1024*102, 5))
        )

        self.bpm_net = nn.Sequential(
            spectral_norm(nn.Linear(1024*102, 3))
        )

    def forward(self, vocals_array, chords_array):
        intermediate = torch.cat((vocals_array, self.chords_net(chords_array)), dim=1)
        intermediate = self.dis_net_t4(self.dense_block_t4(intermediate))
        intermediate = self.dis_net_t3(self.dense_block_t3(intermediate))
        intermediate = self.dis_net_t2(self.dense_block_t2(intermediate))
        intermediate = self.dis_net_t1(self.dense_block_t1(intermediate))
        return self.real_fake(intermediate), self.section_net(intermediate), self.bpm_net(intermediate)

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

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# vocals = netG(noise, real_chords, real_section, real_bpm)

# netD(vocals, real_chords)

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

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

##good combos####
##############
optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))
##############

# optimizerD = optim.Adam(netD.parameters(), lr=0.003, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

# Training Loop

num_epochs=10000
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
valD_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(trainloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_vocals = torch.unsqueeze(data[1],1).to(device)
        real_chords = torch.unsqueeze(data[2],1).to(device)
        real_section = data[3].to(device)
        real_bpm = data[4].to(device)
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
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output[0].mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise, real_chords, real_section, real_bpm)
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
        # Calculate G's loss based on this output
        errG = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section) + criterion_bpm(torch.squeeze(output[2]), real_bpm)
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
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  sec_acc:%.2f/%.2f, bpm_acc:%.2f/%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_sec_accu, fake_sec_accu, real_bpm_accu, fake_bpm_accu))

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
                  }, SAVE_DIR + 'weights/' + str(epoch).zfill(6) + '.pt')

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = torch.unsqueeze(data[1],1).to(device)
            val_chords = torch.unsqueeze(data[2],1).to(device)
            val_section = data[3].to(device)
            val_bpm = data[4].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals, val_chords)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section, val_bpm)
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

            plt.figure(figsize=(8, 6), dpi=100)
            plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
            im = np.squeeze(fake[10].detach().cpu()).T
            # im[im<0.5]=0
            plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')
            plt.savefig(SAVE_DIR + 'figures/' +str(epoch).zfill(4) + '.png')

        iters += 1

epoch = 200
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

      b_size = val_vocals.size(0)
      print('b_size:', b_size)
      labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

      # Fake Batch
      noise = torch.randn(b_size, nz, device=device)
      if v==0:
        fake = netG(noise, val_chords, val_section, val_bpm)
      else:
        fake = torch.concat((fake, netG(noise, val_chords, val_section, val_bpm)), dim=0)

np.savez(SAVE_DIR + 'outputs/validation_output' + str(epoch).zfill(4) + '.npz', names=data_tot[0], fake=fake.cpu(), val_vocals=data_tot[1].cpu(), val_chords=data_tot[2].cpu(), val_section=data_tot[3].cpu(), val_bpm=data_tot[4].cpu())

SAVE_DIR

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
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

plot_grad_flow(netD.named_parameters())
