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

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

torch.autograd.set_detect_anomaly(True)

! zip -r melody-pro.zip melody-pro/

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

all_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

cmajor_small_ref=[]
for num in np.arange(1,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in all_notes]

# for filename in glob.glob('midi_ML_data/Train2/*.npz'):
#   data=np.load(filename)
#   vocals_array = data['vocals_array']
#   chords_array = data['chords_array']
#   chord_end = np.where(chords_array==1)[0][-1]
#   vocal_end = np.where(vocals_array==1)[0][-1]
#   if np.abs(chord_end-vocal_end)>200:
#     print(filename, chord_end, vocal_end)

section_ref = ['verse1', 'pre_chorus', 'chorus', 'post_chorus', 'verse2']
# Define your own class LoadFromFolder
class LoadFromFolder(Dataset):
    def __init__(self, main_dir):

        # Set the loading directory
        self.main_dir = main_dir

        # List all images in folder and count them
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        # Return the previously computed number of images
        return len(self.total_imgs)

    def __getitem__(self, idx):
        data_loc = os.path.join(self.main_dir, self.total_imgs[idx])

        data = np.load(data_loc)

        if 'post_chorus' in data_loc:
          section = [0,0,0,1,0]
        elif 'pre_chorus' in data_loc:
          section = [0,1,0,0,0]
        elif 'chorus' in data_loc:
          section = [0,0,1,0,0]
        elif 'verse_1' in data_loc or 'verse1' in data_loc:
          section = [1,0,0,0,0]
        elif 'verse_2' in data_loc or 'verse2' in data_loc:
          section = [0,0,0,0,1]
        else:
          raise Exception('no section found')

        vocals_array=data['vocals_array']
        vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
        vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
        vocals_array2[:,:-1] = vocals_array
        return data_loc, from_numpy(vocals_array2).float(), from_numpy(data['chords_array']).float(), torch.FloatTensor(section)

        # return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['chords_array']).float(), torch.FloatTensor(section)

! nproc --all

BATCH_SIZE = 48
trainset = LoadFromFolder(main_dir="./midi_ML_data/Train4/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./midi_ML_data/Validation4/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

# data = next(iter(trainloader))
# print(data[0])
# print(data[1].shape)
# print(data[2].shape)
# print(data[3])

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

import math
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

# Generator Code
dropout_rate = 0.4

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.chords_filter_1 = nn.Sequential(
            # input is (3) x 1632 x 12
            spectral_norm(nn.Conv2d(3,4, (1,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_2 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # state size 1 x 1633 x 12
            spectral_norm(nn.Conv2d(3,4, (2,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_4 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 3), 0),
            # state size 1 x 1635 x 12
            spectral_norm(nn.Conv2d(3,4, (4,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_8 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 7), 0),
            # state size 3 x 1639 x 11
            spectral_norm(nn.Conv2d(3,4, (8,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_16 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 15), 0),
            # state size 1 x 1647 x 12
            spectral_norm(nn.Conv2d(3,4, (16,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_32 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 31), 0),
            # state size 3 x 1663 x 12
            spectral_norm(nn.Conv2d(3,4, (32,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_64 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 63), 0),
            # state size 3 x 1697 x 12
            spectral_norm(nn.Conv2d(3,4, (64,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_net_t1 = nn.Sequential(
            # input is (30) x 1632
            spectral_norm(nn.Conv1d(30,64, kernel_size=4, padding=1, stride=2, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size 64 x 816
        )

        self.chords_net_t2 = nn.Sequential(
            # input is (64) x 816
            spectral_norm(nn.Conv1d(64,128, kernel_size=4, padding=1, stride=2, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size 128 x 408
        )

        self.chords_net_t3 = nn.Sequential(
            # input is (128) x 408
            spectral_norm(nn.Conv1d(128,256, kernel_size=4, padding=1, stride=2, bias=False)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size 256 x 204
        )

        self.chords_net_t4 = nn.Sequential(
            # input is (256) x 204
            spectral_norm(nn.Conv1d(256,512, kernel_size=4, padding=1, stride=2, bias=False)),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size 512 x 102
        )

        self.section_net_t1 = nn.Sequential(
            nn.Linear(5,1632)
        )

        self.latent_t1 = nn.Sequential(
            nn.Linear(nz, 1632),
        )

        self.section_net_t4 = nn.Sequential(
            nn.Linear(5,2*102)
        )

        self.latent_t4 = nn.Sequential(
            nn.Linear(nz, 2*102),
        )

        # self.positional_encoding_t1 = PositionalEncoding_BF(d_model=516, max_len=102)
        # self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=516, nhead=4, dim_feedforward=512, batch_first=True), num_layers=1)

        self.upscale_t4 = nn.Sequential(
            # state size 516 x 102
            spectral_norm(nn.ConvTranspose1d(516, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (256) x 204
        )

        self.refine_t4 = nn.Sequential(
            # state size 516 x 102
            spectral_norm(nn.Conv1d(512,256, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (256) x 204

            spectral_norm(nn.Conv1d(256,256, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (256) x 204
        )

        self.upscale_t3 = nn.Sequential(
            # state size 256 x 204
            spectral_norm(nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (128) x 408
        )

        self.refine_t3 = nn.Sequential(
            # state size 256 x 408
            spectral_norm(nn.Conv1d(256,128, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (128) x 408

            spectral_norm(nn.Conv1d(128,128, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (128) x 408
        )

        self.upscale_t2 = nn.Sequential(
            # state size 128 x 408
            spectral_norm(nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (64) x 816
        )

        self.refine_t2 = nn.Sequential(
            # state size 128 x 408
            spectral_norm(nn.Conv1d(128,64, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (64) x 816

            spectral_norm(nn.Conv1d(64,64, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (64) x 816
        )

        self.upscale_t1 = nn.Sequential(
            # state size 64 x 816
            spectral_norm(nn.ConvTranspose1d(64, 32, 4, 2, 1, bias=False)),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (32) x 1632
        )

        self.refine_t1 = nn.Sequential(
            # state size 62 x 1632
            spectral_norm(nn.Conv1d(62,48, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (48) x 1632

            spectral_norm(nn.Conv1d(48,43, kernel_size=3, padding=1, stride=1, bias=False)),
            nn.BatchNorm1d(43),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=dropout_rate),
            # state size. BATCH x (43) x 1632
        )

        self.final = nn.Sequential(
            # state size 1 x 1632 x 43
            spectral_norm(nn.Conv2d(1,2, kernel_size=3, padding=(1,0), stride=1, bias=False)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (1) x 1632 x 41

            # state size 1 x 1632 x 41
            spectral_norm(nn.Conv2d(2,2, kernel_size=3, padding=(1,0), stride=1, bias=False)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (1) x 1632 x 39

            # state size 1 x 1632 x 39
            spectral_norm(nn.Conv2d(2,1, kernel_size=3, padding=(1,0), stride=1, bias=False)),
            nn.Softmax(dim=3),
            # state size. BATCH x (1) x 1632 x 37
        )


    def forward(self, noise, chords_array, section):
        intermediate = torch.squeeze(torch.cat((self.chords_filter_1(chords_array),
                          self.chords_filter_2(chords_array),
                          self.chords_filter_4(chords_array),
                          self.chords_filter_8(chords_array),
                          self.chords_filter_16(chords_array),
                          self.chords_filter_32(chords_array),
                          self.chords_filter_64(chords_array)), dim=1))
        chords_t0 = torch.cat((intermediate, torch.unsqueeze(self.latent_t1(noise), 1), torch.unsqueeze(self.section_net_t1(section), 1)), dim=1)
        chords_t1 = self.chords_net_t1(chords_t0)
        chords_t2 = self.chords_net_t2(chords_t1)
        chords_t3 = self.chords_net_t3(chords_t2)
        intermediate = self.chords_net_t4(chords_t3)
        intermediate = torch.cat((intermediate, self.latent_t4(noise).view(-1,2,102), self.section_net_t4(section).view(-1,2,102)), dim=1) # BATCH x 516 x 102

        # intermediate = intermediate.permute(0,2,1) # BATCH x 102 x 516
        # intermediate = self.attention_t1(self.positional_encoding_t1(intermediate)).permute(0,2,1) # BATCH x 516 x 102

        intermediate = torch.cat((self.upscale_t4(intermediate), chords_t3), dim=1)
        intermediate = self.refine_t4(intermediate)

        intermediate = torch.cat((self.upscale_t3(intermediate), chords_t2), dim=1)
        intermediate = self.refine_t3(intermediate)

        intermediate = torch.cat((self.upscale_t2(intermediate), chords_t1), dim=1)
        intermediate = self.refine_t2(intermediate)

        intermediate = torch.cat((self.upscale_t1(intermediate), chords_t0), dim=1)
        intermediate = self.refine_t1(intermediate)

        return self.final(torch.permute(torch.unsqueeze(intermediate, dim=3), (0,3,2,1)))

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
# real_chords = torch.permute(data[2].to(device), (0,3,1,2))
# real_section = data[3].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# print(netG(noise, real_chords, real_section).shape)
# print(real_vocals.shape)

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(32,3,0,1)
# conv_size(6,4,1,2)

dis_dropout_rate = 0.5
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_filter_1 = nn.Sequential(
            # input is (3) x 1632 x 12
            spectral_norm(nn.Conv2d(3,4, (1,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_2 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # state size 3 x 1633 x 12
            spectral_norm(nn.Conv2d(3,4, (2,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_4 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 3), 0),
            # state size 3 x 1635 x 12
            spectral_norm(nn.Conv2d(3,4, (4,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 12
        )

        self.chords_filter_8 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 7), 0),
            # state size 1 x 1639 x 12
            spectral_norm(nn.Conv2d(3,4, (8,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_16 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 15), 0),
            # state size 3 x 1647 x 12
            spectral_norm(nn.Conv2d(3,4, (16,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_32 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 31), 0),
            # state size 1 x 1663 x 12
            spectral_norm(nn.Conv2d(3,4, (32,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.chords_filter_64 = nn.Sequential(
            # input is (3) x 1632 x 12
            nn.ConstantPad2d((0, 0, 0, 63), 0),
            # state size 3 x 1697 x 12
            spectral_norm(nn.Conv2d(3,4, (64,12), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_1 = nn.Sequential(
            # input is (1) x 1632 x 37
            spectral_norm(nn.Conv2d(1,4, (1,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_2 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # state size 1 x 1633 x 37
            spectral_norm(nn.Conv2d(1,4, (2,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_4 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 3), 0),
            # state size 1 x 1635 x 37
            spectral_norm(nn.Conv2d(1,4, (4,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_8 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 7), 0),
            # state size 1 x 1639 x 37
            spectral_norm(nn.Conv2d(1,4, (8,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_16 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 15), 0),
            # state size 1 x 1647 x 37
            spectral_norm(nn.Conv2d(1,4, (16,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_32 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 31), 0),
            # state size 1 x 1663 x 37
            spectral_norm(nn.Conv2d(1,4, (32,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        self.vocals_filter_64 = nn.Sequential(
            # input is (1) x 1632 x 37
            nn.ConstantPad2d((0, 0, 0, 63), 0),
            # state size 1 x 1697 x 37
            spectral_norm(nn.Conv2d(1,4, (64,37), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size 4 x 1632 x 1
        )

        # self.positional_encoding_t1 = PositionalEncoding_BF(d_model=56, max_len=1632)
        # self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=56, nhead=4, dim_feedforward=1024, batch_first=True), num_layers=1)

        self.final = nn.Sequential(
            # input size: BATCH x 1 x 56 x 1632
            spectral_norm(nn.Conv2d(1,1, (56,1), padding=(0,0), stride=1, bias=False)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
        )

        self.real_fake = nn.Sequential(
            spectral_norm(nn.Linear(1632, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            spectral_norm(nn.Linear(1632, 5)),
        )

    def forward(self, vocals_array, chords_array):
        chords_intermediate = torch.squeeze(torch.cat((self.chords_filter_1(chords_array), # BATCH x 28 x 1632
                          self.chords_filter_2(chords_array),
                          self.chords_filter_4(chords_array),
                          self.chords_filter_8(chords_array),
                          self.chords_filter_16(chords_array),
                          self.chords_filter_32(chords_array),
                          self.chords_filter_64(chords_array)), dim=1))


        vocals_intermediate = torch.squeeze(torch.cat((self.vocals_filter_1(vocals_array), # BATCH x 28 x 1632
                          self.vocals_filter_2(vocals_array),
                          self.vocals_filter_4(vocals_array),
                          self.vocals_filter_8(vocals_array),
                          self.vocals_filter_16(vocals_array),
                          self.vocals_filter_32(vocals_array),
                          self.vocals_filter_64(vocals_array)), dim=1))

        intermediate = torch.cat((vocals_intermediate, chords_intermediate), dim=1)

        # intermediate = intermediate.permute(0,2,1) # BATCH x 56 x 1632
        # intermediate = self.attention_t1(self.positional_encoding_t1(intermediate)).permute(0,2,1)

        intermediate = torch.unsqueeze(intermediate, dim=1)
        intermediate = torch.squeeze(self.final(intermediate))

        return self.real_fake(intermediate), self.section_net(intermediate)

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
# real_chords = torch.permute(data[2].to(device), (0,3,1,2))
# real_section = data[3].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# vocals = netG(noise, real_chords, real_section)

# netD(vocals, real_chords)

# lr_sched_gen = np.concatenate((np.arange(0,0.01, 0.0004), 0.01*np.ones(300), 0.001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_gen)

# lr_sched_dis = np.concatenate((np.arange(0,0.001, 0.00004), 0.001*np.ones(300), 0.0001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_dis)

# Initialize BCELoss function
criterion = nn.BCELoss()
criterion_sec = nn.CrossEntropyLoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

##good combos####
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)
##############

# optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(0.5, 0.99))
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.005, momentum=0.9)

# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(0.2, 0.95))
##############


### SGD
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.99))
# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.99))

def one_value_loss(vocals, b_size):
    values, indices = torch.max(torch.squeeze(vocals), dim=2)
    return torch.mean(torch.abs(values - torch.ones(b_size, 1632).to(device)))

# checkpoint = torch.load('Models/network_A/model_0001_001_batch64_000050.pt')

# netG.load_state_dict(checkpoint['netG_state_dict'])
# netD.load_state_dict(checkpoint['netD_state_dict'])
# optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

# for param_group in optimizerG.param_groups:
#     param_group['lr'] = 0.0001

# for param_group in optimizerD.param_groups:
#     param_group['lr'] = 0.0001

# Training Loop

num_epochs=10000
noise_width = 0 # control bluring of vocals


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
        real_chords = torch.permute(data[2].to(device), (0,3,1,2))
        real_section = data[3].to(device)
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_vocals + (noise_width*torch.rand((b_size,1,1632,37))-(noise_width*0.5)).to(device), real_chords) # blur vocals
        # Calculate section label accuracy for real examples
        real_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section) ## section more important first here
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output[0].mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise, real_chords, real_section)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach() + (noise_width*torch.rand((b_size,1,1632,37))-(noise_width*0.5)).to(device), real_chords.detach())
        # Calculate section label accuracy for fake examples
        fake_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section)
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
        output = netD(fake + (noise_width*torch.rand((b_size,1,1632,37))-(noise_width*0.5)).to(device), real_chords)
        # Calculate G's loss based on this output
        OV_loss = one_value_loss(fake, b_size)
        errG = criterion(torch.squeeze(output[0]), label) + 0.25*criterion_sec(torch.squeeze(output[1]), real_section) + OV_loss ## section less important in beginning
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output[0].mean().item()
        # Update G
        optimizerG.step()

        ind = [ind for ind,x in enumerate(data[0]) if 'Maroon 5_Payphone_chorus_0' in x]
        if ind:
          img_list.append(torch.squeeze(fake[ind].detach().cpu()))

        # Output training stats
        if i % 2 == 0:
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  sec_acc:%.2f/%.2f  OVL:%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_sec_accu, fake_sec_accu, OV_loss.item()))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # LR update
        # if (epoch > 1) and (epoch % 10 == 0) and (i == len(trainloader)-1):
        #   if D_G_z1<0.1 and D_x>0.9: # D is good, but overwhelming G
        #     optimizerG.param_groups[0]['lr'] *= 1.5
        #     optimizerD.param_groups[0]['lr'] /= 10
        #     print('new opt LRs: ', optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr'])

    # if epoch%10==0:
    #   torch.save({
    #               'epoch': epoch,
    #               'netG_state_dict': netG.state_dict(),
    #               'netD_state_dict': netD.state_dict(),
    #               'optimizerG_state_dict': optimizerG.state_dict(),
    #               'optimizerD_state_dict': optimizerD.state_dict(),
    #               'G_losses': G_losses,
    #               'D_losses': D_losses,
    #               'valD_losses': valD_losses,
    #               }, 'Models/network_A/model_0001_001_batch64_' + str(epoch).zfill(6) + '.pt')

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = torch.unsqueeze(data[1],1).to(device)
            val_chords = torch.permute(data[2].to(device), (0,3,1,2))
            val_section = data[3].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals + (noise_width*torch.rand((b_size,1,1632,37))-(noise_width*0.5)).to(device), val_chords)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake.detach() + (noise_width*torch.rand((b_size,1,1632,37))-(noise_width*0.5)).to(device), val_chords.detach())
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real) + criterion(torch.squeeze(output_fake[0].detach()), labels_fake) + criterion_sec(torch.squeeze(output_real[1].detach()), val_section) + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section))
            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)

        iters += 1

netG.eval()
for v, data in enumerate(valloader, 0):
  if v==0:
    data_tot = data
  else:
    for i,subtensor in enumerate(data_tot):
      if i==0:
        subtensor += data[i]
      else:
        subtensor = torch.concat((subtensor,data[i]), dim=0)
  with torch.no_grad():

      # Real Batch
      val_vocals = torch.unsqueeze(data[1],1).to(device)
      val_chords = torch.permute(data[2].to(device), (0,3,1,2))
      val_section = data[3].to(device)

      b_size = val_vocals.size(0)
      print('b_size:', b_size)
      labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

      # Fake Batch
      noise = torch.randn(b_size, nz, device=device)
      if v==0:
        fake = netG(noise, val_chords, val_section)
      else:
        fake = torch.concat((fake, netG(noise, val_chords, val_section)), dim=0)

q=5
print(data_tot[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data_tot[0][q])['chords_array'].sum(axis=2).T, interpolation='nearest', aspect='auto', origin='lower')

print(data_tot[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
im = np.squeeze(fake[q].detach().cpu()).T
# im[im<0.5]=0
plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')

im.sum()

print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['vocals_array'].T, interpolation='nearest', aspect='auto', origin='lower')

!mkdir /content/temp

for q in range(30):
  filename = data[0][q].split('./data/Train/')[1]
  print(filename)
  np.savez_compressed('/content/temp/temp.npz', vocals_array=torch.squeeze(fake.cpu().detach())[q])
  upload_file(filename, '/content/temp/temp.npz', '1FOvSNxUxNv04rr96bNLgnL-y-EkOwYRl')

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
ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=1, blit=True)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)

a = HTML(ani.to_jshtml())

# html = a.data
# with open('Maroon 5_Payphone_chorus_0.html', 'w') as f:
#     f.write(html)

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
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

plot_grad_flow(netG.cpu().named_parameters())
