from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# def upload_file(filename, file_loc, dir_id):
#   uploaded = drive.CreateFile({'title': filename, 'parents': [{'id': dir_id}]})
#   uploaded.SetContentFile(file_loc)
#   uploaded.Upload()
#   # print('Uploaded file with ID {}'.format(uploaded.get('id')))

import os
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from torchvision.transforms import ToTensor
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np

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

cmajor_scale = ['C', 'D', 'D#', 'E', 'F', 'G', 'A', 'B']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

# ! mv data/Train/Avicii_Wake\ Me\ Up_chorus_0.npz data/Val/ ## 1631
# ! mv data/Train/Ed\ Sheeran_Shape\ of\ You_chorus_*.npz data/Val/ ## 863
# ! mv data/Train/Jonas\ Brothers_What\ a\ man\ gotta\ do_verse*.npz data/Val/ ## 1247
# ! mv data/Train/Justin\ Bieber_Peaches_chorus*.npz data/Val/ ## 479

# for filename in glob.glob('data/*.npz'):
#   data=np.load(filename)
#   vocals_array = data['vocals_array']
#   chords_array = data['chords_array']
#   chord_end = np.where(chords_array==1)[0][-1]
#   vocal_end = np.where(vocals_array==1)[0][-1]
#   # if np.abs(chord_end-vocal_end)>200:
#   print(filename, chord_end, vocal_end)

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
        elif 'verse1' in data_loc:
          section = [1,0,0,0,0]
        elif 'verse2' in data_loc:
          section = [0,0,0,0,1]
        else:
          raise Exception('no section found')

        return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['chords_array']).float(), torch.FloatTensor(section)

BATCH_SIZE = 16
trainset = LoadFromFolder(main_dir="./data/Train/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./data/Val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# names = ['vocals_array', 'chords_array']
# data = next(iter(trainloader))
# print(data[0][0])
# print(data[1][0].shape)
# print(data[2][0].shape)
# print(data[3][0])
# print(data[4][0])

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

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

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
  def __init__(self, in_channels, num_layers, kernal_size, padding):
      super(Dense_Block, self).__init__()
      self.num_layers = num_layers
      self.leakyrelu = nn.LeakyReLU(inplace = True)
      
      self.conv_first = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = kernal_size, padding = padding)
      self.bn_first = nn.BatchNorm2d(32)

      self.conv_layers = nn.ModuleList()
      self.bn_layers = nn.ModuleList()
      channels = 32+in_channels
      for i in range(num_layers):
        self.conv_layers.append(nn.Conv2d(in_channels = channels, out_channels = 32, kernel_size = kernal_size, padding = padding))
        self.bn_layers.append(nn.BatchNorm2d(32))
        channels += 32

  def forward(self, x):
      intermediate = torch.cat((self.leakyrelu(self.bn_first(self.conv_first(x))), x), 1)

      for i in range(self.num_layers):
        intermediate = torch.cat([intermediate, self.leakyrelu(self.bn_layers[i](self.conv_layers[i](intermediate)))], 1)
   
      return intermediate

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input is (1) x 1632 x 8
            nn.Conv2d(1,4, (4,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size 4 x 816 x 4 

            nn.Conv2d(4,8,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. (8) x 408 x 3

            nn.Conv2d(8,16,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. (16) x 204 x 2

            nn.Conv2d(16,32,(4,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. (32) x 102 x 1
        )

        self.section_net = nn.Sequential(
            nn.Linear(5,102)
        )

        self.latent = nn.Sequential(
            nn.Linear(nz, 102*1*(1024-32-1)),
        )

        self.gen_net_t1 = nn.Sequential(
            # input state (1024) x 102 x 1
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (512) x 204 x 2
        )

        self.dense_block_t1 = Dense_Block(in_channels=512, num_layers=3, kernal_size=(3,1), padding=(1,0))
        # state size. BATCH x (640) x 204 x 2

        self.gen_net_t2 = nn.Sequential(
            nn.ConvTranspose2d(640, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (256) x 408 x 4
        )

        self.dense_block_t2 = Dense_Block(in_channels=256, num_layers=3, kernal_size=(5,3), padding=(2,1))
        # state size. BATCH x (384) x 408 x 4

        self.gen_net_t3 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (128) x 816 x 8
        )

        self.dense_block_t3 = Dense_Block(in_channels=128, num_layers=3, kernal_size=(7,3), padding=(3,1))
        # state size. BATCH x (256) x 816 x 8
        
        self.gen_net_t4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(2,5), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # state size. BATCH x (64) x 1632 x 19
        )

        self.dense_block_t4 = Dense_Block(in_channels=64, num_layers=3, kernal_size=(9,3), padding=(4,1))

        self.final = nn.Sequential(
            nn.Conv2d(192,1, (9,3), padding=(4,1), stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 1632 x 19
        )

    def forward(self, noise, chords_array, section, tier):
        intermediate = torch.cat((self.chords_net(chords_array), self.latent(noise).view(-1, (1024-32-1), 102, 1), self.section_net(section).view(-1,1,102,1)), dim=1)
        if tier==1:
          return self.gen_net_t1(intermediate)
        elif tier==2:
          return self.gen_net_t2(self.gen_net_t1(intermediate))
        elif tier==3:
          return self.gen_net_t3(self.gen_net_t2(self.gen_net_t1(intermediate)))
        elif tier==4:
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

# chords_noise = torch.randn(BATCH_SIZE, 1, 1632, 8, device=device)
# noise = torch.randn(BATCH_SIZE, nz, device=device)
# section_noise = torch.randn(BATCH_SIZE, 5, device=device)
# tier = torch.ones(1)*4
# netG(noise, chords_noise, section_noise, tier).shape

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_net_t4 = nn.Sequential(
            # input state BATCH_SIZE x (1) x 1632 x 8
            spectral_norm(nn.Linear(8, 19)),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2)
            # state BATCH_SIZE x (1) x 1632 x 19
        )

        self.chords_net_t3 = nn.Sequential(
            # state BATCH_SIZE x (1) x 1632 x 19
            spectral_norm(nn.Conv2d(1,4, (4,5), padding=(1,0), stride=2)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state BATCH_SIZE x (4) x 816 x 8
        )

        self.chords_net_t2 = nn.Sequential(
            # state BATCH_SIZE x (4) x 816 x 8
            spectral_norm(nn.Conv2d(4,8, (4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state BATCH_SIZE x (8) x 408 x 4
        )

        self.dis_net_t4 = nn.Sequential(
            # concatted input is (2) x 1632 x 19
            spectral_norm(nn.Conv2d(2,64, (7,5), padding=(3,2), stride=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (64) x 1632 x 19

            spectral_norm(nn.Conv2d(64,204, (4,5), padding=(1,0), stride=2)),
            nn.BatchNorm2d(204),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (128) x 816 x 8
        )

        self.dis_net_t3 = nn.Sequential(
            spectral_norm(nn.Conv2d(204,200, (5,3), padding=(2,1), stride=1)),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (200) x 816 x 8

            spectral_norm(nn.Conv2d(200,208,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(208),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (208) x 408 x 4
        )

        self.dis_net_t2 = nn.Sequential(
            spectral_norm(nn.Conv2d(208,350, (5,1), padding=(2,0), stride=1)),
            nn.BatchNorm2d(350),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (350) x 408 x 4

            spectral_norm(nn.Conv2d(350,512,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (512) x 204 x 2
        )

        self.dis_net_t1 = nn.Sequential(
            spectral_norm(nn.Conv2d(512,800, (3,1), padding=(1,0), stride=1)),
            nn.BatchNorm2d(800),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (350) x 204 x 2

            spectral_norm(nn.Conv2d(800,1024,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
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

    def forward(self, vocals_array, chords_array, real, tier):
        if real==1:
          intermediate = torch.cat((vocals_array, self.chords_net_t4(chords_array)), dim=1)
          intermediate = self.dis_net_t1(self.dis_net_t2(self.dis_net_t3(self.dis_net_t4(intermediate))))
        elif tier==4:
          intermediate = torch.cat((vocals_array, self.chords_net_t4(chords_array)), dim=1)
          intermediate = self.dis_net_t1(self.dis_net_t2(self.dis_net_t3(self.dis_net_t4(intermediate))))
        elif tier==3:
          intermediate = torch.cat((vocals_array, self.chords_net_t3(self.chords_net_t4(chords_array))), dim=1)
          intermediate = self.dis_net_t1(self.dis_net_t2(self.dis_net_t3(intermediate)))
        elif tier==2:
          intermediate = torch.cat((vocals_array, self.chords_net_t2(self.chords_net_t3(self.chords_net_t4(chords_array)))), dim=1)
          intermediate = self.dis_net_t1(self.dis_net_t2(intermediate))
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

# chords_noise = torch.randn(BATCH_SIZE, 1, 1632, 8, device=device)
# real = torch.ones(1)*0
# tier = torch.ones(1)*2

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# section_noise = torch.randn(BATCH_SIZE, 5, device=device)
# vocals = netG(noise, chords_noise, section_noise, tier)

# netD(vocals, chords_noise, real, tier)[1].shape

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
##############
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.00001, momentum=0.9)
# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, 0.999))
##############
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.005, momentum=0.9)
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))
#############

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.01, momentum=0.9)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.01, betas=(beta1, 0.999))

# Training Loop

num_epochs=5000
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
valD_losses = []
iters = 0
tier_list = [4]

tier_1 = torch.ones(1, device=device)
tier_2 = torch.ones(1, device=device)*2
tier_3 = torch.ones(1, device=device)*3
tier_4 = torch.ones(1, device=device)*4

real_tier = torch.ones(1, device=device)
fake_tier = torch.ones(1, device=device)*0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    # for d in optimizerD.param_groups:
    #     d['lr'] = lr_sched_dis[epoch]
    # for g in optimizerG.param_groups:
    #     g['lr'] = lr_sched_gen[epoch]

    for i, data in enumerate(trainloader, 0):

        # Format batch
        real_vocals = torch.unsqueeze(data[1],1).to(device)
        real_chords = torch.unsqueeze(data[2],1).to(device)
        real_section = data[3].to(device)
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        fake_labels = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()
        # Forward pass real batch through D
        output = netD(real_vocals, real_chords, real_tier, tier_4)
        D_x = output[0].mean().item()
        # Calculate loss on all-real batch
        # Calculate section label accuracy for real examples
        real_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # scale real error by length of tier list
        errD_real = len(tier_list)*(criterion(torch.squeeze(output[0]), real_labels) + criterion_sec(torch.squeeze(output[1]), real_section))
        # Calculate gradients for D in backward pass
        errD_real.backward()


        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        errD_fake = 0
        D_G_z1 = []
        fake_sec_accu = []
        for tier_ind in tier_list:
            if tier_ind==4:
              # Calculate section label accuracy for fake examples
              fake_t4 = netG(noise, real_chords, real_section, tier_4)
              output = netD(fake_t4.detach(), real_chords.detach(), fake_tier, tier_4)
              D_G_z1.append(output[0].mean().item())
              fake_sec_accu.append((torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item())
              errD_fake += criterion(torch.squeeze(output[0]), fake_labels) + criterion_sec(torch.squeeze(output[1]), real_section)
            elif tier_ind==3:
              fake_t3 = netG(noise, real_chords, real_section, tier_3)
              output = netD(fake_t3.detach(), real_chords.detach(), fake_tier, tier_3)
              D_G_z1.append(output[0].mean().item())
              fake_sec_accu.append((torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item())
              errD_fake += criterion(torch.squeeze(output[0]), fake_labels) + criterion_sec(torch.squeeze(output[1]), real_section)
            elif tier_ind==2:
              fake_t2 = netG(noise, real_chords, real_section, tier_2)
              output = netD(fake_t2.detach(), real_chords.detach(), fake_tier, tier_2)
              D_G_z1.append(output[0].mean().item())
              fake_sec_accu.append((torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item())
              errD_fake += criterion(torch.squeeze(output[0]), fake_labels) + criterion_sec(torch.squeeze(output[1]), real_section)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()

        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        errG = 0
        D_G_z2 = []
        for tier_ind in tier_list:
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # Calculate G's loss based on this output
            if tier_ind==4:
              output = netD(fake_t4, real_chords, fake_tier, tier_4)
              D_G_z2.append(output[0].mean().item())
              errG += criterion(torch.squeeze(output[0]), real_labels) + criterion_sec(torch.squeeze(output[1]), real_section)
            elif tier_ind==3:
              output = netD(fake_t3, real_chords, fake_tier, tier_3)
              D_G_z2.append(output[0].mean().item())
              errG += criterion(torch.squeeze(output[0]), real_labels) + criterion_sec(torch.squeeze(output[1]), real_section)
            elif tier_ind==2:
              output = netD(fake_t2, real_chords, fake_tier, tier_2)
              D_G_z2.append(output[0].mean().item())
              errG += criterion(torch.squeeze(output[0]), real_labels) + criterion_sec(torch.squeeze(output[1]), real_section)

        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 5 == 0:
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f sec_acc:%.2f  D(x):%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                    errD.item(), errG.item(), real_sec_accu, D_x))
            for z,x in enumerate(tier_list):
                print('tier:%d  D(G(z)):%.2f  sec_acc:%.2f' % (x, D_G_z1[z], fake_sec_accu[z]))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = torch.unsqueeze(data[1],1).to(device)
            val_chords = torch.unsqueeze(data[2],1).to(device)
            val_section = data[3].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals, val_chords, torch.ones(1), torch.ones(1)*4)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section, torch.ones(1)*4)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake.detach(), val_chords.detach(), torch.ones(1)*0, torch.ones(1)*4)
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real) + criterion(torch.squeeze(output_fake[0].detach()), labels_fake) + criterion_sec(torch.squeeze(output_real[1].detach()), val_section) + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section))
            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)
            img_list.append(vutils.make_grid(torch.permute(torch.squeeze(fake.detach().cpu()), (0,2,1)), padding=2, normalize=True))

    iters += 1

netG.eval()
for v, data in enumerate(valloader, 0):
  with torch.no_grad():

      # Real Batch
      val_vocals = torch.unsqueeze(data[1],1).to(device)
      val_chords = torch.unsqueeze(data[2],1).to(device)
      val_section = data[3].to(device)

      b_size = val_vocals.size(0)
      print('b_size:', b_size)
      labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

      # Fake Batch
      noise = torch.randn(b_size, nz, device=device)
      fake = netG(noise, val_chords, val_section, torch.ones(1)*4)

q=5
print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['chords_array'].T, interpolation='nearest', aspect='auto', origin='lower')

print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
im = np.squeeze(fake[q].detach().cpu()).T
im[im<0.5]=0
plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')

print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['vocals_array'].T, interpolation='nearest', aspect='auto', origin='lower')

!mkdir /content/temp

for q in range(15):
  filename = data[0][q].split('./data/Val/')[1]
  print(filename)
  np.savez_compressed('/content/temp/temp.npz', vocals_array=torch.squeeze(fake.cpu().detach())[q])
  upload_file(filename, '/content/temp/temp.npz', '1oiLnPuH4FknepGehqHAMHy1Y4iLK2Ap7')
  

data[0][0]

torch.save(netG.state_dict(), 'Models/NetG_state.pt')
torch.save(netD.state_dict(), 'Models/NetD_state.pt')

! ls /content/temp

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
ims = [[plt.imshow(i[q], interpolation='nearest', aspect='auto', origin='lower', animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)

HTML(ani.to_jshtml())

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
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

plot_grad_flow(netG.named_parameters())
