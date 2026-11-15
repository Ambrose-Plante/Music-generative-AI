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
from torch import from_numpy, Tensor
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
            nn.Linear(nz, 102*1*1),
        )

        self.gen_net_t1 = nn.Sequential(
            # input state (34) x 102 x 1
            nn.ConvTranspose2d(34, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 204 x 2
        )

        self.positional_encoding_t1 = PositionalEncoding_simple(max_len=204)
        self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=3, nhead=3, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 204 x 3

        self.gen_net_t2 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 408 x 6
        )

        self.positional_encoding_t2 = PositionalEncoding_simple(max_len=408)
        self.attention_t2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=7, nhead=7, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 408 x 7

        self.gen_net_t3 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 816 x 14
        )
        
        self.positional_encoding_t3 = PositionalEncoding_simple(max_len=816)
        self.attention_t3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=15, nhead=5, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 816 x 15

        self.gen_net_t4 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2,5), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            # state size. BATCH x (1) x 1632 x 19
        )

        self.final = nn.Sequential(
            nn.Conv2d(1,1, (9,3), padding=(4,1), stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 1632 x 19
        )

    def forward(self, noise, chords_array, section):
        intermediate = torch.cat((self.chords_net(chords_array), self.latent(noise).view(-1, 1, 102, 1), self.section_net(section).view(-1,1,102,1)), dim=1)
        intermediate = self.gen_net_t1(intermediate)

        intermediate = self.positional_encoding_t1(torch.squeeze(intermediate))
        intermediate = self.attention_t1(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)

        intermediate = self.gen_net_t2(intermediate)

        intermediate = self.positional_encoding_t2(torch.squeeze(intermediate))
        intermediate = self.attention_t2(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)
      
        intermediate = self.gen_net_t3(intermediate)

        intermediate = self.positional_encoding_t3(torch.squeeze(intermediate))
        intermediate = self.attention_t3(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)


        intermediate = self.gen_net_t4(intermediate)
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
# netG(noise, chords_noise, section_noise).shape

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input state BATCH_SIZE x (1) x 1632 x 8
            spectral_norm(nn.Linear(8, 19)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.1)
            # state BATCH_SIZE x (1) x 1632 x 19
        )

        self.dis_net_t4 = nn.Sequential(
            spectral_norm(nn.Conv2d(2,1, (4,5), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (1) x 816 x 8
        )

        self.positional_encoding_t3 = PositionalEncoding_simple(max_len=816)
        self.attention_t3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=9, nhead=9, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 816 x 9

        self.dis_net_t3 = nn.Sequential(
            spectral_norm(nn.Conv2d(1,1,(4,2), padding=(1,1), stride=2)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (1) x 408 x 5
        )

        self.positional_encoding_t2 = PositionalEncoding_simple(max_len=408)
        self.attention_t2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=6, nhead=6, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 408 x 6

        self.dis_net_t2 = nn.Sequential(
            spectral_norm(nn.Conv2d(1,1,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (1) x 204 x 3
        )

        self.positional_encoding_t1 = PositionalEncoding_simple(max_len=204)
        self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4, nhead=4, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 204 x 4

        self.dis_net_t1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1,1,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (1) x 102 x 2
        )

        self.positional_encoding_t0 = PositionalEncoding_simple(max_len=102)
        self.attention_t0 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=3, nhead=3, dim_feedforward=1024, batch_first=True), num_layers=3)
        # state size. BATCH x (1) x 102 x 3

        self.real_fake = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(3*102, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(3*102, 5))
        )

    def forward(self, vocals_array, chords_array):
        intermediate = torch.cat((vocals_array, self.chords_net(chords_array)), dim=1)
        intermediate = self.dis_net_t4(intermediate)

        intermediate = self.positional_encoding_t3(torch.squeeze(intermediate))

        intermediate = self.attention_t3(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)
        intermediate = self.dis_net_t3(intermediate)

        intermediate = self.positional_encoding_t2(torch.squeeze(intermediate))
        intermediate = self.attention_t2(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)

        intermediate = self.dis_net_t2(intermediate)

        intermediate = self.positional_encoding_t1(torch.squeeze(intermediate))
        intermediate = self.attention_t1(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)

        intermediate = self.dis_net_t1(intermediate)

        intermediate = self.positional_encoding_t0(torch.squeeze(intermediate))
        intermediate = self.attention_t0(intermediate)
        intermediate = torch.unsqueeze(intermediate, dim=1)

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
# noise = torch.randn(BATCH_SIZE, nz, device=device)
# section_noise = torch.randn(BATCH_SIZE, 5, device=device)
# vocals = netG(noise, chords_noise, section_noise)

# netD(vocals, chords_noise)[1].shape

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

trans_paramsG = list(map(lambda x: x[1],list(filter(lambda kv: 'attention' in kv[0], netG.named_parameters()))))
base_paramsG = list(map(lambda x: x[1],list(filter(lambda kv: 'attention' not in kv[0], netG.named_parameters()))))

trans_paramsD = list(map(lambda x: x[1],list(filter(lambda kv: 'attention' in kv[0], netD.named_parameters()))))
base_paramsD = list(map(lambda x: x[1],list(filter(lambda kv: 'attention' not in kv[0], netD.named_parameters()))))

##good combos####
##############
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.00001, momentum=0.9)
# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, 0.999))
##############
# optimizerD = optim.SGD([{'params': base_paramsD}, {'params': trans_paramsD, 'lr': 0.1}], lr=0.05, momentum=0.9)
# optimizerG = optim.Adam([{'params': base_paramsG}, {'params': trans_paramsG, 'lr': 0.1}], lr=0.05, betas=(beta1, 0.999))
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0005, momentum=0.9)
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))
#############

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.01, momentum=0.9)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.01, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.01, betas=(beta1, 0.999))

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
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_vocals, real_chords)
        # Calculate section label accuracy for real examples
        real_sec_accu = (torch.sum(real_section.argmax(axis=1)==output[1].argmax(axis=1))/real_section.shape[0]).item()
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section)
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
        output = netD(fake.detach(), real_chords.detach())
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
        output = netD(fake, real_chords)
        # Calculate G's loss based on this output
        errG = criterion(torch.squeeze(output[0]), label) + criterion_sec(torch.squeeze(output[1]), real_section)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output[0].mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 2 == 0:
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  sec_acc:%.2f/%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_sec_accu, fake_sec_accu))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # LR update
        # if (epoch > 1) and (epoch % 10 == 0) and (i == len(trainloader)-1):
        #   if D_G_z1<0.1 and D_x>0.9: # D is good, but overwhelming G
        #     optimizerG.param_groups[0]['lr'] *= 1.5
        #     optimizerD.param_groups[0]['lr'] /= 10
        #     print('new opt LRs: ', optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr'])

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = torch.unsqueeze(data[1],1).to(device)
            val_chords = torch.unsqueeze(data[2],1).to(device)
            val_section = data[3].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals, val_chords)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake.detach(), val_chords.detach())
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
      fake = netG(noise, val_chords, val_section)

q=5
print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['chords_array'].T, interpolation='nearest', aspect='auto', origin='lower')

print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
im = np.squeeze(fake[q].detach().cpu()).T
# im[im<0.1]=0
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
    
    Usage: Plug this function in Trainer class
     after loss.backwards() as 
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

plot_grad_flow(netD.named_parameters())
