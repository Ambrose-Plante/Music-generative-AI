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
import math

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

# ! mv NLP_data/Train/Avicii_Wake\ Me\ Up_chorus_0.npz NLP_data/Val/ ## 1631
# ! mv NLP_data/Train/Ed\ Sheeran_Shape\ of\ You_chorus_*.npz NLP_data/Val/ ## 863
# ! mv NLP_data/Train/Jonas\ Brothers_What\ a\ man\ gotta\ do_verse*.npz NLP_data/Val/ ## 1247
# ! mv NLP_data/Train/Justin\ Bieber_Peaches_chorus*.npz NLP_data/Val/ ## 479

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

        return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['words_vec']).float(), from_numpy(data['syll_vec']).float(), torch.FloatTensor(section)

BATCH_SIZE = 16
trainset = LoadFromFolder(main_dir="./NLP_data/Train/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./NLP_data/Val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# names = ['vocals_array', 'chords_array']
# data = next(iter(trainloader))
# print(data[0][0])
# print(data[1][0].shape)
# print(data[2][0].shape)
# print(data[3][0].shape)

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of training epochs
num_epochs = 1000

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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
        x = torch.cat((x, self.pe.repeat(x.shape[0],1).unsqueeze(2)), dim=2)
        return x

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding_BF(d_model, dropout)
        self.pos_encoder = PositionalEncoding_simple(max_len=200)
        encoder_layers = nn.TransformerEncoderLayer(d_model=(d_model+1), nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output

src_mask = torch.triu(torch.ones(200,200) * float('-inf'), diagonal=1).to(device)

class VocalsNet(nn.Module):

    def __init__(self, ngpu):
        super(VocalsNet, self).__init__()
        self.ngpu = ngpu

        self.vocals_net = nn.Sequential(
            # input is 200 x 1632 x 19
            spectral_norm(nn.Conv2d(200,400,(10,4), padding=(1,1), stride=2, bias=False)),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            spectral_norm(nn.Conv2d(400,600,(10,4), padding=(1,1), stride=2, bias=False)),
            nn.BatchNorm2d(600),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            spectral_norm(nn.Conv2d(600,400,(5,2), padding=(1,1), stride=2, bias=False)),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            spectral_norm(nn.Conv2d(400,300,(5,3), padding=(1,1), stride=2, bias=False)),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            spectral_norm(nn.Conv2d(300,200,(5,2), padding=(2,0), stride=2, bias=False)),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Flatten(start_dim=2)
            # output is BATCH x 200 x 50
        )

    def forward(self, vocals_array: Tensor):

        return self.vocals_net(vocals_array)

Vocals_Net = VocalsNet(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    Vocals_Net = nn.DataParallel(Vocals_Net, list(range(ngpu)))

# Print the model
print(Vocals_Net)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.section_net = nn.Sequential(
            spectral_norm(nn.Linear(5,200*7)),
            nn.Dropout(p=0.2)
        )

        self.latent = nn.Sequential(
            spectral_norm(nn.Linear(nz, 200*6)),
            nn.Dropout(p=0.2)
        )

        self.gen_net = nn.Sequential(
            # input state BATCH x 200 x 63
            # TransformerModel(d_model=63, nhead=2, d_hid=100, nlayers=1, dropout=0.0)
            spectral_norm(nn.Linear(63, 64)),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
            # output state BATCH x 200 x 64
        )

        # self.gen_words = nn.Sequential(
        #     # input state BATCH x 200 x 64
        #     spectral_norm(nn.Linear(64, 300)),
        #     nn.Sigmoid()
        #     # output state BATCH x 200 x 300
        # )

        self.gen_sylls = nn.Sequential(
            # input state BATCH x 200 x 64
            nn.Unflatten(2,(2,32)),
            # state size BATCH x 200 x 2 x 32
            spectral_norm(nn.ConvTranspose2d(200, 200, kernel_size=(1,4), stride=(2,1), padding=0, bias=False)),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size BATCH x 200 x 3 x 35
            spectral_norm(nn.ConvTranspose2d(200, 200, kernel_size=(2,6), stride=(2,1), padding=(0,0), bias=False)),
            nn.BatchNorm2d(200),
            nn.Sigmoid()
            # output state BATCH x 200 x 6 x 40
        )

    def forward(self, noise, vocals_array, section):
        intermediate = torch.cat((Vocals_Net(vocals_array), self.latent(noise).view(-1, 200, 6), self.section_net(section).view(-1,200,7)), dim=2)
        temp = torch.unsqueeze(nn.Flatten(start_dim=2)(vocals_array).sum(axis=2),2)
        temp[temp!=0]=1
        intermediate = self.gen_net(torch.mul(intermediate, temp))
        # return self.gen_words(intermediate), self.gen_sylls(intermediate).view(-1,200,6,40)
        return self.gen_sylls(intermediate)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Print the model
print(netG)

# for v, data in enumerate(valloader, 0):
#   pass
# q=0

# vocals_array = from_numpy(np.load(data[0][q])['vocals_array']).float()
# vocals_array = vocals_array.repeat(16,1,1,1).to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# section_noise = torch.randn(BATCH_SIZE, 5, device=device)
# out= netG(noise, vocals_array, section_noise)
# print(out.shape)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.dis_sylls = nn.Sequential(
            # input state BATCH x 200 x 6 x 40
            nn.Flatten(start_dim=2),
            spectral_norm(nn.Linear(6*40, 99)),
            nn.Dropout(p=0.2),
            # output state BATCH x 200 x 100
        )

        self.dis_net = nn.Sequential(
            # concatted input is BATCH x 200 x 149
            TransformerModel(d_model=149, nhead=5, d_hid=200, nlayers=1, dropout=0.2)
            # output is BATCH x 200 x 250
        )

        self.real_fake = nn.Sequential(
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(200*150, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(200*150, 5)),
        )

    def forward(self, vocals_array, syll_array):
        # return self.vocals_net(vocals_array)
        # return self.dis_words(words_array)
        # return self.dis_sylls(syll_array)
        intermediate = self.dis_net(torch.cat((Vocals_Net(vocals_array), self.dis_sylls(syll_array)), dim=2))
        return self.real_fake(intermediate), self.section_net(intermediate)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Print the model
print(netD)

# vocals_noise = torch.randn(BATCH_SIZE, 200, 1632, 19, device=device)
# syll_noise = torch.randn(BATCH_SIZE, 200,6,40, device=device)
# out = netD(vocals_noise, syll_noise)
# print(out[0].shape, out[1].shape)

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
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(beta1, 0.999))
#############

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.01, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0004, betas=(beta1, 0.999))

# netG.load_state_dict(torch.load('Models/NetG_state.pt'))
# netD.load_state_dict(torch.load('Models/NetD_state.pt'))

# Training Loop

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
        real_vocals = data[1].to(device)
        real_words = data[2].to(device)
        real_sylls = data[3].to(device)
        real_section = data[4].to(device)
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_vocals, real_sylls)
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
        fake_sylls = netG(noise, real_vocals, real_section)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(real_vocals.detach(), fake_sylls.detach())
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
        output = netD(real_vocals, fake_sylls)
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
            val_vocals = data[1].to(device)
            val_words = data[2].to(device)
            val_sylls = data[3].to(device)
            val_section = data[4].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals, val_sylls)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake_sylls = netG(noise, val_vocals, val_section)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(val_vocals.detach(), fake_sylls)
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real) + criterion(torch.squeeze(output_fake[0].detach()), labels_fake) + criterion_sec(torch.squeeze(output_real[1].detach()), val_section) + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section))
            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)
            # img_list.append(vutils.make_grid(torch.permute(torch.squeeze(fake.detach().cpu()), (0,2,1)), padding=2, normalize=True))

        iters += 1

for v, data in enumerate(valloader, 0):
  pass

q=1

vocals_array = np.load(data[0][q])['vocals_array']


# print(data[0][q])
# plt.figure(figsize=(8, 6), dpi=100)
# plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
# plt.imshow(.T, interpolation='nearest', aspect='auto', origin='lower')

plt.plot(vocals_array.reshape((200, -1)).sum(axis=1))

noise = torch.randn(1, nz, device=device)
fake_sylls = netG(noise, torch.unsqueeze(data[1][q].to(device),0), torch.unsqueeze(data[4][q].to(device),0))

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
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
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
