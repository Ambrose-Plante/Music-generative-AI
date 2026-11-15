from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/integrated

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def upload_file(filename, file_loc, dir_id):
  uploaded = drive.CreateFile({'title': filename, 'parents': [{'id': dir_id}]})
  uploaded.SetContentFile(file_loc)
  uploaded.Upload()
  # print('Uploaded file with ID {}'.format(uploaded.get('id')))

import os
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from torchvision.transforms import ToTensor
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np

import argparse
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

        output_array = data['output_array']

        # for i in range(output_array.shape[0]): ## make silence a one at the highest pitch, highest syllable
        #   if output_array[i].flatten().sum()==0:
        #     output_array[i,-1,-1]=1

        return data_loc, torch.transpose(from_numpy(output_array).float(),0,2), from_numpy(data['input_syll_vec']).float(), from_numpy(data['vox_input']).float(), from_numpy(data['acc_input']).float(), from_numpy(data['chords_array']).float(), torch.squeeze(from_numpy(data['key_array']).float().view(-1,24)), from_numpy(data['bpm_array']).float()

BATCH_SIZE = 6
trainset = LoadFromFolder(main_dir="./data/train/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./data/val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #use 0 workers for gpu, 1 for cpu

testset = LoadFromFolder(main_dir="./data/test/")
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #use 0 workers for gpu, 1 for cpu

names = ['midi_array', 'input_syll_vec', 'vox_input', 'acc_input', 'chords_array', 'key_array', 'bpm_array']

for i, x in enumerate(next(iter(trainloader))):
  if i>0:
    print(names[i-1], i, x.shape)  # prints shape of image with single batch
  else:
    print(x)

names, midi_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array = next(iter(trainloader))
plt.plot(vox_input[0,:,2001])

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

def output_size(input_size, kernal_size, padding, stride):
    return ((input_size-kernal_size+(2*padding))/stride)+1

# output_size(input_size=3265, kernal_size=7, padding=2, stride=2)
# output_size(input_size=1632, kernal_size=6, padding=2, stride=2)
output_size(input_size=15, kernal_size=3, padding=1, stride=2)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.midi_net = nn.Sequential(
            # input is (BATCH), 200, 61, 1632
            nn.Conv2d(200,180, (3,4), padding=(1,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(180),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 180 x 31 x 1631

            nn.Conv2d(180,160, (3,4), padding=(1,2), stride=(2,1), bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 160 x 16 x 1632

            nn.Conv2d(160,150, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(150),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 150 x 8 x 1632

            nn.Conv2d(150,140, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(140),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 140 x 4 x 1632

            nn.Conv2d(140,130, (2,3), padding=0, stride=(2,1), bias=False),
            nn.BatchNorm2d(130),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 130 x 2 x 1630

            nn.Conv2d(130,120, (2,3), padding=0, stride=(2,1), bias=False),
            nn.BatchNorm2d(120),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 120 x 1 x 1628
        )

        self.key_net = nn.Sequential(
            # input is (BATCH) x 150
            nn.Linear(24,120)
        )

        self.latent = nn.Sequential(
            nn.Linear(nz, 120*3),
        )

        self.gen_net_t1 = nn.Sequential(
            # input state BATCH x 1 x 120 x 1632
            nn.ConvTranspose2d(1, 2, (3,5), (1,2), 1, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (2) x 120 x 3265
        )

        self.dense_block_t1 = Dense_Block(in_channels=2, num_layers=3, kernal_size=(3,7), padding=(1,3), dropout=0.2)
        # state size. BATCH x (130) x 120 x 3265

        self.final = nn.Sequential(
            # input state BATCH x (130) x 120 x 3265
            nn.Conv2d(130,50, (3,7), padding=(1,3), stride=1, bias=False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (50) x 120 x 3265

            nn.Conv2d(50,10, (3,7), padding=(1,3), stride=1, bias=False),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (10) x 120 x 3265

            nn.Conv2d(10,1, (3,7), padding=(1,3), stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # state size. BATCH x (1) x 120 x 3265
        )


    def forward(self, noise, midi_array, key_array):
        intermediate = torch.cat((self.midi_net(midi_array).transpose(1,2), self.latent(noise).view(-1, 1, 120, 3), self.key_net(key_array).view(-1,1,120,1)), dim=-1)
        intermediate = self.dense_block_t1(self.gen_net_t1(intermediate))
        return self.final(intermediate).view(-1,120,3265)
        # return self.gen_net_t1(intermediate)
        # return self.bpm_net(bpm_array)

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

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# netG(noise, midi_array.to(device), bpm_array.to(device)).shape

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.midi_net = nn.Sequential(
            # input is (BATCH), 200, 61, 1632
            nn.Conv2d(200,180, (3,4), padding=(1,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(180),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 180 x 31 x 1631

            nn.Conv2d(180,160, (3,4), padding=(1,2), stride=(2,1), bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 160 x 16 x 1632

            nn.Conv2d(160,150, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(150),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 150 x 8 x 1632

            nn.Conv2d(150,140, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(140),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 140 x 4 x 1632

            nn.Conv2d(140,130, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(130),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 130 x 2 x 1632

            nn.Conv2d(130,120, (2,3), padding=(0,1), stride=(2,1), bias=False),
            nn.BatchNorm2d(120),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 120 x 1 x 1632
        )

        self.vox_preprocess = nn.Sequential(
            # input is (BATCH), 1, 120, 3265
            nn.Conv2d(1,1, (3,7), padding=(1,2), stride=(1,2), bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            # state size BATCH x 1 x 120 x 1632
        )

        # concatted input is (BATCH) x 2 x 120 x 1632
        self.dense_block_t4 = Dense_Block(in_channels=2, num_layers=3, kernal_size=(3,5), padding=(1,2), dropout=0.3)
        # state size. BATCH x (130) x 120 x 1632

        self.dis_net_t4 = nn.Sequential(
            spectral_norm(nn.Conv2d(130,50, (4,6), padding=(1,2), stride=2)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # state size. (50) x 60 x 816
        )

        self.dense_block_t3 = Dense_Block(in_channels=50, num_layers=3, kernal_size=(3,5), padding=(1,2), dropout=0.3)
        # state size. BATCH x (178) x 60 x 816

        self.dis_net_t3 = nn.Sequential(
            spectral_norm(nn.Conv2d(178,50,(4,4), padding=(1,1), stride=2)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # state size. BATCH x 50 x 30 x 408
        )

        self.dense_block_t2 = Dense_Block(in_channels=50, num_layers=3, kernal_size=(3,5), padding=(1,2), dropout=0.3)
        # state size. BATCH x (178) x 4 x 408

        self.dis_net_t2 = nn.Sequential(
            spectral_norm(nn.Conv2d(178,50,(4,4), padding=(1,1), stride=2)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # state size. (50) x 15 x 204
        )

        self.dense_block_t1 = Dense_Block(in_channels=50, num_layers=3, kernal_size=(3,5), padding=(1,2), dropout=0.3)
        # state size. BATCH x (178) x 15 x 204

        self.dis_net_t1 = nn.Sequential(
            spectral_norm(nn.Conv2d(178,50,(3,4), padding=(1,1), stride=2)),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # state size. (50) x 8 x 102

            nn.Flatten(),
        )

        self.real_fake = nn.Sequential(
            spectral_norm(nn.Linear(50*8*102, 1)),
            nn.Sigmoid()
        )

        self.key_net = nn.Sequential(
            spectral_norm(nn.Linear(50*8*102, 24))
        )

    def forward(self, midi_array, vox_input):
        intermediate = torch.cat((self.midi_net(midi_array).transpose(1,2), self.vox_preprocess(torch.unsqueeze(vox_input, 1))), axis=1)

        intermediate = self.dis_net_t4(self.dense_block_t4(intermediate))
        intermediate = self.dis_net_t3(self.dense_block_t3(intermediate))
        intermediate = self.dis_net_t2(self.dense_block_t2(intermediate))
        intermediate = self.dis_net_t1(self.dense_block_t1(intermediate))
        return self.real_fake(intermediate), self.key_net(intermediate)

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

# [x.shape for x in netD(midi_array.to(device), bpm_array.to(device), vox_input.to(device))]

# lr_sched_gen = np.concatenate((np.arange(0,0.01, 0.0004), 0.01*np.ones(300), 0.001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_gen)

# lr_sched_dis = np.concatenate((np.arange(0,0.001, 0.00004), 0.001*np.ones(300), 0.0001*np.ones(400), 0.000001*np.ones(5000)))
# plt.plot(lr_sched_dis)

# Initialize BCELoss function
criterion = nn.BCELoss()
criterion_bpm = nn.CrossEntropyLoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

##good combos####
##############
# optimizerD = optim.Adam(netD.parameters(), lr=0.0005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))
##############


### SGD
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.01, momentum=0.9)

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
        midi_array_real = data[1].to(device)
        vox_input_real = data[3].to(device)
        key_array_real = data[6].to(device)
        b_size = midi_array_real.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(midi_array_real, vox_input_real)
        # Calculate bpm label accuracy for real examples
        real_bpm_accu = (torch.sum(key_array_real.argmax(axis=1)==output[1].argmax(axis=1))/key_array_real.shape[0]).item()
        # Calculate loss on all-real batch
        errD_real = criterion(torch.squeeze(output[0]), label) + criterion_bpm(torch.squeeze(output[1]), key_array_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output[0].mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise, midi_array_real, key_array_real)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(midi_array_real.detach(), fake.detach())
        # Calculate bpm label accuracy for fake examples
        fake_bpm_accu = (torch.sum(key_array_real.argmax(axis=1)==output[1].argmax(axis=1))/key_array_real.shape[0]).item()
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(torch.squeeze(output[0]), label) + criterion_bpm(torch.squeeze(output[1]), key_array_real)
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
        output = netD(midi_array_real, fake)
        # Calculate G's loss based on this output
        errG = criterion(torch.squeeze(output[0]), label) + criterion_bpm(torch.squeeze(output[1]), key_array_real)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output[0].mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 2 == 0:
            print('[%d/%d][%d/%d]  Loss_D:%.2f Loss_G:%.2f  D(x):%.2f  D(G(z)):%.2f  bpm_acc:%.2f/%.2f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, real_bpm_accu, fake_bpm_accu))

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
            val_midi = data[1].to(device)
            val_vox = data[3].to(device)
            val_bpm = data[6].to(device)
            b_size = val_vox.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_midi, val_vox)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_midi, val_bpm)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(val_midi.detach(), fake.detach())
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real) + criterion(torch.squeeze(output_fake[0].detach()), labels_fake) + criterion_bpm(torch.squeeze(output_real[1].detach()), val_bpm) + criterion_bpm(torch.squeeze(output_fake[1].detach()), val_bpm))
            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_bpm:', (output_real[1].argmax(axis=1)==torch.squeeze(val_bpm).argmax(axis=1)).sum().item()/b_size)

        iters += 1

netG.eval()
for v, data in enumerate(trainloader, 0):
    with torch.no_grad():
        if len([x for x in data[0] if '0_0_chorus' in x])==0:
          continue
        # Real Batch
        val_midi = data[1].to(device)
        val_vox = data[3].to(device)
        val_bpm = data[7].to(device)
        b_size = val_vox.size(0)
        print('b_size:', b_size)

        # Fake Batch
        noise = torch.randn(b_size, nz, device=device)
        fake = netG(noise, val_midi, val_bpm)
        break

data[0]

import pickle
val_data = {}
val_data['names'] = data[0]
val_data['vox'] = fake.cpu().numpy()
val_data['bpm'] = val_bpm.cpu().numpy()
val_data['midi'] = val_midi.cpu().numpy()
with open('val_data.pickle', 'wb') as fp:
    pickle.dump(val_data, fp)

q=6
print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['chords_array'].T, interpolation='nearest', aspect='auto', origin='lower')

print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
im = np.squeeze(fake[q].detach().cpu()).T
# im[im<0.5]=0
plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')

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
  

torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'valD_losses': valD_losses,
            }, '../music_app/Models/AutoVox/model_' + str(epoch).zfill(6) + '.pt')

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
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
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
