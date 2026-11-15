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

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

torch.autograd.set_detect_anomaly(True)


cmajor_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'D_major', 'E_minor', 'E_major', 'F_minor', 'F_major', 'G_major', 'A_minor', 'A_major', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

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
        elif 'verse1' in data_loc:
          section = [1,0,0,0,0]
        elif 'verse2' in data_loc:
          section = [0,0,0,0,1]
        else:
          raise Exception('no section found')

        vocals_array=data['vocals_array']
        vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
        vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
        vocals_array2[:,:-1] = vocals_array
        return data_loc, from_numpy(vocals_array2).float(), from_numpy(data['chords_array']).float(), torch.FloatTensor(section)
        
        # return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['chords_array']).float(), torch.FloatTensor(section)

BATCH_SIZE = 24
trainset = LoadFromFolder(main_dir="./midi_ML_data/Train2/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./midi_ML_data/Validation2/")
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

        self.chords_net = nn.Sequential(
            # input is (1) x 1632 x 11
            nn.Conv2d(1,4, (4,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size 4 x 816 x 6 

            nn.Conv2d(4,8,(4,4), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (8) x 408 x 3

            nn.Conv2d(8,16,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (16) x 204 x 2

            nn.Conv2d(16,32,(4,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
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
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (512) x 204 x 2
        )

        self.dense_block_t1 = Dense_Block(in_channels=512, num_layers=3, kernal_size=(3,1), padding=(1,0), dropout=dropout_rate)
        # state size. BATCH x (640) x 204 x 2

        self.gen_net_t2 = nn.Sequential(
            nn.ConvTranspose2d(640, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (256) x 408 x 4
        )

        self.dense_block_t2 = Dense_Block(in_channels=256, num_layers=3, kernal_size=(5,3), padding=(2,1), dropout=dropout_rate)
        # state size. BATCH x (384) x 408 x 4

        self.gen_net_t3 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (128) x 816 x 8
        )

        self.dense_block_t3 = Dense_Block(in_channels=128, num_layers=3, kernal_size=(7,3), padding=(3,1), dropout=dropout_rate)
        # state size. BATCH x (256) x 816 x 8
        
        self.gen_net_t4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(2,3), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (64) x 1632 x 17

            nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(1,2), padding=(1,0), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. BATCH x (64) x 1632 x 35

            nn.Conv2d(64,64,(3,5), padding=(1,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (64) x 1632 x 31

            nn.Conv2d(64,64,(3,5), padding=(1,0), stride=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (64) x 1632 x 27

            nn.Conv2d(64,64,(3,4), padding=(1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # state size. (64) x 1632 x 26
        )

        self.dense_block_t4 = Dense_Block(in_channels=64, num_layers=3, kernal_size=(5,3), padding=(2,1), dropout=dropout_rate)
        # state size. BATCH x (192) x 1632 x 26

        self.final = nn.Sequential(
            nn.Conv2d(192,1, (5,3), padding=(2,1), stride=1, bias=True),
            nn.Softmax(dim=3),
            # state size. BATCH x (1) x 1632 x 26
        )

    def forward(self, noise, chords_array, section):
        intermediate = torch.cat((self.chords_net(chords_array), self.latent(noise).view(-1, (1024-32-1), 102, 1), self.section_net(section).view(-1,1,102,1)), dim=1)
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

# data = next(iter(trainloader))
# real_vocals = torch.unsqueeze(data[1],1).to(device)
# real_chords = torch.unsqueeze(data[2],1).to(device)
# real_section = data[3].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# netG(noise, real_chords, real_section).shape

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(26,4,0,2)
# conv_size(6,4,1,2)

dis_dropout_rate = 0.4
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input state BATCH_SIZE x (1) x 1632 x 11
            spectral_norm(nn.Linear(11, 26)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dis_dropout_rate)
            # state BATCH_SIZE x (1) x 1632 x 26
        )

        # concatted input is (2) x 1632 x 26
        self.dense_block_t4 = Dense_Block(in_channels=2, num_layers=3, kernal_size=(5,3), padding=(2,1), dropout=dis_dropout_rate)
        # state size. BATCH x (130) x 1632 x 26

        self.dis_net_t4 = nn.Sequential(
            spectral_norm(nn.Conv2d(130,128, (4,4), padding=(1,0), stride=2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size. (128) x 816 x 12

            spectral_norm(nn.Conv2d(128,128, (3,3), padding=(1,0), stride=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size. (128) x 816 x 10

            spectral_norm(nn.Conv2d(128,128, (3,3), padding=(1,0), stride=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size. (128) x 816 x 8
        )

        self.dense_block_t3 = Dense_Block(in_channels=128, num_layers=3, kernal_size=(7,3), padding=(3,1), dropout=dis_dropout_rate)
        # state size. BATCH x (256) x 816 x 8

        self.dis_net_t3 = nn.Sequential(
            spectral_norm(nn.Conv2d(256,256,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size. (256) x 408 x 4
        )

        self.dense_block_t2 = Dense_Block(in_channels=256, num_layers=3, kernal_size=(5,3), padding=(2,1), dropout=dis_dropout_rate)
        # state size. BATCH x (384) x 408 x 4

        self.dis_net_t2 = nn.Sequential(
            spectral_norm(nn.Conv2d(384,512,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
            # state size. (512) x 204 x 2
        )

        self.dense_block_t1 = Dense_Block(in_channels=512, num_layers=3, kernal_size=(3,1), padding=(1,0), dropout=dis_dropout_rate)
        # state size. BATCH x (640) x 204 x 2

        self.dis_net_t1 = nn.Sequential(
            spectral_norm(nn.Conv2d(640,1024,(4,2), padding=(1,0), stride=2)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dis_dropout_rate),
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

    def forward(self, vocals_array, chords_array):
        intermediate = torch.cat((vocals_array, self.chords_net(chords_array)), dim=1)
        intermediate = self.dis_net_t4(self.dense_block_t4(intermediate))
        intermediate = self.dis_net_t3(self.dense_block_t3(intermediate))
        intermediate = self.dis_net_t2(self.dense_block_t2(intermediate))
        intermediate = self.dis_net_t1(self.dense_block_t1(intermediate))
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
# real_chords = torch.unsqueeze(data[2],1).to(device)
# real_section = data[3].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# vocals = netG(noise, real_chords, real_section)

# netD(vocals, real_chords)[1].shape

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
# optimizerD = optim.Adam(netD.parameters(), lr=0.001, betas=(0.5, 0.99))
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.005, momentum=0.9)

# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(0.2, 0.95))
##############


### SGD
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(beta1, 0.999))

def one_value_loss(vocals, b_size):
    values, indices = torch.max(torch.squeeze(vocals), dim=2)
    return torch.mean(torch.abs(values - torch.ones(b_size, 1632).to(device)))

checkpoint = torch.load('Models/softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/model_0001_005_batch24_000700.pt')

netG.load_state_dict(checkpoint['netG_state_dict'])

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

q=0
print(data[0][q])
plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['chords_array'].T, interpolation='nearest', aspect='auto', origin='lower')

print(data[0][q])
im = np.squeeze(fake[q].detach().cpu())

import matplotlib
dpi = matplotlib.rcParams['figure.dpi']
height, width = im.T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(im.T, interpolation='nearest', aspect='auto', origin='lower')
  # plt.xlim(0,900)
  # plt.ylim(0,30)
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);
  # plt.imshow(vocals_array.T, interpolation='nearest', origin='lower')

for q in range(b_size):
  outname = 'outputs/MIDI/softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/' + data[0][q].split('.npz')[0].split('/')[-1]
  np.savez(outname, vocals=np.squeeze(fake[q].detach().cpu()), numpy_datafile=data[0][q])

data[2][q].shape

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
