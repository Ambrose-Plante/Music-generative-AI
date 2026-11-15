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

        return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['chords_array']).float()

BATCH_SIZE = 8
trainset = LoadFromFolder(main_dir="./data/Train/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./data/Val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

names = ['vocals_array', 'chords_array']

for i, x in enumerate(next(iter(trainloader))):
  if i>0:
    print(names[i-1], i, x.shape)  # prints shape of image with single batch
  else:
    print(x)

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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.chords_net = nn.Sequential(
            # input is (1) x 1632 x 8
            nn.Conv2d(1,64, (4,3), padding=(1,1), stride=2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64,128,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (128) x 408 x 6
            nn.Conv2d(128,64,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.2),
            # state size. (64) x 204 x 4
            nn.Conv2d(64,1,(4,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.1),
            # state size. (1) x 102 x 1
        )

        self.latent = nn.Sequential(
            nn.Linear(nz, 102),
            nn.LeakyReLU(0.2),
        )

        self.gen_net = nn.Sequential(
            # input state (1) x 102 x 2
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (102,2)),
            nn.TransformerEncoderLayer(d_model=2, nhead=2, batch_first=True),
            nn.Unflatten(1, (1,102)),
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 204 x 4
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (204,4)),
            nn.TransformerEncoderLayer(d_model=4, nhead=4, batch_first=True),
            nn.Unflatten(1, (1,204)),
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 408 x 8
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (408,8)),
            nn.TransformerEncoderLayer(d_model=8, nhead=8, batch_first=True),
            nn.Unflatten(1, (1,408)),
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            # state size. BATCH x (1) x 816 x 16
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (816,16)),
            nn.TransformerEncoderLayer(d_model=16, nhead=16, batch_first=True),
            nn.Unflatten(1, (1,816)),
            nn.ConvTranspose2d(1, 1, kernel_size=(2,4), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            # # state size. BATCH x (1) x 1634 x 19
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (1632,19)),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=19, nhead=19, batch_first=True), num_layers=1),
            nn.Unflatten(1, (1,1632)),
        )

    def forward(self, noise, chords_array):
        # return self.chords_net(chords_array)
        # return self.gen_net(noise)
        # return self.latent(noise).view(-1, (1024-32), 102, 1)
        return self.gen_net(torch.cat((self.chords_net(chords_array), self.latent(noise).view(-1, 1, 102, 1)), dim=3))

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
# netG(noise, chords_noise).shape

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.dis_net = nn.Sequential(
            # concatted input is BATCH_SIZE x 1632 x (19+8)
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=(19+8), nhead=9, batch_first=True), num_layers=1),
            nn.Flatten(start_dim=1),
            spectral_norm(nn.Linear(1632*(19+8), 1)),
            nn.Sigmoid()
        )

    def forward(self, vocals_array, chords_array):
        # return chords_array
        return self.dis_net(torch.squeeze(torch.cat((vocals_array, chords_array), dim=-1)))

# in_dim=[102,3]
# padding=[1,1]
# kernal=[10,3]
# stride=2
# def out_size(in_dim, padding, kernal, stride):
#   x = (in_dim[0]+2*padding[0]-(kernal[0]-1)-1)/stride + 1
#   y = (in_dim[1]+2*padding[1]-(kernal[1]-1)-1)/stride + 1
#   return x,y

# out_size(in_dim, padding, kernal, stride)

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

# vocal_noise = torch.randn(BATCH_SIZE, 1, 1632, 19, device=device)
# chord_noise = torch.randn(BATCH_SIZE, 1, 1632, 8, device=device)
# netD(vocal_noise, chord_noise)

# Initialize BCELoss function
criterion = nn.BCELoss()

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
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.0001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.00001, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.00001, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
valD_losses = []
iters = 0
counter=0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    if epoch>1 and counter>3:
      if mean_fake_label<0.1 and mean_real_label>0.9:
        optimizerG.param_groups[0]['lr'] = optimizerG.param_groups[0]['lr']*2
        print('new generator lr=', optimizerG.param_groups[0]['lr'])
        counter=0
      elif mean_fake_label>0.9 and mean_real_label<0.1:
        optimizerG.param_groups[0]['lr'] = optimizerG.param_groups[0]['lr']/2
        print('new generator lr=', optimizerG.param_groups[0]['lr'])
        counter=0



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
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = torch.squeeze(netD(real_vocals, real_chords)).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise, real_chords)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = torch.squeeze(netD(fake.detach(), real_chords.detach())).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
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
        output = torch.squeeze(netD(fake, real_chords)).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 2 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

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
            real_vocals = torch.unsqueeze(data[1],1).to(device)
            real_chords = torch.unsqueeze(data[2],1).to(device)
            b_size = real_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = torch.squeeze(netD(real_vocals, real_chords).detach()).view(-1)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, real_chords)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = torch.squeeze(netD(fake.detach(), real_chords.detach())).view(-1)
            valD_losses.append(criterion(output_real, labels_real) + criterion(output_fake, labels_fake))
            mean_fake_label = output_fake.mean().item()
            mean_real_label = output_real.mean().item()
            print('mean_fake_label:', mean_fake_label, 'mean_real_label:', mean_real_label)
            img_list.append(vutils.make_grid(torch.permute(torch.squeeze(fake.detach().cpu()), (0,2,1)), padding=2, normalize=True))

    iters += 1
    counter += 1

q=0

plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(torch.squeeze(fake.cpu().detach())[q].T, interpolation='nearest', aspect='auto', origin='lower')

for v, data in enumerate(valloader, 0):
  pass

plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(np.load(data[0][q])['vocals_array'].T, interpolation='nearest', aspect='auto', origin='lower')

for data in valloader:
  pass
data[0]

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
plt.legend()
plt.show()

#%%capture
fig = plt.figure(figsize=(6,6))
plt.axis("off")
ims = [[plt.imshow(i[-1], interpolation='nearest', aspect='auto', origin='lower', animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)

HTML(ani.to_jshtml())

print(img_list[0][0][10, :500])
