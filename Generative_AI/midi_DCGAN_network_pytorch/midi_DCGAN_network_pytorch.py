from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import os
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from torchvision.transforms import ToTensor
import numpy as np

from __future__ import print_function
#%matplotlib inline
import argparse
import os
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

        return data_loc, from_numpy(data['vocals_array']).float()

# !pip install wandb
# import wandb
# wandb.init()

BATCH_SIZE = 16
trainset = LoadFromFolder(main_dir="./data/")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) #use 0 workers for gpu, 1 for cpu

names = ['vocals_array']

for i, x in enumerate(next(iter(trainloader))):
  if i>0:
    print(names[i-1], i, x.shape)  # prints shape of image with single batch
  else:
    print(x)

# Root directory for dataset
dataroot = "data/"

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of training epochs
num_epochs = 1000

# Learning rate for optimizers
lr = 0.0005

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
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 1024, kernel_size=(102,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. BATCH x (1024) x 86 x 1
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. BATCH x (512) x 172 x 2
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. BATCH x (256) x 344 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. BATCH x (256) x 344 x 4
            nn.ConvTranspose2d(128, 1, kernel_size=(4,5), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)

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

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (1) x 1632 x 19
            nn.Conv2d(1,64, (4,3), padding=(1,1), stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 816 x 10
            nn.Conv2d(64,128,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 408 x 6
            nn.Conv2d(128,256,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 204 x 4
            nn.Conv2d(256,512,(4,2), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (512) x 102 x 3
            nn.Conv2d(512,1024,(10,3), padding=(1,1), stride=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (1024) x 48 x 2
            nn.Flatten(),
            nn.Linear(1024*48*2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

in_dim=[102,3]
padding=[1,1]
kernal=[10,3]
stride=2
def out_size(in_dim, padding, kernal, stride):
  x = (in_dim[0]+2*padding[0]-(kernal[0]-1)-1)/stride + 1
  y = (in_dim[1]+2*padding[1]-(kernal[1]-1)-1)/stride + 1
  return x,y

out_size(in_dim, padding, kernal, stride)

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

noise = torch.randn(BATCH_SIZE, 1, 1632, 19, device=device)
torch.squeeze(netD(noise))

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(16, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
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
        real_cpu = torch.unsqueeze(data[1],1).to(device)
        b_size = real_cpu.size(0)
        if i==0:
          print('b_size:', b_size)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = torch.squeeze(netD(real_cpu)).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = torch.squeeze(netD(fake.detach())).view(-1)
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
        output = torch.squeeze(netD(fake)).view(-1)
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

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(torch.permute(torch.squeeze(fake), (0,2,1)), padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(8, 6), dpi=100)

plt.imshow(torch.squeeze(fake.cpu().detach())[0], interpolation='nearest', aspect='auto', origin='lower')

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
ims = [[plt.imshow(i[0], interpolation='nearest', aspect='auto', origin='lower', animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

[i.shape for i in img_list]
