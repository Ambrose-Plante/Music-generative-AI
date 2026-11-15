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

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

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

BATCH_SIZE = 128
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

# import math
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

# Generator Code
dropout_rate = 0.4

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.chords_embedding = nn.Sequential(
            nn.Linear(11,23)
        )

        self.section_net_t1 = nn.Sequential(
            nn.Linear(5,1632)
        )

        self.latent_t1 = nn.Sequential(
            nn.Linear(nz, 1632),
        )

        self.positional_encoding_t1 = PositionalEncoding_simple(max_len=1632) 
        self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=26, nhead=2, dim_feedforward=64, batch_first=True), num_layers=1)

        self.final = nn.Sequential(
            # state size BATCH x 1632 x 26
            nn.Softmax(dim=2)
        )


    def forward(self, noise, chords_array, section):
        intermediate = torch.cat((self.chords_embedding(chords_array), torch.unsqueeze(self.latent_t1(noise), 2), torch.unsqueeze(self.section_net_t1(section), 2)), dim=2)
        # BATCH x 1632 x 23

        intermediate = self.attention_t1(self.positional_encoding_t1(intermediate))# BATCH x 1632 x 24

        return self.final(intermediate)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
# netG.apply(weights_init)

# Print the model
print(netG)

# data = next(iter(trainloader))
# real_vocals = data[1].to(device)
# real_chords = data[2].to(device)
# real_section = data[3].to(device)

# noise = torch.randn(BATCH_SIZE, nz, device=device)
# netG(noise, real_chords, real_section).shape

def conv_size(input, kernal, padding, stride):
  return ((input-kernal+(2*padding))/stride)+1

conv_size(32,3,0,1)
# conv_size(6,4,1,2)

dis_dropout_rate = 0.5
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.embedding = nn.Sequential(
            spectral_norm(nn.Linear(37,43))
        )

        self.positional_encoding_t1 = PositionalEncoding_simple(max_len=1632)
        self.attention_t1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=44, nhead=4, dim_feedforward=64, batch_first=True), num_layers=1)

        self.final = nn.Sequential(
            # input size: BATCH x 1632 x 44
            nn.Flatten()
        )

        self.real_fake = nn.Sequential(
            spectral_norm(nn.Linear(1632*44, 1)),
            nn.Sigmoid()
        )

        self.section_net = nn.Sequential(
            spectral_norm(nn.Linear(1632*44, 5)),
        )

    def forward(self, vocals_array, chords_array):   
        
        intermediate = torch.cat((vocals_array, chords_array), dim=2) # BATCH x 1632 x 37

        intermediate = self.embedding(intermediate)
        intermediate = self.attention_t1(self.positional_encoding_t1(intermediate))

        intermediate = self.final(intermediate)
        return self.real_fake(intermediate), self.section_net(intermediate)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# netD.apply(weights_init)

# Print the model
print(netD)

# data = next(iter(trainloader))
# real_vocals = data[1].to(device)
# real_chords = data[2].to(device)
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
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)
##############

# optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.99))
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.005, momentum=0.9)

# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.99))
##############


### SGD
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
# optimizerG = torch.optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.99))
# optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.99))

def one_value_loss(vocals, b_size):
    values, indices = torch.max(torch.squeeze(vocals), dim=2)
    return torch.mean(torch.abs(values - torch.ones(b_size, 1632).to(device)))

# checkpoint = torch.load('Models/network_B/model_0001_001_batch64_000040.pt')

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
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
valD_losses = []
iters = 0
noise_width = 0.2

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(trainloader, 0):
        real_vocals = data[1].to(device) 
        real_chords = data[2].to(device)
        real_section = data[3].to(device)
        b_size = real_vocals.size(0)
        if i==0:
          print('b_size:', b_size)


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netG.zero_grad()
        netD.zero_grad()
        # Format batch
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_vocals + (noise_width*torch.rand((b_size,1632,26))-(noise_width*0.5)).to(device), real_chords) # blur vocals
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
        fake = netG(noise, real_chords, real_section).detach()
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake + (noise_width*torch.rand((b_size,1632,26))-(noise_width*0.5)).to(device), real_chords.detach())
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

        for g in range(100):
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake + (noise_width*torch.rand((b_size,1632,26))-(noise_width*0.5)).to(device), real_chords)
            # Calculate G's loss based on this output
            OV_loss = one_value_loss(fake, b_size)
            errG = criterion(torch.squeeze(output[0]), label) + 0.25*criterion_sec(torch.squeeze(output[1]), real_section) + OV_loss ## section less important in beginning
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output[0].mean().item()
            # Update G
            optimizerG.step()
            netD.zero_grad()

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

    if epoch%10==0:
      torch.save({
                  'epoch': epoch,
                  'netG_state_dict': netG.state_dict(),
                  'netD_state_dict': netD.state_dict(),
                  'optimizerG_state_dict': optimizerG.state_dict(),
                  'optimizerD_state_dict': optimizerD.state_dict(),
                  'G_losses': G_losses,
                  'D_losses': D_losses,
                  'valD_losses': valD_losses,
                  }, 'Models/network_C/model_0001_001_batch64_' + str(epoch).zfill(6) + '.pt')

    for v, data in enumerate(valloader, 0):
        with torch.no_grad():

            # Real Batch
            val_vocals = data[1].to(device)
            val_chords = data[2].to(device)
            val_section = data[3].to(device)
            b_size = val_vocals.size(0)
            print('b_size:', b_size)
            labels_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(val_vocals + (noise_width*torch.rand((b_size,1632,26))-(noise_width*0.5)).to(device), val_chords)

            # Fake Batch
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise, val_chords, val_section)
            labels_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake.detach() + (noise_width*torch.rand((b_size,1632,26))-(noise_width*0.5)).to(device), val_chords.detach())
            valD_losses.append(criterion(torch.squeeze(output_real[0].detach()), labels_real) + criterion(torch.squeeze(output_fake[0].detach()), labels_fake) + criterion_sec(torch.squeeze(output_real[1].detach()), val_section) + criterion_sec(torch.squeeze(output_fake[1].detach()), val_section))
            print('mean_fake_label:', output_fake[0].mean().item(), 'mean_real_label:', output_real[0].mean().item())
            print('accuracy_fake_section:', (output_fake[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size, 'accuracy_real_section:', (output_real[1].argmax(axis=1)==torch.squeeze(val_section).argmax(axis=1)).sum().item()/b_size)

        iters += 1

netG.eval()
for v, data in enumerate(valloader, 0):
  with torch.no_grad():

      # Real Batch
      val_vocals = data[1].to(device)
      val_chords = data[2].to(device)
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
