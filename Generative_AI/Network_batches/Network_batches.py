from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/integrated

import os
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from torchvision.transforms import ToTensor
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

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

        for i in range(output_array.shape[0]): ## make silence a one at the highest pitch, highest syllable
          if output_array[i].flatten().sum()==0:
            output_array[i,-1,-1]=1

        return data_loc, torch.transpose(from_numpy(output_array).float(),0,2), from_numpy(data['input_syll_vec']).float(), from_numpy(data['vox_input']).float(), from_numpy(data['acc_input']).float(), from_numpy(data['chords_array']).float(), from_numpy(data['key_array']).float(), from_numpy(data['bpm_array']).float()

# !pip install wandb
# import wandb
# wandb.init()

BATCH_SIZE = 3

trainset = LoadFromFolder(main_dir="./data/train/")
random_sampler = torch.utils.data.RandomSampler(data_source="./data/train/", replacement=True, num_samples=900)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=8, drop_last=True, sampler=random_sampler)

valset = LoadFromFolder(main_dir="./data/val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

testset = LoadFromFolder(main_dir="./data/test/")
testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)

names = ['output_array', 'input_syll_vec', 'vox_input', 'acc_input', 'chords_array', 'key_array', 'bpm_array']

for i, x in enumerate(next(iter(trainloader))):
  if i>0:
    print(names[i-1], i, x.shape)  # prints shape of image with single batch
  else:
    print(x)

# in_dim=[100,1632]
# padding=[0,1]
# kernal=[4,3]
# stride=1
# def out_size(in_dim, padding, kernal, stride):
#   x = (in_dim[0]+2*padding[0]-(kernal[0]-1)-1)/stride + 1
#   y = (in_dim[1]+2*padding[1]-(kernal[1]-1)-1)/stride + 1
#   return x,y

# for i in range(13):
#   in_dim = out_size(in_dim, padding, kernal, stride)
#   print(in_dim)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        ######################input_syll_vec##########################################
        # BATCHx200x7x39 => BATCHx200x7x7
        self.conv_s1 = nn.Conv2d(in_channels=200, out_channels=200, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s1 = nn.BatchNorm2d(200)
        self.conv_s2 = nn.Conv2d(in_channels=200, out_channels=100, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s2 = nn.BatchNorm2d(100)
        self.conv_s3 = nn.Conv2d(in_channels=100, out_channels=50, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s3 = nn.BatchNorm2d(50)
        self.conv_s4 = nn.Conv2d(in_channels=50, out_channels=40, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s4 = nn.BatchNorm2d(40)
        self.conv_s5 = nn.Conv2d(in_channels=40, out_channels=30, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s5 = nn.BatchNorm2d(30)
        self.conv_s6 = nn.Conv2d(in_channels=30, out_channels=20, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s6 = nn.BatchNorm2d(20)
        self.conv_s7 = nn.Conv2d(in_channels=20, out_channels=10, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s7 = nn.BatchNorm2d(10)
        self.conv_s8 = nn.Conv2d(in_channels=10, out_channels=9, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s8 = nn.BatchNorm2d(9)
        self.conv_s9 = nn.Conv2d(in_channels=9, out_channels=8, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s9 = nn.BatchNorm2d(8)
        self.conv_s10 = nn.Conv2d(in_channels=8, out_channels=7, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s10 = nn.BatchNorm2d(7)
        self.conv_s11 = nn.Conv2d(in_channels=7, out_channels=6, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s11 = nn.BatchNorm2d(6)
        self.conv_s12 = nn.Conv2d(in_channels=6, out_channels=5, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s12 = nn.BatchNorm2d(5)
        self.conv_s13 = nn.Conv2d(in_channels=5, out_channels=4, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s13 = nn.BatchNorm2d(4)
        self.conv_s14 = nn.Conv2d(in_channels=4, out_channels=3, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s14 = nn.BatchNorm2d(3)
        self.conv_s15 = nn.Conv2d(in_channels=3, out_channels=2, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s15 = nn.BatchNorm2d(2)
        self.conv_s16 = nn.Conv2d(in_channels=2, out_channels=1, padding=(1,0), kernel_size=(3,3), stride=1)
        self.bn_s16 = nn.BatchNorm2d(1)

        # input_syll_vec (1): BATCHx1x7x7 => BATCHx39*1632
        self.d1 = nn.Linear(1*7*7, 39*1632)
        self.d1_do = nn.Dropout(p=0.4)
        #######################################end input syll vec##############################################

        #######################################Start vox input#################################################
        # vox input (2): 1x120x3265 => 16x61x1632
        self.conv_v1 = nn.Conv2d(in_channels=1, out_channels=16, padding=(2,1), kernel_size=(4,5), stride=2)
        self.bn_v1 = nn.BatchNorm2d(16)

        # 16x61x1632 => 1x61x1632
        self.conv_v2 = nn.Conv2d(in_channels=16, out_channels=15, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v2 = nn.BatchNorm2d(15)
        self.conv_v3 = nn.Conv2d(in_channels=15, out_channels=14, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v3 = nn.BatchNorm2d(14)
        self.conv_v4 = nn.Conv2d(in_channels=14, out_channels=13, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v4 = nn.BatchNorm2d(13)
        self.conv_v5 = nn.Conv2d(in_channels=13, out_channels=12, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v5 = nn.BatchNorm2d(12)
        self.conv_v6 = nn.Conv2d(in_channels=12, out_channels=11, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v6 = nn.BatchNorm2d(11)
        self.conv_v7 = nn.Conv2d(in_channels=11, out_channels=10, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v7 = nn.BatchNorm2d(10)
        self.conv_v8 = nn.Conv2d(in_channels=10, out_channels=9, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v8 = nn.BatchNorm2d(9)
        self.conv_v9 = nn.Conv2d(in_channels=9, out_channels=8, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v9 = nn.BatchNorm2d(8)
        self.conv_v10 = nn.Conv2d(in_channels=8, out_channels=7, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v10 = nn.BatchNorm2d(7)
        self.conv_v11 = nn.Conv2d(in_channels=7, out_channels=6, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v11 = nn.BatchNorm2d(6)
        self.conv_v12 = nn.Conv2d(in_channels=6, out_channels=5, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v12 = nn.BatchNorm2d(5)
        self.conv_v13 = nn.Conv2d(in_channels=5, out_channels=4, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v13 = nn.BatchNorm2d(4)
        self.conv_v14 = nn.Conv2d(in_channels=4, out_channels=3, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v14 = nn.BatchNorm2d(3)
        self.conv_v15 = nn.Conv2d(in_channels=3, out_channels=2, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v15 = nn.BatchNorm2d(2)
        self.conv_v16 = nn.Conv2d(in_channels=2, out_channels=1, padding=(1,1), kernel_size=(3,3), stride=1)
        self.bn_v16 = nn.BatchNorm2d(1)
        #######################################End vox input#################################################

        #######################################Start transformer#################################################
        encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=10, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        #######################################End transformer#################################################

        #######################################Start decoder#################################################
        # 1x1x100x1632 => 1x200x61x1632
        self.conv_c1 = nn.Conv2d(in_channels=1, out_channels=15, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c1 = nn.BatchNorm2d(1632)
        self.conv_c2 = nn.Conv2d(in_channels=15, out_channels=30, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c2 = nn.BatchNorm2d(1632)
        self.conv_c3 = nn.Conv2d(in_channels=30, out_channels=45, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c3 = nn.BatchNorm2d(1632)
        self.conv_c4 = nn.Conv2d(in_channels=45, out_channels=60, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c4 = nn.BatchNorm2d(1632)
        self.conv_c5 = nn.Conv2d(in_channels=60, out_channels=75, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c5 = nn.BatchNorm2d(1632)
        self.conv_c6 = nn.Conv2d(in_channels=75, out_channels=90, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c6 = nn.BatchNorm2d(1632)
        self.conv_c7 = nn.Conv2d(in_channels=90, out_channels=105, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c7 = nn.BatchNorm2d(1632)
        self.conv_c8 = nn.Conv2d(in_channels=105, out_channels=120, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c8 = nn.BatchNorm2d(1632)
        self.conv_c9 = nn.Conv2d(in_channels=120, out_channels=135, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c9 = nn.BatchNorm2d(1632)
        self.conv_c10 = nn.Conv2d(in_channels=135, out_channels=150, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c10 = nn.BatchNorm2d(1632)
        self.conv_c11 = nn.Conv2d(in_channels=150, out_channels=165, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c11 = nn.BatchNorm2d(1632)
        self.conv_c12 = nn.Conv2d(in_channels=165, out_channels=180, padding=(0,1), kernel_size=(4,3), stride=1)
        self.bn_c12 = nn.BatchNorm2d(1632)
        self.conv_c13 = nn.Conv2d(in_channels=180, out_channels=200, padding=(0,1), kernel_size=(4,3), stride=1)
        #######################################End decoder#################################################

        self.end = nn.LogSoftmax(dim=1)

    def forward(self, input_syll_vec, vox_input, comb):

#####################Start Syll_input_vec####################################
        # initial convs
        syll = self.conv_s1(input_syll_vec)
        syll = F.relu(syll)
        syll = self.bn_s1(syll)
        syll = self.conv_s2(syll)
        syll = F.relu(syll)
        syll = self.bn_s2(syll)
        syll = self.conv_s3(syll)
        syll = F.relu(syll)
        syll = self.bn_s3(syll)
        syll = self.conv_s4(syll)
        syll = F.relu(syll)
        syll = self.bn_s4(syll)
        syll = self.conv_s5(syll)
        syll = F.relu(syll)
        syll = self.bn_s5(syll)
        syll = self.conv_s6(syll)
        syll = F.relu(syll)
        syll = self.bn_s6(syll)
        syll = self.conv_s7(syll)
        syll = F.relu(syll)
        syll = self.bn_s7(syll)
        syll = self.conv_s8(syll)
        syll = F.relu(syll)
        syll = self.bn_s8(syll)
        syll = self.conv_s9(syll)
        syll = F.relu(syll)
        syll = self.bn_s9(syll)
        syll = self.conv_s10(syll)
        syll = F.relu(syll)
        syll = self.bn_s10(syll)
        syll = self.conv_s11(syll)
        syll = F.relu(syll)
        syll = self.bn_s11(syll)
        syll = self.conv_s12(syll)
        syll = F.relu(syll)
        syll = self.bn_s12(syll)
        syll = self.conv_s13(syll)
        syll = F.relu(syll)
        syll = self.bn_s13(syll)
        syll = self.conv_s14(syll)
        syll = F.relu(syll)
        syll = self.bn_s14(syll)
        syll = self.conv_s15(syll)
        syll = F.relu(syll)
        syll = self.bn_s15(syll)
        syll = self.conv_s16(syll)
        syll = F.relu(syll)
        syll = self.bn_s16(syll)
        
        # flatten =>  BATCH x (1x7x7)
        syll = syll.flatten(start_dim = 1)
        # liner => BATCHx39x1632
        syll = self.d1(syll)
        syll = F.relu(syll)
        syll = self.d1_do(syll) # dropout

        # reshape
        syll = torch.unsqueeze(syll, 1) # BATCHx(39*1632) => BATCHx1x(39*1632)
        syll = torch.reshape(syll, (-1,1,39,1632)) # BATCHx1x(39*1632) => BATCHx1x39x1632)
#####################End Syll_input_vec######################################

#####################Start Vox_input#########################################
        vox_input = torch.unsqueeze(vox_input, 1) # BATCHx120x3265 => BATCHx1x120x3265
        vox = self.conv_v1(vox_input) # BATCHx1x120x3265 => BATCHx16x61x1632
        vox = F.relu(vox)
        vox = self.bn_v1(vox)

        # BATCHx16x61x1632 => BATCHx1x61x1632
        vox = self.conv_v2(vox)
        vox = F.relu(vox)
        vox = self.bn_v2(vox)
        vox = self.conv_v3(vox)
        vox = F.relu(vox)
        vox = self.bn_v3(vox)
        vox = self.conv_v4(vox)
        vox = F.relu(vox)
        vox = self.bn_v4(vox)
        vox = self.conv_v5(vox)
        vox = F.relu(vox)
        vox = self.bn_v5(vox)
        vox = self.conv_v6(vox)
        vox = F.relu(vox)
        vox = self.bn_v6(vox)
        vox = self.conv_v7(vox)
        vox = F.relu(vox)
        vox = self.bn_v7(vox)
        vox = self.conv_v8(vox)
        vox = F.relu(vox)
        vox = self.bn_v8(vox)
        vox = self.conv_v9(vox)
        vox = F.relu(vox)
        vox = self.bn_v9(vox)
        vox = self.conv_v10(vox)
        vox = F.relu(vox)
        vox = self.bn_v10(vox)
        vox = self.conv_v11(vox)
        vox = F.relu(vox)
        vox = self.bn_v11(vox)
        vox = self.conv_v12(vox)
        vox = F.relu(vox)
        vox = self.bn_v12(vox)
        vox = self.conv_v13(vox)
        vox = F.relu(vox)
        vox = self.bn_v13(vox)
        vox = self.conv_v14(vox)
        vox = F.relu(vox)
        vox = self.bn_v14(vox)
        vox = self.conv_v15(vox)
        vox = F.relu(vox)
        vox = self.bn_v15(vox)
        vox = self.conv_v16(vox)
        vox = F.relu(vox)
        vox = self.bn_v16(vox)

####################End Vox_input########################################

####################Start Combined transformer#################################
        # BATCHx1x61x1632 + BATCHx1x39x1632) => BATCHx1x100x1632
        comb[:,:,:61,:] = vox
        comb[:,:,61:,:] = syll

        comb = torch.reshape(comb, (-1,100,1632)) # BATCHx1x100x1632 => BATCHx100x1632
        comb = torch.transpose(comb, 1,2) # BATCHx100x1632 => BATCHx1632x100
        comb = self.transformer_encoder(comb) # BATCHx1632x100 => BATCHx1632x100
        comb = torch.transpose(comb,1,2) # BATCHx1632x100 => BATCHx100x1632
        comb = torch.unsqueeze(comb, 1) # BATCHx100x1632 => BATCHx1x100x1632
####################End Combined transformer#############################

####################Start decoder######################################

        comb = self.conv_c1(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c1(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c2(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c2(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c3(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c3(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c4(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c4(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c5(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c5(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c6(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c6(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c7(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c7(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c8(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c8(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c9(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c9(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c10(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c10(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c11(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c11(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c12(comb)
        comb = F.relu(comb)

        comb = comb.permute(0,3,1,2) # BATCHxHxWx1632 => BATCHx1632xHxW
        comb = self.bn_c12(comb)
        comb = comb.permute(0,2,3,1) # BATCHx1632xHxW => BATCHxHxWx1632

        comb = self.conv_c13(comb) # => BATCHx200x61x1632
####################End decoder########################################

####################Start final softmax##################################
        comb = torch.reshape(comb, (-1, 200*61,1632))
        comb = self.end(comb)
        comb = torch.reshape(comb, (-1, 200, 61, 1632))
####################End final softmax####################################

        return comb

## test the model with 1 batch
# model = MyModel()

# for name, output_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array in trainloader:
#     print("batch size:", output_array.shape)
#     comb = torch.zeros(BATCH_SIZE,1,100,1632)
#     out = model(input_syll_vec, vox_input, comb)
#     # print(vox.shape)
#     # print(syll.shape)
#     print(out.shape)
#     break

def pitch_max_indices(output_array):
    target_max1, target_indexes1 = torch.max(output_array, dim=1)
    return target_max1

def syll_max_indices(output_array):
    target_max1, target_indexes1 = torch.max(output_array, dim=2)
    return target_max1

def my_loss(output, target):
    return nn.NLLLoss(reduction='mean')(syll_max_indices(output), torch.max(syll_max_indices(target), 1)[1]) + nn.NLLLoss(reduction='mean')(pitch_max_indices(output), torch.max(pitch_max_indices(target), 1)[1])

  # compute accuracy
def get_accuracy(output, target):
    with torch.no_grad():
        ''' Obtain average (over batch) accuracy for training round '''
        pitch_acc = (torch.max(pitch_max_indices(output), dim=1)[1]==torch.max(pitch_max_indices(target), dim=1)[1]).sum()/(1632*output.size()[0])
        syll_acc = (torch.max(syll_max_indices(output), dim=1)[1]==torch.max(syll_max_indices(target), dim=1)[1]).sum()/(1632*output.size()[0])
        return pitch_acc.detach().item(), syll_acc.detach().item()

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
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

learning_rate = 0.01
num_epochs = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyModel()

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# wandb.watch(model, log='all', log_freq=1)

best_val_loss = 100000000

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_pitch_acc = 0.0
    train_syll_acc = 0.0

    model = model.train()

    ## training step
    for i, (name, output_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array) in enumerate(trainloader):
          
        output_array = output_array.to(device, dtype=torch.int64).detach()
        input_syll_vec = input_syll_vec.to(device).detach()
        vox_input = vox_input.to(device).detach().detach()
        comb = torch.zeros(BATCH_SIZE,1,100,1632).to(device).detach()

        ## forward + backprop + loss
        
        output = model(input_syll_vec, vox_input, comb)

        loss = my_loss(output, output_array)

        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        ## update model params
        optimizer.step()

        # plot_grad_flow(model.named_parameters())

        train_running_loss += loss.detach().item()

        temp = get_accuracy(output, output_array)

        # print('check 13')
        # !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'

        train_pitch_acc += temp[0]
        train_syll_acc += temp[1]

        if i%18==0:
          print('\r' + str(i) + ' / ' + str(len(trainloader)) + ': ' + str(loss.detach().item()) + ', ' + str(temp[0]) + '/' + str(temp[1]), end='')

        # print(train_running_loss, loss.detach().item())
        del loss, output, output_array, input_syll_vec, vox_input, comb
        torch.cuda.empty_cache()
    #     break
    # break
    
    model.eval()
    print('\nEpoch: Train %d | Loss: %.20f | Accuracy(pitch/syll): %.2f / %.2f' \
          %(epoch, train_running_loss / (i+1), train_pitch_acc/(i+1), train_syll_acc/(i+1)))

    with torch.no_grad():
        ## validation step
        val_running_loss = 0
        val_pitch_acc = 0
        val_syll_acc = 0
        for i, (name, output_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array) in enumerate(valloader):

            output_array = output_array.to(device, dtype=torch.int64).detach()
            input_syll_vec = input_syll_vec.to(device).detach()
            vox_input = vox_input.to(device).detach().detach()
            comb = torch.zeros(BATCH_SIZE,1,100,1632).to(device).detach()

            ## forward + loss
            output = model(input_syll_vec, vox_input, comb)

            val_running_loss += my_loss(output, output_array).detach().item()

            temp = get_accuracy(output, output_array)

            val_pitch_acc += temp[0]
            val_syll_acc += temp[1]
            del output, output_array, input_syll_vec, vox_input, comb
            torch.cuda.empty_cache()
        
        print('Epoch: Vali %d | Loss: %.20f | Accuracy(pitch/syll): %.2f / %.2f' \
              %(epoch, val_running_loss / (i+1), val_pitch_acc/(i+1), val_syll_acc/(i+1)))
        print('\n')

        if (val_running_loss/(i+1))<best_val_loss:
          print('New val loss record. Saving model weights...')
          best_val_loss = (val_running_loss/(i+1))

          torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'best_val_loss': best_val_loss,
                      }, 'models/best_model.pt')

learning_rate = 0.01

model = MyModel()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']

model.train()

plot_grad_flow(model.named_parameters())

# for n, p in model.named_parameters():
#     if(p.requires_grad) and ("bias" not in n):
#         print(n, p.grad.abs().mean())
        # max_grads.append(p.grad.abs().max())

# import torch
# import gc
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             print(type(obj), obj.size())
#             # del obj
#     except:
#         pass

model.eval()

with torch.no_grad():
    for i, (name, output_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array) in enumerate(valloader, 0):

        output_array = output_array.to(device, dtype=torch.int64).detach()
        input_syll_vec = input_syll_vec.to(device).detach()
        vox_input = vox_input.to(device).detach().detach()
        comb = torch.zeros(BATCH_SIZE,1,100,1632).to(device).detach()

        ## forward + backprop + loss
        outputs = model(input_syll_vec, vox_input, comb)

        val_pitch_acc, val_syll_acc = get_accuracy(outputs, output_array)
            
        print(name, 'Val Accuracy(pitch/syll): %.2f / %.2f' \
              %(val_pitch_acc, val_syll_acc))

with torch.no_grad():
    for i, (name, output_array, input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array) in enumerate(testloader, 0):

        output_array = output_array.to(device, dtype=torch.int64).detach()
        input_syll_vec = input_syll_vec.to(device).detach()
        vox_input = vox_input.to(device).detach().detach()
        comb = torch.zeros(1,1,100,1632).to(device).detach()

        ## forward + backprop + loss
        outputs = model(input_syll_vec, vox_input, comb)

        temp = get_accuracy(outputs, output_array)
            
        print(name, 'Test Accuracy(pitch/syll): %.2f / %.2f' \
              %(temp[0], temp[1]))  

i=600
print(torch.max(pitch_max_indices(output_array)[:,:,i], 1)[1])
print(torch.max(pitch_max_indices(outputs)[:,:,i], 1)[1])

syll_max_indices(outputs)[:,:,i]

nn.NLLLoss()(syll_max_indices(outputs)[:,:,i], torch.max(syll_max_indices(output_array)[:,:,i], 1)[1])

data_loc = 'data/test/Dua Lipa_Break My Heart_0_0_verse1.npz'

data = np.load(data_loc)
output_array = torch.transpose(from_numpy(data['output_array']).float(),0,2)
input_syll_vec = from_numpy(data['input_syll_vec']).float()
vox_input = from_numpy(data['vox_input']).float()
acc_input = from_numpy(data['acc_input']).float()
chords_array = from_numpy(data['chords_array']).float()
key_array = from_numpy(data['key_array']).float()
bpm_array = from_numpy(data['bpm_array']).float()

output_array = torch.unsqueeze(output_array, 0).to(device)
input_syll_vec = torch.unsqueeze(input_syll_vec, 0).to(device)
vox_input = torch.unsqueeze(vox_input, 0).to(device)
acc_input = torch.unsqueeze(acc_input, 0).to(device)
chords_array = torch.unsqueeze(chords_array, 0).to(device)
key_array = torch.unsqueeze(key_array, 0).to(device)
bpm_array = torch.unsqueeze(bpm_array, 0).to(device)

outputs = model(input_syll_vec, vox_input, acc_input, chords_array, key_array, bpm_array)
test_acc += get_accuracy(outputs, output_array)
print(' Accuracy: %.2f'%( get_accuracy(outputs, output_array)))

nn.Softmax2d()(outputs)

in_dim=[7,39]
padding=[1,0]
kernal=[3,3]
stride=1
def out_size(in_dim, padding, kernal, stride):
  x = (in_dim[0]+2*padding[0]-(kernal[0]-1)-1)/stride + 1
  y = (in_dim[1]+2*padding[1]-(kernal[1]-1)-1)/stride + 1
  return x,y

for i in range(16):
  in_dim = out_size(in_dim, padding, kernal, stride)
  print(in_dim)
