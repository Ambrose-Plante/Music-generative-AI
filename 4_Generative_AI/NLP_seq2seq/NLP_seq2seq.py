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
import pickle

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

data_list=[]
for data_loc in glob.glob('NLP_data/Train/*.npz'):
  data=np.load(data_loc)

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

  data_list.append((data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['indices']), from_numpy(data['syll_vec']).float(), torch.FloatTensor(section)))

seq_lengths = np.asarray([(i,(x[2]!=1403).sum().item()) for i,x in enumerate(data_list)])

min_len = min(seq_lengths[:,1])
max_len = max(seq_lengths[:,1])

box_edges = np.arange(min_len, max_len, 5)

batches = []
for i in range(len(box_edges)):
    if i==len(box_edges)-1:
      batches.append(seq_lengths[np.squeeze(np.where(seq_lengths[:,1]>=box_edges[i])),0].tolist())
    else:
      batches.append(seq_lengths[np.squeeze(np.where(np.logical_and(seq_lengths[:,1]>=box_edges[i],seq_lengths[:,1]<box_edges[i+1]))),0].tolist())

batches = [x for x in batches if len(x)>0]

def trainloader(max_batch_size):

  ##shuffle batch order, as well as order of indices in each batch####
  random.shuffle(batches)
  for i in batches:
      random.shuffle(i)
  #######################################

  epoch_batches = []
  for temp in batches:
    for x in [temp[i:i + max_batch_size] for i in range(0, len(temp), max_batch_size)]:
      epoch_batches.append(x)

  random.shuffle(epoch_batches)

  return epoch_batches

def batch_preprocess(batch):
  
  indices = torch.stack([data_list[i][2] for i in batch])
  vocals_array = torch.stack([data_list[i][1] for i in batch])
  sylls_array = torch.stack([data_list[i][3] for i in batch])
  max_seq = (indices!=1403).sum(axis=1).max()
  indices = indices[:,:max_seq]
  vocals_array = vocals_array[:,:max_seq]
  sylls_array = sylls_array[:,:max_seq]

  return indices, vocals_array, sylls_array

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

        return data_loc, from_numpy(data['vocals_array']).float(), from_numpy(data['indices']), from_numpy(data['syll_vec']).float(), torch.FloatTensor(section)

BATCH_SIZE = 10
# trainset = LoadFromList()
# trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True) #use 0 workers for gpu, 1 for cpu

valset = LoadFromFolder(main_dir="./NLP_data/Val/")
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
        seq_len = x.size(1)
        x = torch.cat((x, self.pe[:seq_len].repeat(x.shape[0],1).unsqueeze(2)), dim=2)
        return x

class VocalsEmbedding(nn.Module):

    def __init__(self):
        super(VocalsEmbedding, self).__init__()

        self.vocals_net = nn.Sequential(
            # state size is BATCH*seq_len x 1 x 1632 x 21

            nn.Conv2d(1,1,kernel_size=(10,3),stride=2,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([812,10]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq_len x 1 x 812 x 10

            nn.Conv2d(1,1,kernel_size=(10,2),stride=2,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([402,5]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq_len x 1 x 402 x 5

            nn.Conv2d(1,1,kernel_size=(10,3),stride=2,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([197,2]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq_len x 1 x197 x 2

            nn.Conv2d(1,1,kernel_size=(9,2),stride=1,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([189,1]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq_len x 1 x 189 x 1
        )

        self.out = nn.Sequential(
            nn.Linear(189,149),
            # nn.BatchNorm1d(200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
            # output size is BATCH x seq_len x 149
        )

    def forward(self, vocals_array: Tensor):
        batch_size = vocals_array.size(0)
        intermediate = nn.Flatten(start_dim=2)(nn.Unflatten(0, (batch_size, -1))(self.vocals_net(vocals_array.contiguous().view(-1,1,1632,21))))
        return self.out(intermediate)

# epoch_batches = trainloader(max_batch_size=16)
# batch = epoch_batches[0]

# indices, vocals_array, sylls_array = batch_preprocess(batch)

# VocalsEmbedding()(vocals_array).shape

class SyllableEmbedding(nn.Module):

    def __init__(self):
        super(SyllableEmbedding, self).__init__()

        self.syll_net1 = nn.Sequential(
            # input size is BATCH*seq x 1 x 6 x 42

            nn.Conv2d(1,1,kernel_size=(2,6),stride=1,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([5,37]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq x 1 x 5 x 37

            nn.Conv2d(1,1,kernel_size=(2,6),stride=1,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([4,32]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq x 1 x 4 x 32

            nn.Conv2d(1,1,kernel_size=(2,6),stride=1,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([3,27]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq x 1 x 3 x 27

            nn.Conv2d(1,1,kernel_size=(2,6),stride=1,padding=0),
            # nn.BatchNorm2d(1),
            nn.LayerNorm([2,22]),
            nn.LeakyReLU(inplace=True),
            # state size is BATCH*seq x 1 x 2 x 22
        )

        self.out = nn.Sequential(
            # state size is BATCH x seq x 44
            nn.Linear(44, 149, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
            # output size is BATCH x seq x 149
        )

    def forward(self, syll_array: Tensor):
        batch_size = syll_array.size(0)
        intermediate = nn.Flatten(start_dim=2)(nn.Unflatten(0,(batch_size, -1))(self.syll_net1(syll_array.contiguous().view(-1,1,6,42))))
        return self.out(intermediate)

# sylls_array = data[3]
# SyllableEmbedding()(sylls_array[:,:-1]).shape

def get_tgt_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask

with open('syll_lib.pkl', 'rb') as f:
  syll_lib = pickle.load(f)

def indices_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    return (matrix == pad_token)

def create_pad_mask(array: torch.tensor):
  return nn.Flatten(start_dim=2)(array).sum(axis=2)==0

class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.vocals_embedder = VocalsEmbedding()
        self.syllable_embedder = SyllableEmbedding()
        self.positional_encoder = PositionalEncoding_simple(max_len=200)
        self.transformer = nn.Transformer(
            d_model=dim_model+1,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model+1, num_tokens)

    def forward(self, input_vocals, target_sylls, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        input_vocals = self.vocals_embedder(input_vocals) # * math.sqrt(self.dim_model)
        target_sylls = self.syllable_embedder(target_sylls) # * math.sqrt(self.dim_model)
        input_vocals = self.positional_encoder(input_vocals)
        target_sylls = self.positional_encoder(target_sylls)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        input_vocals = input_vocals.permute(1, 0, 2)
        target_sylls = target_sylls.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(input_vocals, target_sylls, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out

model = Transformer(
    num_tokens=(len(syll_lib)), dim_model=149, num_heads=10, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.2
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# opt = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
# loss_fn = nn.CrossEntropyLoss()
# This was important from their code. Initialize parameters with Glorot or fan_avg.
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

print(model)

# peak=0.002
# step_size = 0.00002
# decay_slope=-0.9

# warmup_steps = int(peak/step_size)
# offset = warmup_steps**(decay_slope)-peak
# lr_sched = np.concatenate((np.arange(0,peak,step_size), np.arange(warmup_steps, warmup_steps+decay_steps)**(decay_slope)-offset))

lr_sched = np.concatenate((np.arange(0,0.002, 0.00002), np.arange(0.001,0.002,0.00001)[::-1], 0.001*np.ones(2000), 0.0005*np.ones(5000)))

plt.plot(lr_sched)

def batch_accuracy(indices, pred):
  temp = []
  for q in range(len(indices)):
    arr1 = [syll_lib[x.item()] for x in indices[q]]
    arr2 = [syll_lib[x.topk(1)[1].item()] for x in pred[q].T]
    temp.append((np.asarray(arr1) == np.asarray(arr2)).mean())

  return np.asarray(temp).mean()

def train_loop(model, opt, loss_fn):
  model.train()
  total_loss = 0

  acc_temp = []
  num_batches=0
  for batch in trainloader(max_batch_size=16):
    num_batches+=1
    indices, vocals_array, syll_vec = batch_preprocess(batch)
    vocals_array = vocals_array.to(device)
    indices = indices.to(device)
    syll_vec = syll_vec.to(device)

    # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
    syll_vec = syll_vec[:,:-1] # syll embedding starts at SOS and goes to -1
    indices = indices[:,1:]    # output indices start at first syll and goes to end
    
    # Get masks
    sequence_length = indices.size(1)
    src_pad_mask = create_pad_mask(vocals_array).to(device)
    tgt_pad_mask = create_pad_mask(syll_vec).to(device)
    tgt_mask = get_tgt_mask(sequence_length).to(device)

    # Standard training with masks
    pred = model(vocals_array, syll_vec, tgt_mask, src_pad_mask, tgt_pad_mask)

    # Permute pred to have batch size first again
    pred = pred.permute(1, 2, 0)
    loss = loss_fn(pred, indices)

    opt.zero_grad()
    loss.backward()
    opt.step()

    acc_temp.append(batch_accuracy(indices.detach(), pred.detach()))
    total_loss += loss.detach().item()
    
  return total_loss / num_batches, np.asarray(acc_temp).mean()

def validation_loop(model, loss_fn, dataloader):
  model.eval()
  total_loss = 0
  
  with torch.no_grad():
    for batch in dataloader:
      vocals_array = batch[1].to(device)
      indices = batch[2].to(device)
      syll_vec = batch[3].to(device)

      # shift the tgt by one so with the <SOS> we predict the token at pos 1
      syll_vec = syll_vec[:,:-1] # syll embedding starts at SOS and goes to -1
      indices = indices[:,1:]    # output indices start at first syll and goes to end
      
      # Get mask to mask out the next words
      sequence_length = indices.size(1)
      src_pad_mask = create_pad_mask(vocals_array).to(device)
      tgt_pad_mask = create_pad_mask(syll_vec).to(device)
      tgt_mask = get_tgt_mask(sequence_length).to(device)

      # Standard training with masks
      pred = model(vocals_array, syll_vec, tgt_mask, src_pad_mask, tgt_pad_mask)

      # Permute pred to have batch size first again
      pred = pred.permute(1, 2, 0)
      loss = loss_fn(pred, indices)

      total_loss += loss.detach().item()
      
    return total_loss / len(dataloader)

starting_epoch = 5500

checkpoint = torch.load('Models/NLP_seq2seq/model' + str(starting_epoch).zfill(6) + '.pt')
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
train_loss_list = checkpoint['train_loss_list']
train_acc_list = checkpoint['train_acc_list']
validation_loss_list = checkpoint['validation_loss_list']



for p in opt.param_groups:
    p['lr'] = 0.0015

epochs = 7000

# Used for plotting later on
# train_loss_list, train_acc_list, validation_loss_list = [], [], []

print("Training and validating model")

for epoch in np.arange(starting_epoch+1, starting_epoch+epochs):
    # if epoch==0:
    #   for p in opt.param_groups:
    #       p['lr'] = lr_sched[1]
    # else:
    #   for p in opt.param_groups:
    #       p['lr'] = lr_sched[epoch]

    print("-"*25, f"Epoch {epoch + 1}","-"*25)
    # print(lr_sched[epoch])
    train_loss, train_accuracy = train_loop(model, opt, loss_fn)
    train_loss_list += [train_loss]
    train_acc_list += [train_accuracy]

    validation_loss = validation_loop(model, loss_fn, valloader)
    validation_loss_list += [validation_loss]

    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print()

    if epoch>(starting_epoch+1) and epoch%100==0:
      torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': opt.state_dict(),
                  'train_loss_list': train_loss_list,
                  'train_acc_list': train_acc_list,
                  'validation_loss_list': validation_loss_list,
                  }, 'Models/NLP_seq2seq/model' + str(epoch).zfill(6) + '.pt')

plt.plot(train_loss_list, label = "Train loss")
plt.plot(validation_loss_list, label = "Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

plt.plot(train_acc_list, label = "Train Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.show()

vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
              , "V", "W", "Y", "Z", "ZH"]

phoneme_ref = vowels + consonants + ['-', 'BOS', 'EOS']


cmajor_scale = ['C', 'D', 'D#', 'E', 'F', 'G', 'A', 'B']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref
full_vocal_ref = cmajor_small_ref + ['BOS', 'EOS']


def syll_to_vec(syllable):
  max_phonemes_per_syllable=6
  num_unique_phonemes=len(phoneme_ref)

  syll_embed_vec = np.zeros((max_phonemes_per_syllable, num_unique_phonemes))

  for j,phoneme in enumerate(syllable.split()):
    if syllable=='PAD':
      break
    else:
      q = [q for q,temp in enumerate(phoneme_ref) if phoneme==temp]
      if len(q)==1:
          syll_embed_vec[j,q] = 1
      else:
        raise Exception('not exactly one phoneme match')

  return syll_embed_vec

def vec_to_syll(syll_embed_vec):
  syll_temp=[]
  for phoneme_vec in syll_embed_vec:
    if phoneme_vec.sum()>0:
      syll_temp.append(phoneme_ref[phoneme_vec.argmax()])

  return ' '.join(syll_temp)

BOS_token = np.zeros((6,42))
BOS_token[0,-2] = 1

def predict(model, input_sequence, max_length=200, BOS_token=BOS_token, EOS_index=1):

    model.eval()

    y_input = torch.tensor([[BOS_token]], device=device).float()

    num_tokens = len(input_sequence[0])

    src_pad_mask = create_pad_mask(input_sequence).to(device)

    output = []
    for _ in range(num_tokens):
        tgt_mask = get_tgt_mask(y_input.size(1)).to(device)
        pred = model(input_sequence, y_input, tgt_mask, src_pad_mask)

        next_indx = pred.topk(1)[1].view(-1)[-1].item() #indx with highest probability
        output.append(syll_lib[next_indx])
        next_item = torch.tensor([[syll_to_vec(syll_lib[next_indx])]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1).float()

        # Stop if model predicts end of sentence
        if next_indx == EOS_index:
            break

    return output

# data = data_list[30]

# song_name = data[0]
# vocals_array = data[1]
# indices = data[2]
# syll_vec = data[3]

# max_seq = (vocals_array.flatten(start_dim=1).sum(axis=1)!=0).sum().item()
# vocals_array = vocals_array[:max_seq].unsqueeze(0).to(device)

# print(song_name)
# output_lyrics = predict(model, vocals_array, max_length=200, BOS_token=BOS_token, EOS_index=1)

data = next(iter(valloader))
q=3
song_name = data[0][q]
print(song_name)
vocals_array = data[1][q].unsqueeze(0).to(device)

output_lyrics = predict(model, vocals_array, max_length=200, BOS_token=BOS_token, EOS_index=1)

output_lyrics

print(len(output_lyrics), 'notes')
for i,note in enumerate(vocals_array[0][1:]):
  if note.sum()>0:
    if i<len(output_lyrics):
      print(full_vocal_ref[note.sum(axis=0).argmax()], ' | ', output_lyrics[i])
    else:
      print(full_vocal_ref[note.sum(axis=0).argmax()], ' | ', 'end of output')
  else:
    print('PAD', ' | ', output_lyrics[i])

# torch.save(model.state_dict(), 'Models/NLP_seq2seq/model.pt')

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

plot_grad_flow(model.named_parameters())

for n, p in netD.named_parameters():
  print(p)
