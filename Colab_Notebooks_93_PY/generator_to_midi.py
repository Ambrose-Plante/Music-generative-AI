import numpy as np
%cd /content
!rm -rf temp
!mkdir temp
%cd temp

!which python # should return /usr/local/bin/python
!python --version
!echo $PYTHONPATH
%env PYTHONPATH=

%%bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

!which conda # should return /usr/local/bin/conda
!conda --version # should return 4.5.4
!which python # still returns /usr/local/bin/python

%%bash
conda install --channel defaults conda python=3.6 --yes
conda update --channel defaults --all --yes

!conda --version # now returns 4.8.3
!python --version # now returns Python 3.6.10 :: Anaconda, Inc.
import sys
sys.path

_ = (sys.path
        .append("/usr/local/lib/python3.6/site-packages"))

!conda install -c conda-forge pickle5 --yes
!conda install -c roebel pretty_midi --yes
!apt install fluidsynth
!cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2
! conda install -yc conda-forge pydub

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import os
import pickle5
import glob
import IPython.display
import pretty_midi
from pretty_midi.utilities import note_number_to_name, note_name_to_number
import json
import csv
import random
import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pydub import AudioSegment
from Models.dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section.dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section import Generator

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator(ngpu).to(device)

weights_path = 'Models/dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/model_0005_005_batch32_001967.pt'
# weights_path = 'Models/dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/model001233.pt'

netG.load_state_dict(torch.load(weights_path,  map_location=torch.device('cpu'))['netG_state_dict'])
netG.eval()

cmajor_chords = ['Cmajor', 'Dminor', 'Eminor', 'Emajor', 'Fmajor', 'Gmajor', 'Aminor', 'Bminor']
cmajor_scale = ['C', 'D', 'D#', 'E', 'F', 'G', 'A', 'B']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

artist_name = 'Justin Bieber'
song_name = 'Peaches'
section_name = 'chorus'

with open('song_list.csv', newline='', mode='r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if artist_name in row[0] and song_name in row[1]:
          print(row)
          key=[row[3], row[4]]
          artist=row[0]
          song=row[1]
          bpm=int(row[2])

data_loc = 'data/Val/' + artist +'_' + song + '_' + section_name + '_0.npz'
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

real_vocals = torch.from_numpy(data['vocals_array']).float()
real_chords = torch.from_numpy(data['chords_array']).float()
section = torch.FloatTensor(section)

plt.figure(figsize=(8, 6), dpi=100)
plt.yticks(np.arange(len(cmajor_chords)), cmajor_chords)
plt.imshow(real_chords.T, interpolation='nearest', aspect='auto', origin='lower')

plt.savefig('out/chords.png', dpi=300)

# Fake Batch
noise = torch.randn(1, 100, device=device)
fake = netG(noise, torch.unsqueeze(torch.unsqueeze(real_chords, 0), 0), torch.unsqueeze(section, 0)).detach().cpu()

vocals_array = fake[0,0]
# vocals_array = real_vocals
array_shape=vocals_array.shape
if section_name=='pre_chorus':
  acc_file = 'midi_data/' + artist + '_' + song + '/prechorus_acc.wav'
elif section_name=='post_chorus':
  acc_file = 'midi_data/' + artist + '_' + song + '/post_acc.wav'
else:
  acc_file = 'midi_data/' + artist + '_' + song + '/' + section_name + '_acc.wav'

zero_x = np.where(vocals_array<0.05)
# vocals_array[zero_x]=0
vocals_array[np.where(vocals_array>0.1)] = 1

plt.figure(figsize=(8, 6), dpi=100)
# plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
plt.imshow(vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')
plt.yticks(range(len(cmajor_small_ref)), cmajor_small_ref)
plt.colorbar()

plt.savefig('out/fake.png', dpi=300)

vocals_array = np.asarray([[0] + [float(x) for x in q] for q in vocals_array])
vocals_array = vocals_array.argmax(axis=1)

plt.plot(vocals_array, 'o', markersize=1)

for i,x in enumerate(vocals_array):
  if x!=0:
    note_start=i
    break

print(note_start)

current_note = vocals_array[note_start]
j=note_start-1 # allows usage of 'j+=1' at the top of loop
midi_list=[]
for x in vocals_array[note_start:]:
  j+=1
  if current_note==0:
    if x==0:
      continue # wait until next note
    else: # a new note
      current_note=x
      note_start=j
      continue # nothing more to do
  elif x==current_note: # note is still going, don't do anything
    continue
  else: # note has ended
      midi_list.append([note_start, j, current_note])
      if x==0:
        current_note=0 # set to zero
        continue # wait until next note starts
      else: # directly into new note
        current_note=x
        note_start=j
        
midi_list = [[x[0], x[1], x[2]-1] for x in midi_list]

# current_note = vocals_array[note_start]
# j=note_start-1 # allows usage of 'j+=1' at the top of loop
# midi_list=[]
# for x in vocals_array[note_start:]:
#   j+=1
#   if j==len(vocals_array)-1: # last step. clean up and exit
#     if current_note==0: # silence. just exit.
#       break
#     else: # there was a note still going. write it and exit
#       midi_list.append([note_start, j, current_note])
#       break

#   if current_note==0:
#     if x==0:
#       continue # wait until next note
#     else: # a new note
#       current_note=x
#       note_start=j
#       continue # nothing more to do
#   elif x==current_note: # note is still going, don't do anything
#     continue
#   else: # note may have ended
#       if x!=current_note and vocals_array[j+1]!=current_note: # two different pitches follow; note has ended
#         midi_list.append([note_start, j, current_note])
#         if vocals_array[j+1]==vocals_array[j+2]: ## clear that new note has started
#           if vocals_array[j+1]==0: # it is silence
#             current_note=0 # set to zero
#             continue # wait until next note starts
#           else: # it is a new note
#             current_note=x
#             note_start=j
#             continue
#         else: ## unclear what new note is. Look at third next one
#           if vocals_array[j+1]==vocals_array[j+3]: ## Assume this is the note
#             if vocals_array[j+1]==0: # it is silence
#               current_note=0 # set to zero
#               continue # wait until next note starts
#             else: # it is a new note
#               current_note=x
#               note_start=j
#               continue
#           elif vocals_array[j+2]==vocals_array[j+3]: ## Assume this is the note
#             if vocals_array[j+2]==0: # it is silence
#               current_note=0 # set to zero
#               j=j+1 # skip next step
#               continue # wait until next note starts
#             else: # it is a new note
#               current_note=x
#               note_start=j
#               j=j+1
#               continue
#           else: # next three notes are all different
#             current_note=0 ## assume next note is silence and try again next step
#             continue

if len(cmajor_small_ref)!=array_shape[1]:
  raise Exception('check if scale has changed')

bps=bpm/60
steps_per_sec=bps*4
sixth_step_per_sec=steps_per_sec*6

def key_shift(key_list):
  key=key_list[0]
  key_quality = key_list[1]

  if key_quality=='major':
    base_key = note_name_to_number("C0")
  elif key_quality=='minor':
    base_key = note_name_to_number("A0")
  else:
    raise Exception('key naming issue')

  return -base_key+note_name_to_number(key+'0')

octave_shift=2
velocity = 100

pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello') 
pm.instruments.append(inst)

for start,end,pitch in [[x[0],x[1],cmajor_small_ref[x[2]][:-1]+str(int(cmajor_small_ref[x[2]][-1])+octave_shift)] for x in midi_list]:
    inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(pitch)+key_shift(key), start/sixth_step_per_sec, end/sixth_step_per_sec))
print(inst.notes)

pm.write('out/out.mid')

!fluidsynth -ni SalC5Light2.sf2 out/out.mid -F output.wav -r 44100

sound1 = AudioSegment.from_file("output.wav")+10
sound2 = AudioSegment.from_file(acc_file)-10

combined = sound2.overlay(sound1)

combined.export("combined.wav", format='wav')

from IPython.display import Audio
Audio('combined.wav')

AudioSegment.from_wav("combined.wav").export("out/fake.mp3", format="mp3")

! ls out
