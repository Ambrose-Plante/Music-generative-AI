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

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default

creds, _ = default()
gc = gspread.authorize(creds)

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1YFVNyICk6GVUBu3o3bE-1JXqkZaaS0UD7DU4R-twYoo/edit#gid=0')

song_list = gsheets.worksheet('Sheet1').get_all_values()

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
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pydub import AudioSegment

cmajor_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'D_major', 'E_minor', 'E_major', 'F_minor', 'F_major', 'G_major', 'A_minor', 'A_major', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

cmajor_big_ref=[]
for num in np.arange(0,10):
  cmajor_big_ref = cmajor_big_ref + [x+str(num) for x in cmajor_scale]

q=9
filename = sorted(glob.glob('outputs/MIDI/softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/*.npz'))[q]

if 'post_chorus' in filename:
  section = 'post_chorus'
elif 'pre_chorus' in filename:
  section = 'pre_chorus'
elif 'chorus' in filename:
  section = 'chorus'
elif 'verse1' in filename:
  section = 'verse1'
elif 'verse2' in filename:
  section = 'verse2'
else:
  raise Exception('no section found')

song_name = filename.split(section)[0].split('_')[-2]
artist_name = filename.split(section)[0].split('_')[-3].split('/')[-1]

print(song_name, artist_name, section)

for row in song_list:
  if artist_name in row[0] and song_name in row[1]:
    print(row)
    key=[row[3], row[4]]
    artist=row[0]
    song=row[1]
    bpm=int(row[2])

    if row[5]=='AP':
      BASE_DIR = '/content/drive/MyDrive/AP_AP_corrected2/'
    elif row[5]=='JF':
      BASE_DIR = '/content/drive/MyDrive/AP_Josh_corrected/'
    else:  ## not completed by AP or JF
      raise Exception('something wrong')

#############################################################
# real_vocals = 'midi_ML_data/Validation2/' + artist +'_' + song + '_' + section + '_0.npz'
# vocals_array = np.load(real_vocals)['vocals_array']
# vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
# vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
# vocals_array2[:,:-1] = vocals_array
# vocals_array = vocals_array2
#############################################################

vocals_array = np.load(filename)['vocals']
threshold = 0.0
for i,x in enumerate(vocals_array.max(axis=1)):
  if x<threshold:
    vocals_array[i,:-1] = 0
    vocals_array[i,-1] = 1

array_shape=vocals_array.shape
if section=='pre_chorus':
  end_name = 'prechorus'
elif section=='post_chorus':
  end_name = 'post'
else:
  end_name = section

acc_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/' + end_name + '_acc.wav'


dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array.T.shape
figsize = width/float(dpi), 5
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])
ax.imshow(vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);

vocals_array = np.asarray([[float(x) for x in q] for q in vocals_array])
vocals_array = vocals_array.argmax(axis=1)

plt.plot(vocals_array, 'o', markersize=1)

!fluidsynth -ni SalC5Light2.sf2 out.mid -F output.wav -r 44100

sound1 = AudioSegment.from_file("output.wav")+10
sound2 = AudioSegment.from_file(acc_file)-10

combined = sound1.overlay(sound2)

combined.export("combined.wav", format='wav')

from IPython.display import Audio
Audio('combined.wav')

silence = len(cmajor_small_ref)+1
for i,x in enumerate(vocals_array):
  if x!=silence:
    note_start=i
    break

print(note_start)

current_note = vocals_array[note_start]
j=note_start-1 # allows usage of 'j+=1' at the top of loop
midi_list=[]
for i,x in enumerate(vocals_array[note_start:]):
  j+=1
  if current_note==silence:
    if x==silence:
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
        current_note=silence # set to zero
        continue # wait until next note starts
      else: # directly into new note
        current_note=x
        note_start=j

if current_note!=silence: # last note was not added
  midi_list.append([note_start, j, current_note])

midi_list = [[x[0], x[1], x[2]] for x in midi_list]

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

octave_shift=1
velocity = 100

pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello')
pm.instruments.append(inst)

for start,end,pitch in [[x[0],x[1],x[2]] for x in midi_list]:
    if pitch!=len(cmajor_small_ref):
      pitch = cmajor_small_ref[pitch][:-1]+str(int(cmajor_small_ref[pitch][-1])+octave_shift)
      inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(pitch)+key_shift(key), start/sixth_step_per_sec, end/sixth_step_per_sec))
print(inst.notes)

pm.write('out.mid')

!fluidsynth -ni SalC5Light2.sf2 out.mid -F output.wav -r 44100

sound1 = AudioSegment.from_file("output.wav")+10
sound2 = AudioSegment.from_file(acc_file)-10

combined = sound1.overlay(sound2)

combined.export("combined.wav", format='wav')

from IPython.display import Audio
Audio('combined.wav')

!pwd
