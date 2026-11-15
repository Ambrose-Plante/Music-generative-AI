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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

max_steps = 17*4*4*6

def key_shift(key_list):
  key=key_list[0]
  key_quality = key_list[1]

  if key_quality=='major':
    if key in ['G', 'Ab', 'G#', 'A', 'Bb', 'A#', 'B']:
      key_shift=note_name_to_number('C1')-note_name_to_number(key+'0')
    elif key in ['C', 'Db', 'C#', 'D', 'Eb', 'D#', 'E', 'F', 'Gb', 'F#']:
      key_shift=note_name_to_number('C0')-note_name_to_number(key+'0')
    else:
      raise Exception('key shift issue', key)
  elif key_quality=='minor':
    if key in ['Bb', 'A#', 'B']:
      key_shift=note_name_to_number('A0')-note_name_to_number(key+'0')
    elif key in ['C', 'Db', 'C#', 'D', 'Eb', 'D#']:
      key_shift=note_name_to_number('A0')-note_name_to_number(key+'1')
    elif key in ['A', 'E', 'F', 'Gb', 'F#', 'G', 'Ab', 'G#']:
      key_shift=note_name_to_number('A1')-note_name_to_number(key+'1')
    else:
      raise Exception('key shift issue', key)
  else:
    raise Exception('key naming issue')

  return -key_shift

cmajor_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'E_minor', 'F_major', 'G_major', 'A_minor', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

cmajor_big_ref=[]
for num in np.arange(2,7):
  cmajor_big_ref = cmajor_big_ref + [x+str(num) for x in cmajor_scale]

def chords_process(filename, bpm, key):
    midi_data = pretty_midi.PrettyMIDI(filename)

    midi_list = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            if note.velocity>70:
              quality=0 # major
            else:
              quality=1 # minor

            root_pitch = note_number_to_name(note.pitch-key_shift(key)+12)[:-1]
            if len(root_pitch)>1:
              if root_pitch[1]=='#' or root_pitch[1]=='b':
                root_pitch = root_pitch[:1] + root_pitch[2:] # remove accidental

            cmajor_ind = [i for i,y in enumerate(cmajor_scale) if y == root_pitch]

            if len(cmajor_ind)==1:
              midi_list.append([start, end, cmajor_ind[0]])
            else:
              raise Exception('chord not in scale')

    midi_list = sorted(midi_list, key=lambda z: (z[0], z[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    midi_simple = [(round(z[0]*sixth_step_per_sec), round(z[1]*sixth_step_per_sec), z[2]) for z in midi_list]
    max_steps = 17*4*4*6
    num_scale_degrees = len(c_major_chords)

    chords_array = np.zeros((max_steps, num_scale_degrees))
    for z in midi_simple:
        chords_array[z[0]:z[1],z[2]]=1

    return midi_simple, chords_array

def bpm_to_array(bpm):
    max_bpm=200+1
    bpm_array=np.zeros(max_bpm)
    bpm_array[bpm]=1
    return bpm_array

def vocals_process(vocal_dict, bpm):

    vocal_notes = []
    for note in vocal_dict:
      vocal_notes.append([note['start'], note['start']+note['duration'], note_number_to_name(note['pitch'])])

    for note in vocal_notes:
      if note[2][:-1] not in cmajor_scale:
        note[2] = note_number_to_name(note_name_to_number(note[2])+1)
        if note[2][:-1] not in cmajor_scale:
          raise Exception('still not in scale somehow')

    max_steps = 17*4*4*6
    num_scale_degrees = len(cmajor_small_ref)

    vocals_array = []
    while len(vocals_array)==0:
      for shift in np.arange(-7,7): # try all possible octave shifts
        temp_midi = [[x[0], x[1], x[2][:-1]+str(int(x[2][-1])+shift)] for x in vocal_notes.copy()]
        # if shift==-1:
        if len([x[2] for x in temp_midi if x[2] not in cmajor_small_ref])==0: # if shift fits into small array

          temp2_midi=[]
          for x in temp_midi: ##convert to degrees in cmajor small ref
            arg=[i for i,z in enumerate(cmajor_small_ref) if z==x[2]]
            if len(arg)==1:
              temp2_midi.append((x[0], x[1], arg[0]))
            else:
              raise Exception('note not in cmajor_small_ref')

          temp_array = np.zeros((max_steps, num_scale_degrees))
          for num,x in enumerate(temp2_midi):
            temp_array[x[0]:(x[1]),x[2]]=1
          vocals_array.append(temp_array)

      if len(vocals_array)==0: # move lowest note up an octave
        lowest_note = vocal_notes[np.argmin([note_name_to_number(x[2]) for x in vocal_notes])][2]
        print('moving up note: ', lowest_note)
        for note in vocal_notes:
          if note[2]==lowest_note:
            note[2] = note[2][:-1] + str(int(note[2][-1])+1)

    return vocal_notes, vocals_array

row_start = 2 # the actual row number on the spreadsheet
row_end = 126   # the actual row number on the spreadsheet

song_count=0
for row,track in enumerate(song_list):

  if row not in np.arange(row_start-1,row_end):
    continue

  artist = track[0]
  song = track[1].replace('/', '-')
  bpm = int(track[2])
  ML_set=track[10]

  if track[5]=='AP':
    BASE_DIR = '/content/drive/MyDrive/AP_AP_corrected2/'
  elif track[5]=='JF':
    BASE_DIR = '/content/drive/MyDrive/AP_Josh_corrected/'
  else:  ## not completed by AP or JF
    continue

  if track[6]=='midi_only':
    print(artist, song, 'midi_only')
    continue

  key = [track[3],track[4]]
  song_count += 1
  print('-----------------')
  print(artist + '_' + song)
  syll_file = BASE_DIR + "saved_pronunciations/" + artist + "_" + song +".pkl"
  lyrics_file = BASE_DIR + "saved_lyrics/" + artist + "_" + song +".pkl"

  if not os.path.exists(syll_file):
    raise Exception('syllable file missing')

  if not os.path.exists(lyrics_file):
    raise Exception('lyrics file missing')

  if os.path.exists(syll_file) and os.path.exists(lyrics_file):
    with open(lyrics_file, 'rb') as f:
      lyrics = pickle5.load(f)

    for section in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:
      if len(lyrics[section])>2:
        if section=='verse1':
          vocals_json = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse1.json'
          chords_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/ch_verse1.mid'
          acc_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse1_acc.wav'
          vox_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse1_vocals.wav'

        elif section=='pre_chorus':
          vocals_json = BASE_DIR + 'midi_data/' + artist + '_' + song + '/prechorus.json'
          chords_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/ch_prechorus.mid'
          acc_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/prechorus_acc.wav'
          vox_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/prechorus_vocals.wav'

        elif section=='chorus':
          vocals_json = BASE_DIR + 'midi_data/' + artist + '_' + song + '/chorus.json'
          chords_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'
          acc_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/chorus_acc.wav'
          vox_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/chorus_vocals.wav'

        elif section=='post_chorus':
          vocals_json = BASE_DIR + 'midi_data/' + artist + '_' + song + '/post_chorus.json'
          chords_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/ch_post.mid'
          acc_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/post_acc.wav'
          vox_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/post_vocals.wav'

        elif section=='verse2':
          vocals_json = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse2.json'
          chords_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/ch_verse2.mid'
          acc_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse2_acc.wav'
          vox_wav = BASE_DIR + 'midi_data/' + artist + '_' + song + '/verse2_vocals.wav'
      else:
        continue

      if not os.path.exists(vocals_json):
        raise Exception('vocals_json missing for ', section)

      if not os.path.exists(chords_file):
        raise Exception('chords file missing for ', section)

      if not os.path.exists(acc_wav):
        raise Exception('instrumental wav missing for ', section)

      if not os.path.exists(vox_wav):
        raise Exception('vocals wav missing for ', section)

      with open(vocals_json, 'rb') as f:
        vocals_dict = json.load(f)

      for note in vocals_dict:
        note['pitch'] = note['pitch']-key_shift(key) ## shift to C major

      while len([x['pitch'] for x in vocals_dict if x['pitch']<12])>0: ## there's a negative pitch
        for note in vocals_dict:
          note['pitch'] = note['pitch']+12 ## shift up an octave

      midi_simple, vocals_array = vocals_process(vocals_dict, bpm)
      midi_simple, chords_array = chords_process(chords_file, bpm, key)
      bpm_array = bpm_to_array(bpm)

      for q,array in enumerate(vocals_array):
        if ML_set=='Train':
          outfile = 'midi_ML_data/Train_Cmajor/' + artist + '_' + song + '_' + section + '_' + str(q) + '.npz'
        elif ML_set=='Validation':
          outfile = 'midi_ML_data/Validation_Cmajor/' + artist + '_' + song + '_' + section + '_' + str(q) + '.npz'
        else:
          raise Exception('cannot find which set')
        print(outfile)
        np.savez(outfile, vocals_array=array, chords_array=chords_array, bpm_array=bpm_array, section=section)

  print('-----------------')

print('song_count:', song_count)

midi_simple, chords_array = chords_process(chords_file, bpm, key)
print(chords_file)
chords_array = chords_array
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(chords_array.T, interpolation='nearest', aspect='auto', origin='lower')
plt.yticks(np.arange(len(c_major_chords)), c_major_chords);

midi_simple, vocals_array = vocals_process(vocals_dict, bpm)

print(len(vocals_array), 'possible transpose(s) found')
print(midi_simple)

import matplotlib
dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array[0].T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(vocals_array[0].T, interpolation='nearest', aspect='auto', origin='lower')
  # plt.xlim(0,900)
  # plt.ylim(0,30)
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);
  # plt.imshow(vocals_array.T, interpolation='nearest', origin='lower')

!ls /content/drive/MyDrive/music_app/midi_ML_data/Train_Cmajor/ | wc -l

len(glob.glob('/content/drive/MyDrive/music_app/midi_ML_data/Train/*'))

data_loc = '../music_app/midi_ML_data/Train_Cmajor/Shawn Mendes_I know what you did last summer_chorus_0.npz'
data = np.load(data_loc)
vocals_array=data['vocals_array']
# vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
# vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
# vocals_array2[:,:-1] = vocals_array

dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array.T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);

chords_array = data['chords_array']
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(chords_array.T, interpolation='nearest', aspect='auto', origin='lower')
plt.yticks(np.arange(len(c_major_chords)), c_major_chords);
