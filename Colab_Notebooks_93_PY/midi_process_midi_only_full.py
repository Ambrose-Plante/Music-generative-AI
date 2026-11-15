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
%cd /content/drive/My\ Drive/

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

all_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

cmajor_small_ref=[]
for num in np.arange(1,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in all_notes]
# cmajor_small_ref = ['G1', 'G#1', 'A1', 'A#1', 'B1'] + cmajor_small_ref
# cmajor_small_ref = cmajor_small_ref + ['C4']

cmajor_big_ref=[]
for num in np.arange(0,10):
  cmajor_big_ref = cmajor_big_ref + [x+str(num) for x in all_notes]

def remove_end_space(string):
  if len(string)<1:
    return string
  else:
    while string[-1]==' ':
      string = string[:-1]
    return string

def load_midi(midi_file, new_bpm):
    old_bpm = round(pretty_midi.PrettyMIDI(midi_file).get_tempo_changes()[1][0])
    temp_midi_file = pretty_midi.PrettyMIDI(midi_file, old_bpm)

    for instrument in temp_midi_file.instruments:
      for note in instrument.notes:
        note.start = note.start*(old_bpm/new_bpm)
        note.end = note.end*(old_bpm/new_bpm)

    return temp_midi_file

def midi_to_list(midi_data, bpm):

    midi_list = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    c2_midi_note=note_name_to_number("C2")
    note_range=note_name_to_number("C7")-note_name_to_number("C2")

    midi_simple = [(round(x[0]*sixth_step_per_sec), round(x[1]*sixth_step_per_sec)-1, x[2]-c2_midi_note) for x in midi_list]
    return midi_simple, note_range

triad_dic = {}
triad_dic['major_root'] = [0,4,7]
triad_dic['major_first'] = [0,3,8]
triad_dic['major_second']= [0,5,9]

triad_dic['minor_root'] = [0,3,7]
triad_dic['minor_first'] = [0,4,9]
triad_dic['minor_second'] = [0,5,8]

triad_dic['dim_root'] = [0,3,6]
triad_dic['dim_first'] = [0,3,9]
triad_dic['dim_second'] = [0,6,9]

def get_triad(chord_pitches):
    test = np.asarray(chord_pitches)-chord_pitches[0]

    answer = []
    for key, value in triad_dic.items():
        if np.array_equal(test, value):
            answer.append(key)
    if len(answer)!=1:
      raise Exception('triad not in dictionary')

    if 'major' in answer[0]:
      quality = 0
    elif 'minor' in answer[0]:
      quality = 1
    elif 'dim' in answer[0]:
      quality = 2

    if 'root' in answer[0]:
      position = 0
    elif 'first' in answer[0]:
      position = 1
    elif 'second' in answer[0]:
      position = 2

    return quality, position

def chords_process(chords_file, bpm, key):

    chords_list = []
    midi_data = load_midi(chords_file, actual_bpm)
    instrument = midi_data.instruments[0]
    for chord_start_time in np.unique([note.start for note in instrument.notes]):
      chord_pitches = sorted([note.pitch for note in instrument.notes if note.start==chord_start_time])
      chord_end_time = [note.end for note in instrument.notes if note.start==chord_start_time][0]
      quality_ind, position_ind = get_triad(chord_pitches)

      chord_shifted = note_number_to_name(chord_pitches[position_ind]-key_shift(key))[:-1], quality_ind
      all_note_ind = [i for i,y in enumerate(all_notes) if y == chord_shifted[0]]

      if len(all_note_ind)==1:
        chords_list.append([chord_start_time, chord_end_time, all_note_ind[0], quality_ind])
      else:
        raise Exception('chord not in scale')

    chords_list = sorted(chords_list, key=lambda z: (z[0], z[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    midi_simple = [(round(z[0]*sixth_step_per_sec), round(z[1]*sixth_step_per_sec), z[2], z[3]) for z in chords_list]
    max_steps = 17*4*4*6
    num_scale_degrees = len(all_notes)

    chords_array = np.zeros((max_steps, num_scale_degrees, 3)) # 3 is for major/minor/diminished
    for z in midi_simple:
        chords_array[z[0]:z[1],z[2],z[3]]=1

    return midi_simple, chords_array

def bpm_to_array(bpm):
    max_bpm=200+1
    bpm_array=np.zeros(max_bpm)
    bpm_array[bpm]=1
    return bpm_array

def vocals_process(vocal_dict, bpm):

    vocal_notes = [list(y) for y in midi_to_list(load_midi(vox_file, actual_bpm), actual_bpm)[0]]

    for note in vocal_notes:
      note[2] = note_number_to_name(note[2]-key_shift(key)+12) ## shift to C major

    max_steps = 17*4*4*6
    num_scale_degrees = len(cmajor_small_ref)

    vocals_array = []
    for shift in np.arange(-7,7): # try all possible octave shifts
      temp_midi = [[x[0], x[1], x[2][:-1]+str(int(x[2][-1])+shift)] for x in vocal_notes]
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

    if len(vocals_array)==0:

      temp2_midi=[]
      for x in vocal_notes: ##convert to degrees in full scale
        arg=[i for i,z in enumerate(cmajor_big_ref) if z==x[2]]
        if len(arg)==1:
          temp2_midi.append((x[0], x[1], arg[0]))
        else:
          print(x)
          raise Exception('note not in cmajor_big_ref')

      temp_array = np.zeros((max_steps, len(cmajor_big_ref)))
      for num,x in enumerate(temp2_midi):
        temp_array[x[0]:(x[1]),x[2]]=1
      plt.imshow(temp_array.T, aspect='auto', origin='lower')
      plt.yticks(np.arange(len(cmajor_big_ref)), cmajor_big_ref)
      raise Exception('no suitable transpose found')

    return vocal_notes, vocals_array

row_start = 127 # the actual row number on the spreadsheet
row_end = 422   # the actual row number on the spreadsheet

dir_list = glob.glob('pdf_to_midi/checked_midi/**/*')
special_characters = ['&', '!', '\'']

chord_list = []
note_list = []
vocal_data_dict = {}
bpm_dict = {}
song_count=0
for row,x in enumerate(song_list):

  if row not in np.arange(row_start-1,row_end):
    continue

  artist = remove_end_space(x[0])
  song = remove_end_space(x[1]).replace('/', '-')


  if x[6]=='midi_only':
    actual_bpm = int(x[2])
    # print(artist, song, 'midi_only')
  else:
    continue

  potential_dir = [x for x in dir_list if artist in x.split('/')[-1] and song in x.split('/')[-1]]

  if len(potential_dir)==0: # try again with replacing special characters
    for char in special_characters:
      artist = artist.replace(char, '_')
      song = song.replace(char, '_')
    potential_dir = [x for x in dir_list if artist in x.split('/')[-1] and song in x.split('/')[-1]]

  if len(potential_dir)!=1:
    raise Exception("either no dir or multiple matches")

  dir_name = potential_dir[0]

  key = [remove_end_space(x[3]),remove_end_space(x[4])]
  song_count += 1
  print('-----------------')
  print(dir_name, actual_bpm)

  if not os.path.exists(dir_name + '/ch_chorus.mid'):
    raise Exception('missing chorus')

  for section in ["verse_1", "pre_chorus", "chorus", "post_chorus", "verse_2"]:
      vox_file = dir_name + '/' + section + '.mid'
      chords_file = dir_name + '/ch_' + section + '.mid'

      if not os.path.exists(chords_file):
        continue

      midi_simple, vocals_array = vocals_process(vox_file, actual_bpm)
      midi_simple, chords_array = chords_process(chords_file, actual_bpm, key)
      bpm_array = bpm_to_array(actual_bpm)

      for q,array in enumerate(vocals_array):
        outfile = 'music_app/midi_ML_data/Train4/' + artist + '_' + song + '_' + section + '_' + str(q) + '.npz'
        print(outfile)
        np.savez(outfile, vocals_array=array, chords_array=chords_array, bpm_array=bpm_array, section=section)

  print('-----------------')



print('song_count:', song_count)
# for name_key in vocal_data_dict.keys():
#   for note in vocal_data_dict[name_key]:
#     note['duration'] += 1 ## I subtracted 1 when saving the json during data collection

vocal_notes = [list(y) for y in midi_to_list(load_midi(vox_file, actual_bpm), actual_bpm)[0]]

for note in vocal_notes:
  note[2] = note_number_to_name(note[2]-key_shift(key)+12) ## shift to C major

print(note_number_to_name(min([note_name_to_number(x[2]) for x in vocal_notes])), note_number_to_name(max([note_name_to_number(x[2]) for x in vocal_notes])))

cmajor_small_ref

data_loc = 'music_app/midi_ML_data/Train3/Red Hot Chili Peppers_Californication_chorus_0.npz'
data = np.load(data_loc)
vocals_array=data['vocals_array']
vocals_array2 = np.zeros((vocals_array.shape[0], vocals_array.shape[1]+1))
vocals_array2[:,-1] = (1-vocals_array.sum(axis=1))
vocals_array2[:,:-1] = vocals_array

dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array2.T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(vocals_array2.T, interpolation='nearest', aspect='auto', origin='lower')
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);

def array_to_midi(vocals_array):

    vocals_array = np.asarray([[float(x) for x in q] for q in vocals_array])
    vocals_array = vocals_array.argmax(axis=1)

    silence = len(cmajor_small_ref)+1
    for i,x in enumerate(vocals_array):
      if x!=silence:
        note_start=i
        break

    # print(note_start)

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

    return midi_list

note_change_fraction = 0.1
midi_list = array_to_midi(vocals_array2)
num_notes_to_change = round(len(midi_list)*note_change_fraction)
np.random.choice(np.arange(len(midi_list)), num_notes_to_change, replace=False)

np.load('music_app/midi_ML_data/Train3/Red Hot Chili Peppers_Californication_chorus_0.npz')['bpm_array'].argmax()

midi_simple, chords_array = chords_process(chords_file, actual_bpm, key)
chords_array = chords_array.sum(axis=2)
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(chords_array.T, interpolation='nearest', aspect='auto', origin='lower')
plt.yticks(np.arange(len(all_notes)), all_notes);

midi_simple, vocals_array = vocals_process(vox_file, actual_bpm)

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

!ls /content/drive/MyDrive/music_app/midi_ML_data/Train | wc -l

len(glob.glob('/content/drive/MyDrive/music_app/midi_ML_data/Train/*'))
