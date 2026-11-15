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
import time
plt.ioff()

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

cmajor_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'E_minor', 'F_major', 'G_major', 'A_minor', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

def array_to_list(music_array):

    arr_shape = music_array.shape
    vocals_array = np.zeros((arr_shape[0], arr_shape[1]+1))
    vocals_array[:,:-1] = music_array
    for i,x in enumerate(vocals_array.max(axis=1)):
      if x<0.2:
        vocals_array[i,:-1] = 0
        vocals_array[i,-1] = 1

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
    midi_list = [x for x in midi_list if x[2]!=17]

    return midi_list

def write_metronome_midi(new_bpm, output_path, vox_end_time, acc_end_time):
    # Create a PrettyMIDI object
    cello_c_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    # Iterate over note names, which will be converted to note number later
    for beat in range(16*4):
        if ((beat+0.5)/(new_bpm*60))>max(vox_end_time, acc_end_time): # stop at end of chords or vox
          break

        # Retrieve the MIDI note number for this note name
        note_number = pretty_midi.note_name_to_number('C5')
        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(
            velocity=100, pitch=note_number, start=beat/(new_bpm/60), end=(beat+0.5)/(new_bpm*60))
        # Add it to our cello instrument
        cello.notes.append(note)
    # Add the cello instrument to the PrettyMIDI object
    cello_c_chord.instruments.append(cello)
    # Write out the MIDI data
    cello_c_chord.write(output_path)

def write_chords_midi(chords_array, bpm, output_path):
    chords_list = [x for x in array_to_list(chords_array) if x[2]!=7]
    chords_list = [[c_major_chords[x[2]],x[0],x[1]] for x in chords_list]

    # Create a PrettyMIDI object
    chord_midi_object = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    # Create an Instrument instance for a cello instrument
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    for chord in chords_list:
        fifth = 7
        if chord[0][2:]=='minor':
          third = 3
        elif chord[0][2:]=='major':
          third = 4
        else:
          raise Exception('chord went wrong')

        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3'), start=chord[1]/sixth_step_per_sec, end=chord[2]/sixth_step_per_sec)
        cello.notes.append(note)
        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3')+third, start=chord[1]/sixth_step_per_sec, end=chord[2]/sixth_step_per_sec)
        cello.notes.append(note)
        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3')+fifth, start=chord[1]/sixth_step_per_sec, end=chord[2]/sixth_step_per_sec)
        cello.notes.append(note)
        # Add it to our cello instrument

    # Add the cello instrument to the PrettyMIDI object
    chord_midi_object.instruments.append(cello)

    # write it
    chord_midi_object.write(output_path)
    ####################################################################

DIR_DIR = 'midi_ML_data/dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_C/'

input_filename = DIR_DIR + 'outputs/validation_output0400.npz'

OUTPUT_DIR = DIR_DIR + 'audio_outputs/'

vocal_type='GENERATED' ## REAL or GENERATED
ACC_TYPE='REAL' ## REAL or CHORDS

#################################################

threshold = 0.5
data = np.load(input_filename)

for val_ind in range(data['fake'].shape[0]):

    section_ind = data['val_section'][val_ind].argmax()
    section = ['verse1', 'pre_chorus', 'chorus', 'post_chorus', 'verse2'][section_ind]
    if 'verse_1' in data['names'][val_ind]:
      section = 'verse_1'
    if 'verse_2' in data['names'][val_ind]:
      section = 'verse_2'
    song_name = data['names'][val_ind].split('/')[-1].split(section)[0].split('_')[-2]
    artist_name = data['names'][val_ind].split('/')[-1].split('_')[0]
    number = data['names'][val_ind].split('_')[-1].split('.npz')[0]

    artist = ''
    song = ''
    for row in song_list:
      if artist_name in row[0] and song_name in row[1]:
        print(row)
        key=[row[3], row[4]]
        artist=row[0]
        song=row[1]
        bpm=int(row[2])

        if row[6]=='midi_only':
          pass
        elif row[5]=='AP':
          BASE_DIR = '/content/drive/MyDrive/AP_AP_corrected2/'
        elif row[5]=='JF':
          BASE_DIR = '/content/drive/MyDrive/AP_Josh_corrected/'
        else:  ## not completed by AP or JF
          raise Exception('something wrong')
    if len(artist)==0 or len(song)==0:
      raise Exception('name not found')

    if section=='pre_chorus':
      end_name = 'prechorus'
    elif section=='post_chorus':
      end_name = 'post'
    else:
      end_name = section

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    if ACC_TYPE=='CHORDS':
        chords_midi_path = OUTPUT_DIR + artist + '_' + song + '_' + section + '_chords.mid'
        acc_file = OUTPUT_DIR + artist + '_' + song + '_' + section + '_chords.wav'
        write_chords_midi(data['val_chords'][val_ind], bpm, chords_midi_path)
        !fluidsynth -ni SalC5Light2.sf2 "$chords_midi_path" -F "$acc_file" -r 44100
    elif ACC_TYPE=='REAL':
        acc_file = BASE_DIR + 'midi_data/' + artist + '_' + song + '/' + end_name + '_acc.wav'
    else:
      raise Exception('invalid ACC type')


    if vocal_type=='REAL':
      vocal_midi_list = array_to_list(np.squeeze(data['val_vocals'][val_ind]))
    elif vocal_type=='GENERATED':
      vocal_midi_list = array_to_list(np.squeeze(data['fake'][val_ind]))
    else:
      raise Exception('invalid vocal type')

    octave_shift=2
    velocity = 100

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello')
    pm.instruments.append(inst)

    for start,end,pitch in [[x[0],x[1],x[2]] for x in vocal_midi_list]:
        if pitch!=len(cmajor_small_ref):
          pitch = cmajor_small_ref[pitch][:-1]+str(int(cmajor_small_ref[pitch][-1])+octave_shift)
          if ACC_TYPE=='CHORDS': ## dont shift key
            inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(pitch), start/sixth_step_per_sec, end/sixth_step_per_sec))
          else: # do shift key for real instrumental
            inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(pitch)+key_shift(key), start/sixth_step_per_sec, end/sixth_step_per_sec))
    print(inst.notes)

    pm.write('out.mid')
    time.sleep(1)

    !fluidsynth -ni SalC5Light2.sf2 out.mid -F output.wav -r 44100

    time.sleep(1)

    if ACC_TYPE=='REAL':
      sound1 = AudioSegment.from_file("output.wav")+10
      sound2 = AudioSegment.from_file(acc_file)-10
    elif ACC_TYPE=='CHORDS':
      sound1 = AudioSegment.from_file("output.wav")
      sound2 = AudioSegment.from_file(acc_file)

    metronome_midi_path = OUTPUT_DIR + artist + '_' + song + '_' + section + '_metronome.mid'
    metronome_wav_path = OUTPUT_DIR + artist + '_' + song + '_' + section + '_metronome.wav'
    write_metronome_midi(bpm, metronome_midi_path, vocal_midi_list[-1][1]/sixth_step_per_sec, sound2.duration_seconds)
    !fluidsynth -ni PVCDrum.sf2 "$metronome_midi_path" -F "$metronome_wav_path" -r 44100
    metronome = AudioSegment.from_file(metronome_wav_path)

    combined = sound1.overlay(sound2)
    combined = combined.overlay(metronome)

    combined.export(OUTPUT_DIR + artist_name + '_' +song_name + '_' + section + '_' + number + '.mp3', format='mp3')

    os.remove(metronome_midi_path)
    os.remove(metronome_wav_path)

    if ACC_TYPE=='CHORDS':
      os.remove(chords_midi_path)
      os.remove(acc_file)

    # Create the first subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)

    # First subplot
    ax1.set_yticks(np.arange(len(c_major_chords)))
    ax1.set_yticklabels(c_major_chords)
    ax1.imshow(data['val_chords'][val_ind].T, interpolation='nearest', aspect='auto', origin='lower')

    # Second subplot
    ax2.set_yticks(np.arange(len(cmajor_small_ref)))
    ax2.set_yticklabels(cmajor_small_ref)
    ax2.imshow(data['val_vocals'][val_ind].T, interpolation='nearest', aspect='auto', origin='lower')

    plt.savefig(DIR_DIR + 'val_figures/' + artist_name + '_' +song_name + '_' + section + '_' + number + '.png')

    # from IPython.display import Audio
    # Audio('combined.wav')

data['val_chords'][val_ind].shape
