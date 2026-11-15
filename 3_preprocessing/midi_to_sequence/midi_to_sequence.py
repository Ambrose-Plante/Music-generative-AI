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

def is_complete(artist, song):
    if os.path.exists("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl"):
        if os.path.exists("saved_pronunciations/"+artist+"_"+song.replace('/', '-')+".pkl"):

            file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
            loaded_dictionary = pickle5.load(file_to_read)

            complete=True
            rdy=True
            for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:
                if len(loaded_dictionary[x])>2:
                    if x=='verse1':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_verse1.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='pre_chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_prechorus.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_chorus.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='post_chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_chorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_chorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_post.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='verse2':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_verse2.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

            if complete==True:
                return (artist, song)

complete_list=[]
with open('song_list.csv', newline='', mode='r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if is_complete(row[0],row[1]):
            complete_list.append(row)

cmajor_chords = ['Cmajor', 'Dminor', 'Eminor', 'Emajor', 'Fmajor', 'Gmajor', 'Aminor', 'Bminor']

# import warnings
# warnings.filterwarnings("error")

cmajor_scale = ['C', 'D', 'D#', 'E', 'F', 'G', 'A', 'B']

def chords_process(filename, bpm, key, key_quality):
    midi_data = pretty_midi.PrettyMIDI(filename)

    if key_quality=='major':
      base_key = note_name_to_number("C0")
    elif key_quality=='minor':
      base_key = note_name_to_number("A0")
    else:
      raise Exception('key naming issue')

    key_shift = base_key-note_name_to_number(key+'0')

    midi_list = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch + key_shift
            velocity = note.velocity
            if velocity<50:
                chord = note_number_to_name(pitch)[:-1]+'minor'
            else:
                chord = note_number_to_name(pitch)[:-1]+'major'

            arg = [i for i,x in enumerate(cmajor_chords) if chord==x]
            if len(arg)==1:
              midi_list.append([start, end, arg[0]])
            else:
              raise Exception('chord not in scale', chord)
              
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    midi_simple = [(round(x[0]*sixth_step_per_sec), round(x[1]*sixth_step_per_sec), x[2]) for x in midi_list]
    
    max_steps = 17*4*4*6
    num_scale_degrees = len(cmajor_chords)

    chords_array = np.zeros((50, max_steps, num_scale_degrees))
    for z,x in enumerate(midi_simple):
        chords_array[z,x[0]:x[1],x[2]]=1

    return midi_simple, chords_array

def array3d_to_2d(array3d):
  array2d = np.zeros((array3d.shape[1], array3d.shape[2]))

  for note in array3d:
    array2d+=note

  return array2d

q=3
artist = complete_list[q][0]
song = complete_list[q][1]
bpm = int(complete_list[q][2])
key = complete_list[q][3]
key_quality = complete_list[q][4]
filename = 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'
print(filename)
midi_simple, chords_array = chords_process(filename, bpm, key, key_quality)

plt.figure(figsize=(8, 6), dpi=100)

plt.imshow(array3d_to_2d(chords_array).T, interpolation='nearest', aspect='auto', origin='lower')

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

cmajor_big_ref=[]
for num in np.arange(2,7):
  cmajor_big_ref = cmajor_big_ref + [x+str(num) for x in cmajor_scale]

def vocals_process(filename, bpm, key, key_quality):
    midi_data = pretty_midi.PrettyMIDI(filename)

    if key_quality=='major':
      if key in ['G', 'G#', 'A', 'A#', 'B']:
        key_shift=note_name_to_number('C1')-note_name_to_number(key+'0')
      elif key in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#']:
        key_shift=note_name_to_number('C0')-note_name_to_number(key+'0')
      else:
        raise Exception('key shift issue', key)
    elif key_quality=='minor':
      if key in ['A#', 'B']:
        key_shift=note_name_to_number('A0')-note_name_to_number(key+'0')
      elif key in ['C', 'C#', 'D', 'D#']:
        key_shift=note_name_to_number('A0')-note_name_to_number(key+'1')
      elif key in ['A', 'E', 'F', 'F#', 'G', 'G#']:
        key_shift=note_name_to_number('A1')-note_name_to_number(key+'1')
      else:
        raise Exception('key shift issue', key)
    else:
      raise Exception('key naming issue')

    midi_list = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch + key_shift
            velocity = note.velocity
            v_note = note_number_to_name(pitch)

            if v_note[:-1] in cmajor_scale:
              midi_list.append([start, end, v_note])
            else:
              print(filename)
              raise Exception('note not in scale', v_note)

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    midi_simple = [(round(x[0]*sixth_step_per_sec), round(x[1]*sixth_step_per_sec), x[2]) for x in midi_list]

    max_steps = 17*4*4*6
    num_scale_degrees = len(cmajor_small_ref)

    vocals_array = []
    for shift in np.arange(-4,5): # try all possible octave shifts
      temp_midi = [[x[0], x[1], x[2][:-1]+str(int(x[2][-1])+shift)] for x in midi_simple]
      # if shift==-1:
      if len([x[2] for x in temp_midi if x[2] not in cmajor_small_ref])==0: # if shift fits into small array

        temp2_midi=[]
        for x in temp_midi: ##convert to degrees in cmajor small ref
          arg=[i for i,z in enumerate(cmajor_small_ref) if z==x[2]]
          if len(arg)==1:
            temp2_midi.append((x[0], x[1], arg[0]))
          else:
            raise Exception('note not in cmajor_small_ref')

        temp_array = np.zeros((200, max_steps, num_scale_degrees))
        for num,x in enumerate(temp2_midi):
            temp_array[num,x[0]:(x[1]),x[2]]=1
        vocals_array.append(temp_array)

    ### If no transpose found, print info for debug#####
    if len(vocals_array)==0: 
      print(filename)

      temp2_midi=[]
      for x in midi_simple: ##convert to degrees in full scale
        arg=[i for i,z in enumerate(cmajor_big_ref) if z==x[2]]
        if len(arg)==1:
          temp2_midi.append((x[0], x[1], arg[0]))
        else:
          raise Exception('note not in cmajor_big_ref')

      temp_array = np.zeros((max_steps, len(cmajor_big_ref)))
      for num,x in enumerate(temp2_midi):
        temp_array[x[0]:(x[1]),x[2]]=1
      plt.imshow(temp_array.T, aspect='auto', origin='lower', cmap=cmap)
      plt.yticks(np.arange(len(cmajor_big_ref)), cmajor_big_ref)
      raise Exception('no suitable transpose found')
      ################

    return midi_simple, vocals_array

plt.figure(figsize=(8, 6), dpi=100)
import matplotlib.colors as colors
cmap = colors.ListedColormap(['white', 'black'])
  
for q in range(len(complete_list)):
  if q!=11:
    continue
  artist = complete_list[q][0]
  song = complete_list[q][1]
  bpm = int(complete_list[q][2])
  key = complete_list[q][3]
  key_quality = complete_list[q][4]
  filename = 'midi_data/' + artist + '_' + song + '/chorus.mid'

  midi_simple, vocals_array = vocals_process(filename, bpm, key, key_quality)

  print(filename)
  print(len(vocals_array), 'possible transpose found')
  print(midi_simple)

  plt.imshow(array3d_to_2d(vocals_array[1]).T, aspect='auto', origin='lower', cmap=cmap)
  # plt.xlim(0,900)
  # plt.ylim(0,30)
  plt.yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref)
  # plt.imshow(vocals_array.T, interpolation='nearest', origin='lower')

def bpm_to_array(bpm):
    max_bpm=200
    min_bpm=50
    bpm_array=np.zeros(max_bpm-min_bpm)
    bpm_array[bpm-min_bpm]=1
    return bpm_array

def get_syllables(syll_file, section):
    if section=='prechorus':
        section='pre_chorus'
    stanza = pickle5.load( open( syll_file, "rb" ) )[section]
    stanza = stanza.replace('0', '')
    stanza = stanza.replace('1', '')
    stanza = stanza.replace('2', '')
    stanza = stanza.replace('3', '')
    stanza = stanza.replace('4', '')
    stanza = stanza.replace('5', '')

    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

    consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
                  , "V", "W", "Y", "Z", "ZH"]

    syllables = []

    for line in stanza.split('\r\n'):
        for word in line.split('  '):
    #         print(word)
            vowel_list=[]
            for phoneme in word.split(' '):
                if phoneme in vowels:
                    vowel_list.append(phoneme)
                elif phoneme not in consonants:
                    raise Exception("uh oh weird phoneme: ", phoneme, 'in line: ', stanza)
            if len(vowel_list)==1:
                syllables.append(word)
            else:
                temp_word=word
                for i in range(len(vowel_list)):
                    if i==(len(vowel_list)-2):
                        syllables.append(temp_word.split(vowel_list[i], 1)[0] + vowel_list[i])
                        syllables.append(temp_word.split(vowel_list[i] + " ", 1)[1])
                        break
    #                 print(vowel_list[i])
                    syllables.append(temp_word.split(vowel_list[i], 1)[0] + vowel_list[i])
                    temp_word = temp_word.split(vowel_list[i]+" ", 1)[1]
    return syllables

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def upload_file(filename, file_loc, dir_id):
  uploaded = drive.CreateFile({'title': filename, 'parents': [{'id': dir_id}]})
  uploaded.SetContentFile(file_loc)
  uploaded.Upload()
  # print('Uploaded file with ID {}'.format(uploaded.get('id')))

# ! rm -rf data_list
# ! mkdir data_list
# ! mkdir data_list/train

from google.colab import files

uploaded = files.upload() # upload clients_secret.json

!mv client_secret_529386604270-3eelg2ggsjf47hgbj6ugb5d1ebnccnbl.apps.googleusercontent.com.json client_secrets.json

for q in range(len(complete_list)):
  artist = complete_list[q][0]
  song = complete_list[q][1]
  bpm = int(complete_list[q][2])
  key = complete_list[q][3]
  key_quality = complete_list[q][4]
  print(artist, song, bpm, key, key_quality)
      
  file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
  loaded_dictionary = pickle5.load(file_to_read)

  for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:
      if len(loaded_dictionary[x])>2:
          if x=='verse1':
              vocals_file = 'midi_data/' + artist + '_' + song + '/verse1.mid'
              chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse1.mid'

          elif x=='pre_chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/prechorus.mid'
              chords_file = 'midi_data/' + artist + '_' + song + '/ch_prechorus.mid'

          elif x=='chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/chorus.mid'
              chords_file = 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'

          elif x=='post_chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/post_chorus.mid'
              chords_file = 'midi_data/' + artist + '_' + song + '/ch_post.mid'

          elif x=='verse2':
              vocals_file = 'midi_data/' + artist + '_' + song + '/verse2.mid'
              chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse2.mid'
        
          midi_list, vocals_array = vocals_process(vocals_file, bpm, key, key_quality)
          midi_list, chords_array = chords_process(chords_file, bpm, key, key_quality)
          bpm_array = bpm_to_array(bpm)
          for q,array in enumerate(vocals_array):
            outfile = artist + '_' + song + '_' + x + '_' + str(q) + '.npz'
            print(outfile)
            np.savez_compressed('/content/temp/temp.npz', vocals_array=array, chords_array=chords_array, bpm_array=bpm_array)
            upload_file(outfile, '/content/temp/temp.npz', '1-6BQCCwCXcuhah29WzEYUD6SrGMo5u7f')
            !rm /content/temp/temp.npz
                    

!ls /content/drive/MyDrive/music_app/data_list/train/*.npz | wc -l

for file in np.load('/content/drive/MyDrive/music_app/data_list/train/' + outfile).files:
  print(file, np.load('/content/drive/MyDrive/music_app/data_list/train/' + outfile)[file].shape)
