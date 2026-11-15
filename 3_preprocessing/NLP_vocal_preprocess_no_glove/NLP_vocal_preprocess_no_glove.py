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

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pickle
import pickle5
import json
import glob

import pretty_midi
from pretty_midi.utilities import note_number_to_name, note_name_to_number

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

BOS_vocals = np.zeros((1632,19))
BOS_vocals[-200:-100,:] = 1

EOS_vocals = np.zeros((1632,19))
EOS_vocals[-100:,:] = 1

BOS_syll = np.zeros((6,40))
BOS_syll[-1,:20] = 1

EOS_syll = np.zeros((6,40))
EOS_syll[-1,20:] = 1

dash_syll = np.zeros((6,40))
dash_syll[-2,:] = 1

# plt.figure(figsize=(8, 6), dpi=100)
# plt.imshow(dash_syll.T, interpolation='nearest', aspect='auto', origin='lower')

# syll_lib = ['BOS', 'EOS']

# print(len(syll_lib))

# for q in range(len(complete_list)):
#   artist = complete_list[q][0]
#   song = complete_list[q][1]
#   bpm = int(complete_list[q][2])
#   key = complete_list[q][3]
#   key_quality = complete_list[q][4]
#   # print(artist, song, bpm, key, key_quality)
#   for filename in glob.glob('midi_data/' + artist + '_' + song + '/*.json'):

#     with open(filename) as f:
#       dic = json.load(f)

#     for note in dic:
#       if not [x for x in syll_lib if np.array_equal(x, note['syllable'])]:
#         syll_lib.append(note['syllable'])

#     print(len(syll_lib), filename)

# with open('syll_lib.pkl', 'wb') as f:
#   pickle.dump(syll_lib, f)

with open('syll_lib.pkl', 'rb') as f:
  syll_lib = pickle.load(f)

vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
              , "V", "W", "Y", "Z", "ZH"]

phoneme_ref = vowels + consonants + ['-', 'BOS', 'EOS']

def process_lyrics(vocals_json):

    with open(vocals_json) as f:
        dic = json.load(f)

    max_notes=200
    max_phonemes_per_syllable=6
    num_unique_phonemes=len(phoneme_ref)

    output_syll_vec = np.zeros((max_notes, max_phonemes_per_syllable, num_unique_phonemes))
    indices=[]

    q = [q for q,temp in enumerate(phoneme_ref) if 'BOS'==temp][0]
    output_syll_vec[0,0,q]=1 # make first note 'BOS'
    indices.append([z for z,y in enumerate(syll_lib) if y=='BOS'][0]) # make first index 'BOS' token
 
    for i,x in enumerate(dic):
      i+=1 # shift notes by one

      indx = [z for z,y in enumerate(syll_lib) if y==x['syllable']]
      if len(indx)!=1:
        raise Exception('Did not find exactly 1 index')
      indices.append(indx[0])

      for j,phoneme in enumerate(x['syllable'].split()):
          q = [q for q,temp in enumerate(phoneme_ref) if phoneme==temp]
          if len(q)==1:
              output_syll_vec[i,j,q] = 1
          else:
            raise Exception('not exactly one phoneme match')

    i+=1
    q = [q for q,temp in enumerate(phoneme_ref) if 'EOS'==temp][0]
    output_syll_vec[i,0,q]=1 # make note after last note'EOS'
    indices.append([z for z,y in enumerate(syll_lib) if y=='EOS'][0]) # make last index 'EOS' token

    pad_ind = [z for z,y in enumerate(syll_lib) if y=='PAD'][0]
    while len(indices)<max_notes:
      indices.append(pad_ind)
      
    return output_syll_vec, indices
        

cmajor_scale = ['C', 'D', 'D#', 'E', 'F', 'G', 'A', 'B']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref
full_vocal_ref = cmajor_small_ref + ['BOS', 'EOS']

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
    num_scale_degrees = len(cmajor_small_ref)+2 # last two will be BOS and EOS

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
        temp_array[0,:,-2]=1 # BOS
        for num,x in enumerate(temp2_midi):
            num+=1
            if x[0]>=1632:
              raise Exception('note start after 16 bars')
            temp_array[num,x[0]:(x[1]),x[2]]=1
        vocals_array.append(temp_array)

        num+=1
        temp_array[num,:,-1]=1 #EOS

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

# q=20
# artist_name = complete_list[q][0]
# song_name = complete_list[q][1]
# section = 'verse1'
# print(artist_name, song_name, section)
# vocals_midi = 'midi_data/' + artist_name + '_' + song_name + '/verse1.mid'
# vocals_json = 'midi_data/' + artist_name + '_' + song_name + '/verse1.json'
# bpm = int(complete_list[q][2])
# key = complete_list[q][3]
# key_quality = complete_list[q][4]

# output_syll_vec, indices = process_lyrics(vocals_json)
# midi_simple, vocals_array = vocals_process(vocals_midi, bpm, key, key_quality)
# for i in range(len(indices)):
#   note = output_syll_vec[i]
#   voc_note = vocals_array[0][i]
#   if (note.sum(axis=1)>1).sum()>0:
#     print(i)
#     raise Exception('two phonemes')
#   syll = []
#   for ind in note[np.where(note.sum(axis=1)>0)].argmax(axis=1):
#     syll.append(phoneme_ref[ind])

#   if voc_note.sum()==0:
#     voc='PAD'
#   else:
#     voc = full_vocal_ref[voc_note.sum(axis=0).argmax()]

#   print(' '.join(syll), ' | ', syll_lib[indices[i]], ' | ', voc)

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

for q in range(len(complete_list)):
  artist = complete_list[q][0]
  song = complete_list[q][1]
  bpm = int(complete_list[q][2])
  key = complete_list[q][3]
  key_quality = complete_list[q][4]
  print(artist, song, bpm, key, key_quality)

  lyrics_file = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
  lyrics_dictionary = pickle5.load(lyrics_file)

  for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:
      if len(lyrics_dictionary[x])>2:
          if x=='verse1':
              vocals_file = 'midi_data/' + artist + '_' + song + '/verse1.mid'
              vocals_json = 'midi_data/' + artist + '_' + song + '/verse1.json'

          elif x=='pre_chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/prechorus.mid'
              vocals_json = 'midi_data/' + artist + '_' + song + '/prechorus.json'

          elif x=='chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/chorus.mid'
              vocals_json = 'midi_data/' + artist + '_' + song + '/chorus.json'

          elif x=='post_chorus':
              vocals_file = 'midi_data/' + artist + '_' + song + '/post_chorus.mid'
              vocals_json = 'midi_data/' + artist + '_' + song + '/post_chorus.json'

          elif x=='verse2':
              vocals_file = 'midi_data/' + artist + '_' + song + '/verse2.mid'
              vocals_json = 'midi_data/' + artist + '_' + song + '/verse2.json'

          midi_list, vocals_array = vocals_process(vocals_file, bpm, key, key_quality)
          syll_vec, indices = process_lyrics(vocals_json)
          indices = np.asarray(indices)
          for num,array in enumerate(vocals_array):

            ####make sure all notes are aligned#######
            arr1 = np.asarray([syll_lib[i] for i in indices])!='PAD'
            arr2 = syll_vec.reshape(200,-1).sum(axis=1)>0
            arr3 = vocals_array[0].sum(axis=1).sum(axis=1)>0
            if not (np.array_equal(arr1, arr2) and np.array_equal(arr1, arr3)):
              raise Exception('notes misaligned')
            ###############################################

            outfile = artist + '_' + song + '_' + x + '_' + str(num) + '.npz'
            print(outfile)
            np.savez_compressed('/content/temp/temp.npz', vocals_array=array, indices=indices, syll_vec=syll_vec)
            upload_file(outfile, '/content/temp/temp.npz', '12AVulFeTFvzVSFZjZUhFUtcoTZUcEUiB')
            !rm /content/temp/temp.npz
