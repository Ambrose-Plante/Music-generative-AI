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
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pydub import AudioSegment
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
arpabet = cmudict.dict()
from functions.functions import *

max_steps = 17*4*4*6

def key_shift(key_list):
  key=key_list[0]
  key_quality = key_list[1]

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

  return -key_shift

vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
              , "V", "W", "Y", "Z", "ZH"]

def convert_to_sylls(stanza):

  stanza = convert_to_pron(lite_clean(stanza))[0]

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

  
  for word in stanza.split('  '):
  #         print(word)
      vowel_list=[]
      for phoneme in word.split(' '):
          if phoneme in vowels:
              vowel_list.append(phoneme)
          elif phoneme not in consonants:
              raise Exception("uh oh weird phoneme: " + phoneme + 'in line: ' + stanza)
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

with open('probabilities/markov_dict.pkl', 'rb') as filename:
  markov_probabilities = pickle5.load(filename)

with open('probabilities/pitch_prob_dict.pkl', 'rb') as filename:
  pitch_prob_dict = pickle5.load(filename)

# cleaned_lyrics = []
# for filename in glob.glob('lyrics_data/cleaned_lyrics/*'):
#   with open(filename, 'rb') as f:
#     cleaned_lyrics += pickle5.load(f)
# total_lines_num = len(cleaned_lyrics)

# cleaned_lyrics = []
# count = 0
# for filename in glob.glob('lyrics_data/cleaned_lyrics/*'):
#   with open(filename, 'rb') as f:
#     for line in pickle5.load(f):
#       cleaned_lyrics.append([line, convert_to_sylls(line)])
#       if count%1000==0:
#         print(count,'/', total_lines_num)
#       count+=1

# with open('lyrics_data/tot_cleaned_lyrics.pkl', 'wb') as saveeee_filename:
#   pickle5.dump(cleaned_lyrics, saveeee_filename)

with open('lyrics_data/tot_cleaned_lyrics.pkl', 'rb') as filename:
  tot_cleaned_lyrics = pickle5.load(filename)

# def get_vowel(sylls):
#   vowel = [x for x in sylls[-1].split(' ') if x in vowels]
#   if len(vowel)!=1:
#     raise Exception('not exactly 1 vowel')

#   return vowel
  
# lyrics1, sylls1 = tot_cleaned_lyrics[np.random.choice(np.arange(len(tot_cleaned_lyrics)))]
# current_vowel = get_vowel(sylls1)[0]

# test_vowel ='blah'
# while current_vowel!=test_vowel:
#   lyrics2, sylls2 = tot_cleaned_lyrics[np.random.choice(np.arange(len(tot_cleaned_lyrics)))]
#   test_vowel = get_vowel(sylls2)[0]


# lyrics3, sylls3 = tot_cleaned_lyrics[np.random.choice(np.arange(len(tot_cleaned_lyrics)))]
# current_vowel = get_vowel(sylls3)[0]

# test_vowel ='blah'
# while current_vowel!=test_vowel:
#   lyrics4, sylls4 = tot_cleaned_lyrics[np.random.choice(np.arange(len(tot_cleaned_lyrics)))]
#   test_vowel = get_vowel(sylls4)[0]

# print(lyrics1)
# print(lyrics2)
# print(lyrics3)
# print(lyrics4)

np.random.choice(np.arange(max(temp_probs['bpm_range'][0]-10, 50), min(temp_probs['bpm_range'][1]+10, 200)))

p=[]
for markov_key in markov_probabilities.keys():
  p.append(markov_probabilities[markov_key]['probability_bin'])

for iter in range(10):
  print(iter)

  # print('bpm:', bpm)

  markov_key = np.random.choice(list(markov_probabilities.keys()), p=p)
  temp_probs = markov_probabilities[markov_key] # fraction and syll bins are closed on bottom and open on top

  bpm = np.random.choice(np.arange(max(temp_probs['bpm_range'][0]-10, 60), min(temp_probs['bpm_range'][1]+10, 190)))
  bps=bpm/60
  steps_per_sec=bps*4
  sixth_step_per_sec=steps_per_sec*6

  # print('fraction_bins:', temp_probs['fraction_bins'])
  # print('syllable_bins:', temp_probs['syllable_bins'])

  target_num_sylls = np.random.choice(np.arange(temp_probs['syllable_bins'][0], temp_probs['syllable_bins'][1]))
  num_non_held_sylls = round(target_num_sylls*(1-temp_probs['held_syll_probability']))
  # print('target_num_sylls:', target_num_sylls, '  num_non_held_sylls:', num_non_held_sylls)

  temp_sylls = []
  temp_lyrics = []

  while len(temp_sylls)<num_non_held_sylls:
    lyrics, sylls = tot_cleaned_lyrics[np.random.choice(np.arange(len(tot_cleaned_lyrics)))]
    if (len(temp_sylls)+len(sylls))>num_non_held_sylls:
      lyrics_piece = ''
      for i,word in enumerate(lyrics.split(' ')):
        temp_sylls += convert_to_sylls(lyrics.split(' ')[i])
        if i==0:
          lyrics_piece += word
        else:
          lyrics_piece += ' ' + word
        if (len(temp_sylls)+len(convert_to_sylls(lyrics.split(' ')[i+1])))>num_non_held_sylls:
          break

      temp_lyrics.append(lyrics_piece)
      break

    temp_sylls += sylls
    temp_lyrics.append(lyrics)

  final_lyrics = ' '.join(temp_lyrics)


  probability_hold_note = temp_probs['held_syll_probability']

  mod_syllables=[]

  for i,x in enumerate(temp_sylls):
    if i==0:
      mod_syllables.append(x)
      continue

    while np.random.random_sample(1)[0]<probability_hold_note:
      mod_syllables.append('-')

    mod_syllables.append(x)
  # print('actual_num_sylls:', len(mod_syllables))

  midi_notes = []
  start=0 
  current_pitch = np.random.choice([int(x) for x in pitch_prob_dict.keys() if type(pitch_prob_dict[x])!=str])

  start += np.random.randint(low=0, high=97) #simulating pickup in first measure

  for i,x in enumerate(mod_syllables):
    if i>0:
      current_pitch = np.random.choice(np.arange(pitch_prob_dict[str(current_pitch)].shape[0]), p=pitch_prob_dict[str(current_pitch)])

    note_length=100000 ## initiate while loop
    while note_length>=1000: # keep rolling dice till you get a non-rest note for the syllable
      note_length=np.random.choice(np.arange(temp_probs['note_probabilities'].shape[0]), p=temp_probs['note_probabilities'])
      if note_length>=1000: # its a rest note
        if i<(len(mod_syllables)-1): # to stop index out of range error
          if mod_syllables[i+1]!='-':  # don't add rest when next note is a held note
            actual_note_duration = note_length - 1000
            start += actual_note_duration ## simulate rest
      else: # finally a non-rest note
        if x=='-': # x is a held note
          new_pitch=np.random.choice(np.arange(pitch_prob_dict[str(current_pitch)].shape[0]), p=pitch_prob_dict[str(current_pitch)])
          while current_pitch==new_pitch: ## held note pitch has to be different than previous note
            new_pitch=np.random.choice(np.arange(pitch_prob_dict[str(current_pitch)].shape[0]), p=pitch_prob_dict[str(current_pitch)])
          current_pitch = new_pitch
          end=start+note_length-1
          midi_notes.append([midi_notes[-1][1]-2,end,current_pitch,x]) ## held note overlaps 2 with previous note.
          start += note_length
        else: # x is not a held note
          end=start+note_length-1
          midi_notes.append([start,end,current_pitch,x])
          start += note_length

  max_steps = 17*4*4*6
  # print('actual_len_fraction:', midi_notes[-1][1]/max_steps)



  if midi_notes[-1][1]>max_steps:
    print('line too long. skipping...')
    continue

  accept = False
  while accept==False:
    key=[np.random.choice(['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#']), 'major'] ## must write in pitch shifter
    proposed_shift = key_shift(key)
    if max([x[2]+proposed_shift for x in midi_notes])<=note_name_to_number("C7") and min([x[2]+proposed_shift for x in midi_notes])>=note_name_to_number("C2"):
      # print(key)
      accept=True

  for note in midi_notes:
    note[2]+=proposed_shift


  velocity = 100
  pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
  inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello') 
  pm.instruments.append(inst)

  for start,end,pitch,syllable in midi_notes:
      inst.notes.append(pretty_midi.Note(velocity, pitch, start/sixth_step_per_sec, end/sixth_step_per_sec))

  midi_file_name = 'lyrics_data/simulated_data/plogue_midis/' + str(iter).zfill(10) + '_' + str(bpm) + '_' + key[0] +'_' + key[1] + '.mid'
  data_file_name = 'lyrics_data/simulated_data/lyrics_and_data/' + str(iter).zfill(10) + '_' + str(bpm) + '_' + key[0] +'_' + key[1] + '.pkl'

  pm.write(midi_file_name)

  for note in midi_notes:
    if note[3]=='-':
      note[0]+=3

  final_dict = {}
  final_dict['bpm'] = bpm
  final_dict['key'] = key
  final_dict['notes'] = midi_notes
  final_dict['lyrics'] = final_lyrics

  with open(data_file_name, 'wb') as saveeee_filename:
    pickle5.dump(final_dict, saveeee_filename)

for filename in glob.glob('lyrics_data/simulated_data/lyrics_and_data/*.pkl'):
  with open(filename, 'rb') as filename2:
    data_dict = pickle5.load(filename2)
    number = '4'
    if filename.split('_')[4].split('/')[-1]==number.zfill(10):
      print(data_dict['lyrics'])
      break

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default

creds, _ = default()
gc = gspread.authorize(creds)

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1YFVNyICk6GVUBu3o3bE-1JXqkZaaS0UD7DU4R-twYoo/edit#gid=0')

song_list = gsheets.worksheet('Sheet1').get_all_values()

max_steps = 17*4*4*6

def key_shift(key_list):
  key=key_list[0]
  key_quality = key_list[1]

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

  return -key_shift

# for filename in glob.glob('lyrics_data/augment_vocals/lyrics_and_data/*'):
#   os.remove(filename)

# for filename in glob.glob('lyrics_data/augment_vocals/plogue_midis/*'):
#   os.remove(filename)

all_data_dict = {}
vocal_data_dict = {}
bpm_dict = {}
song_count=0
augment_count = 0
section_count = 0
## FL studio defines C4 as 12 (in our json)
## lowest possible singing note is probably FL studio's C3 (defined as 0 in json)
## highest possible singing note is probably FL studio's C7 (defined as 48 in json)
lowest_pitch = 0
highest_pitch = 48
lowest_pitch += 24 ## to compensate for FL studio's piano roll being off an octave, and plogue apparently
highest_pitch += 24 ## to compensate for FL studio's piano roll being off an octave, and plogue

for x in song_list:
  artist = x[0]
  song = x[1].replace('/', '-')

  if x[5]=='AP':
    BASE_DIR = '/content/drive/MyDrive/AP_AP_corrected2/'
  elif x[5]=='JF':
    BASE_DIR = '/content/drive/MyDrive/AP_Josh_corrected/'
  else:  ## not completed by AP or JF
    continue

  if x[6]=='midi_only':
    print(artist, song, 'midi_only')
    continue

  key = [x[3],x[4]]
  song_count += 1
  print('-----------------')
  print(artist + '_' + song)
  syll_file = BASE_DIR + "saved_pronunciations/" + artist + "_" + song +".pkl"
  lyrics_file = BASE_DIR + "saved_lyrics/" + artist + "_" + song +".pkl"

  if not os.path.exists(syll_file):
    print('syllable file missing')

  if not os.path.exists(lyrics_file):
    print('lyrics file missing')

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

      section_count += 1

      if not os.path.exists(vocals_json):
        print('vocals_json missing for ', section)

      if not os.path.exists(chords_file):
        print('chords file missing for ', section)

      if not os.path.exists(acc_wav):
        print('instrumental wav missing for ', section)

      if not os.path.exists(vox_wav):
        print('vocals wav missing for ', section)

      with open(vocals_json, 'rb') as f:
        vocals_dict = json.load(f)

      for note in vocals_dict:
        note['pitch'] = note['pitch']-key_shift(key)+12 ## shift to C major

      original_bpm = int(x[2])

      midi_notes=[]
      for i,note in enumerate(vocals_dict):
        midi_notes.append([note['start']+0, note['start']+note['duration']+0, note['pitch']+0, note['syllable']+''])

      for i,note in enumerate(midi_notes):
        if note[3]=='-':
          note[0]=midi_notes[i-1][1]-2

      accept_count=0
      for q in range(1000):
        if accept_count>=100:
          break

        bpm_shift = np.random.choice(np.arange(-20,20))
        pitch_shift = np.random.choice(np.arange(-100, 100))
        bpm = original_bpm + bpm_shift
        pitch_accept = max([x[2]+pitch_shift for x in midi_notes])<=highest_pitch and min([x[2]+pitch_shift for x in midi_notes])>=lowest_pitch
        bpm_accept = bpm>60 and bpm<190
        if pitch_accept and bpm_accept:
          accept_count += 1
          augment_count += 1

          if pitch_shift<0:
            data_key = artist+'_'+song+'_'+section+'_' + str(bpm) + '_' + 'neg' + str(pitch_shift)[1:]
          else:
            data_key = artist+'_'+song+'_'+section+'_' + str(bpm) + '_' + str(pitch_shift)

          # bps=bpm/60
          # steps_per_sec=bps*4
          # sixth_step_per_sec=steps_per_sec*6

          # velocity = 100
          # pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
          # inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello') 
          # pm.instruments.append(inst)

          # for start,end,pitch,syllable in midi_notes:
          #     inst.notes.append(pretty_midi.Note(velocity, pitch+pitch_shift, start/sixth_step_per_sec, end/sixth_step_per_sec))

          # pm.write(midi_file_name)

          shifted_midi_notes=[]
          for i,note in enumerate(vocals_dict):
            shifted_midi_notes.append([note['start']+0, note['start']+note['duration']+0, note['pitch']+pitch_shift, note['syllable']+''])

          final_dict = {}
          final_dict['bpm'] = bpm
          final_dict['unshifted_midi_notes'] = midi_notes
          final_dict['pitch_shift'] = pitch_shift
          final_dict['key'] = [note_number_to_name(note_name_to_number("C4")+pitch_shift)[:-1], 'major']
          final_dict['shifted_vocal_notes'] = shifted_midi_notes
          final_dict['lyrics'] = ' '.join(lyrics[section].split('\r\n'))

          all_data_dict[data_key] = final_dict
  print('-----------------')


with open('all_augment_vocals_data2.pkl', 'wb') as saveeee_filename:
  pickle5.dump(all_data_dict, saveeee_filename)

os.path.exists('all_augment_vocals_data.pkl')

! ls lyrics_data/augment_vocals/lyrics_and_data 

len(glob.glob('lyrics_data/augment_vocals/lyrics_and_data/*'))

len(glob.glob('lyrics_data/augment_vocals/plogue_midis/*'))

(1/(50/60))*4*17
