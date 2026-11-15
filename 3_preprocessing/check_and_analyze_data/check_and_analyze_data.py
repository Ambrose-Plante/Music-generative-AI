from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

! pip install pickle5

pip install pretty_midi

import os
import pickle5
import glob
import json
from pretty_midi import note_number_to_name, note_name_to_number
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

# for filename in glob.glob('../AP_Josh_feldman_data_folder/midi_data/Katy Perry_Chained to the Rhythm/*.mid'):
#   section = filename.split('/')[-1]
#   print(section)
#   pm = pretty_midi.PrettyMIDI(filename)
#   for instrument in pm.instruments:
#     for i, note in enumerate(instrument.notes):
#       if i==0:
#         shift = note.start - (note.start/2)

#       note.start /= 2
#       note.start += shift
#       note.end /= 2
#       note.end += shift

#   pm.write('../chained_corrected/'+section)

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

cmajor_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'D_major', 'E_minor', 'E_major', 'F_major', 'F_minor', 'G_major', 'A_minor', 'A_major', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

cmajor_big_ref=[]
for num in np.arange(2,7):
  cmajor_big_ref = cmajor_big_ref + [x+str(num) for x in cmajor_scale]

vocal_data_dict = {}
bpm_dict = {}
song_count=0
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

      if sum([x['start']%2 for x in vocals_dict]):
        print(section, str(round(100*sum([x['start']%2 for x in vocals_dict])/len(vocals_dict)))+'% odd')

      for note in vocals_dict:
        note['pitch'] = note_name_to_number(note_number_to_name(note['pitch']-key_shift(key)+12)) ## shift to C major

      vocal_data_dict[artist+'_'+song+'_'+section] = vocals_dict

      bpm_dict[artist+'_'+song+'_'+section] = int(x[2])

      abberent_chords=[]
      midi_data = pretty_midi.PrettyMIDI(chords_file)
      for instrument in midi_data.instruments:
        for note in instrument.notes:
          if note.velocity>70:
            quality='major'
          else:
            quality='minor'
          chord = note_number_to_name(note.pitch)[:-1] + '_' + quality
          chord_shifted = note_number_to_name(note.pitch-key_shift(key))[:-1] + '_' + quality
          if chord_shifted not in c_major_chords:
            print(section, chord_shifted, chord)

      notes = [note_number_to_name(x['pitch'])[:-1] for x in vocals_dict]
      abberant_notes = []
      for note in notes:
        if note not in cmajor_scale:
          # raise Exception('abberant_note')
          abberant_notes.append(note)
      if len(abberant_notes)>0:
        print(section, abberant_notes)
  print('-----------------')

print('song_count:', song_count)
for name_key in vocal_data_dict.keys():
  for note in vocal_data_dict[name_key]:
    note['duration'] += 1 ## I subtracted 1 when saving the json during data collection

notes=[]
for key in vocal_data_dict.keys():
  for note in vocal_data_dict[key]:
    # if note['pitch']>40:
    #   print(key)
    #   break
    notes.append(note['pitch'])

def get_track_data(track):
  track_data = []
  start = 0

  if track[0]['start']!=0:
    duration = track[0]['start']
    # track_data.append([start, start+duration, duration, 'pickup_rest']) ## initial rest
    start += duration

  for i, note in enumerate(track):
    duration = note['duration']
    track_data.append([start, start+duration, duration, note['pitch']])
    start += duration

    if i<(len(track)-1): ## skip for the last note
      if start==track[i+1]['start']: ## there is no rest in between
        pass
      else:  ## there is a rest before the next note
        duration = track[i+1]['start']-start
        track_data.append([start, start+duration, duration, 'rest'])
        start += duration

  return track_data

## check for sections over 17 bars
for key in vocal_data_dict.keys():
  length_fraction = (vocal_data_dict[key][-1]['start'] + vocal_data_dict[key][-1]['duration'])/max_steps
  if length_fraction>1.0 or length_fraction<0:
    print(key, length_fraction)

# check for sections with very short notes
# not always pathologic
for key in vocal_data_dict.keys():
  for note in vocal_data_dict[key]:
    if note['duration']<=2:
      print(key)
      break

#check for overlapping notes
for key in vocal_data_dict.keys():
  notes = vocal_data_dict[key]
  for i,note in enumerate(notes):
    if i<(len(notes)-1):
      if (note['start']+note['duration'])>notes[i+1]['start']:
        print(key)
        break

temp=[]
for key in vocal_data_dict.keys():
  length_fraction = (vocal_data_dict[key][-1]['start'] + vocal_data_dict[key][-1]['duration'])/max_steps
  temp.append([len(vocal_data_dict[key]), length_fraction])

syllable_bins = np.append(np.arange(0,140,20)[:,None], np.arange(20,160,20)[:,None], axis=1)
fraction_bins = [[0,0.35], [0.35, 0.75], [0.75, 1.1]]

plt.scatter(np.asarray(temp)[:,0], np.asarray(temp)[:,1])
for x in syllable_bins:
  plt.vlines(x[0], 0 ,1.1, color='green')
  plt.vlines(x[1], 0 ,1.1, color='green')

for x in fraction_bins:
  plt.hlines(x[0], 0, 140, color='red')
  plt.hlines(x[1], 0, 140, color='red')
## for a given number of syllables, tracks are stratified by length fraction because it either:
## 1) has high probabilty of longer length for each syllable (high length fraction)
## 2) has low probability of longer length, and got cut off due to short chords length (low length fraction)

def get_probabilities(data):
  probabilities = np.zeros(1000)

  for duration in np.unique(data):
    if duration>1000:
      raise Exception('really long note')
      
    if duration<=0:
      raise Exception('weird duration ', duration)
    matches = np.squeeze(np.where(data==duration))
    if matches.ndim==0: ## exactly one match
      probability = 1/len(temp_data)
    else:
      probability = matches.shape[0]/len(temp_data)
    probabilities[duration] = probability

  return probabilities

markov_dict = {}
orphaned_keys = list(vocal_data_dict.keys()).copy()
for syll_bin in syllable_bins:
  for f_bin in fraction_bins:
    hit_count = 0
    held_syllable_count = 0
    total_syllable_count = 0
    min_bpm = 10000
    max_bpm = 0
    temp_data = []
    for key in vocal_data_dict.keys():
      length_fraction = (vocal_data_dict[key][-1]['start'] + vocal_data_dict[key][-1]['duration'])/max_steps
      if f_bin[0]<=length_fraction<f_bin[1] and syll_bin[0]<=len(vocal_data_dict[key])<syll_bin[1]:
        hit_count +=1
        orphaned_keys = [x for x in orphaned_keys if x!=key]
        temp_data += get_track_data(vocal_data_dict[key])
        ## indices 0-999 are probs of durations of active notes, 1000-1999 are probs of durations of rest
        probabilities = np.append(get_probabilities(np.asarray([x[2] for x in temp_data if type(x[3])==int])), get_probabilities(np.asarray([x[2] for x in temp_data if x[3]=='rest'])))
        held_syllable_count += len([x for x in vocal_data_dict[key] if x['syllable']=='-'])
        total_syllable_count += len(vocal_data_dict[key])
        if bpm_dict[key]<min_bpm:
          min_bpm = bpm_dict[key]
        if bpm_dict[key]>max_bpm:
          max_bpm = bpm_dict[key]

    len(vocal_data_dict.keys())
    name_string = 'f_bin'+'_'+str(f_bin[0])+'_'+str(f_bin[1])+'_syll_bin'+'_'+str(syll_bin[0])+'_'+str(syll_bin[1])
    markov_dict[name_string] = {
        'fraction_bins': f_bin,
        'syllable_bins': syll_bin,
        'probability_bin': hit_count/len(vocal_data_dict.keys()),
        'note_probabilities': probabilities,
        'held_syll_probability': held_syllable_count/(total_syllable_count+0.000000001),
        'bpm_range': [min_bpm, max_bpm]
    }

if len(orphaned_keys)!=0:
  raise Exception('at least one track does not fit into any bin')

temp_sum = 0
for mark_key in markov_dict.keys():
  temp_sum+=markov_dict[mark_key]['probability_bin']
  print(mark_key, '------', round(markov_dict[mark_key]['probability_bin'], 3), '-----', round(markov_dict[mark_key]['held_syll_probability'], 3))
  print(markov_dict[mark_key]['bpm_range'])

# with open('probabilities/markov_dict.pkl', 'wb') as filename:
#   pickle5.dump(markov_dict, filename)

## FL studio defines C4 as 12
## lowest possible singing note is probably FL studio's C3 (defined as 0 in json)
## highest possible singing note is probably FL studio's C7 (defined as 48 in json)

bottom_note = 0
top_note = 48
note_list = []
for name_key in vocal_data_dict.keys():
  for pitch_shift in np.arange(-120,120+12,12):
    if np.logical_and(min([x['pitch']+pitch_shift for x in vocal_data_dict[name_key]])>=bottom_note, max([x['pitch']+pitch_shift for x in vocal_data_dict[name_key]])<=top_note):
      for i in range(len(vocal_data_dict[name_key])):
        if i<(len(vocal_data_dict[name_key])-1):
          note_list.append([vocal_data_dict[name_key][i]['pitch']+pitch_shift, vocal_data_dict[name_key][i+1]['pitch']+pitch_shift])
    else:
      continue

note_list = np.asarray(note_list)

np.unique(note_list[:,0])

def pitch_probabilities(data):
  probabilities = np.zeros(top_note+1)

  for i in np.unique(data):
    if i>(top_note+1):
      raise Exception('pitch too high')
      
    if i<0:
      raise Exception('weird pitch ', i)
    matches = np.squeeze(np.where(data==i))
    if matches.ndim==0: ## exactly one match
      probability = 1/data.shape[0]
    else:
      probability = matches.shape[0]/data.shape[0]
    probabilities[i] = probability

  return probabilities

pitch_prob_dict = {}
for pitch in np.arange(top_note+1):
  indices = np.squeeze(np.where(note_list[:,0]==pitch))
  if indices.shape[0]==0:
    pitch_prob_dict[str(pitch)] = 'out_of_key'
  else:
    pitch_prob_dict[str(pitch)] = pitch_probabilities(note_list[indices,1])

pitch_prob_dict

# with open('probabilities/pitch_prob_dict.pkl', 'wb') as filename:
#   pickle5.dump(pitch_prob_dict, filename)
