from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/

! pip install pickle5
! pip install pretty_midi
! apt install fluidsynth
! cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2
! pip install pydub

import os
import pickle5
import glob
import json
from pretty_midi import note_number_to_name, note_name_to_number
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from music_app.functions.midi_functions import *
from pydub import AudioSegment

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

from pydub import AudioSegment
import time

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

def write_metronome_midi(new_bpm, output_path, midi_data_1, midi_data_2):
    # Create a PrettyMIDI object
    cello_c_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    # Iterate over note names, which will be converted to note number later
    for beat in range(16*4):
        if ((beat+0.5)/(new_bpm*60))>max(midi_data_1.instruments[0].notes[-1].end, midi_data_2.instruments[0].notes[-1].end): # stop at end of chords or vox
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

def midi_to_mp3(artist, song, section, actual_bpm, key, BASE_DIR):
    #########################################################
    midi_data_1 = load_midi(vox_file, actual_bpm)

    for instrument in midi_data_1.instruments:
        for note in instrument.notes:
            note.pitch = note_name_to_number(note_number_to_name(note.pitch-key_shift(key))) ## shift to C major
            if note_number_to_name(note.pitch)[:-1] not in cmajor_scale:
              note.pitch += 1
              if note_number_to_name(note.pitch)[:-1] not in cmajor_scale:
                raise Exception('still not in scale somehow')


    ####################################################################

    midi_data = load_midi(chords_file, actual_bpm)
    instrument = midi_data.instruments[0]
    chord_list = []
    for chord_start_time in np.unique([note.start for note in instrument.notes]):
      chord_end_time = [note.end for note in instrument.notes if note.start==chord_start_time][0]
      chord_pitches = sorted([note.pitch for note in instrument.notes if note.start==chord_start_time])
      quality_ind, position_ind = get_triad(chord_pitches)
      quality = ['major', 'minor', 'dim'][quality_ind]

      chord = note_number_to_name(chord_pitches[position_ind])[:-1] + '_' + quality
      chord_shifted = note_number_to_name(chord_pitches[position_ind]-key_shift(key))[:-1] + '_' + quality
      if chord_shifted not in c_major_chords:
        if chord_shifted[1]=='#' or chord_shifted[1]=='b':
          chord_shifted = chord_shifted[:1] + chord_shifted[2:] # remove accidental
        new_ind = [i for i,x in enumerate(c_major_chords) if x[0]==chord_shifted[0]]
        if len(new_ind)==1:
          chord_shifted = c_major_chords[new_ind[0]]
        else:
          raise Exception('oops')
      print(section, chord_shifted, chord)

      chord_list.append([chord_shifted, chord_start_time, chord_end_time])

    # Create a PrettyMIDI object
    chord_midi_object = pretty_midi.PrettyMIDI(initial_tempo=actual_bpm)
    # Create an Instrument instance for a cello instrument
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    for chord in chord_list:
        fifth = 7
        if chord[0][2:]=='minor':
          third = 3
        elif chord[0][2:]=='major':
          third = 4
        else:
          raise Exception('chord went wrong')

        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3'), start=chord[1], end=chord[2])
        cello.notes.append(note)
        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3')+third, start=chord[1], end=chord[2])
        cello.notes.append(note)
        note = pretty_midi.Note(velocity=100, pitch=note_name_to_number(str(chord[0][0])+'3')+fifth, start=chord[1], end=chord[2])
        cello.notes.append(note)
        # Add it to our cello instrument

    # Add the cello instrument to the PrettyMIDI object
    chord_midi_object.instruments.append(cello)
    ####################################################################

    metronome_midi_path = BASE_DIR + artist + '_' + song + '_' + section + '_metronome.mid'
    metronome_wav_path = BASE_DIR + artist + '_' + song + '_' + section + '_metronome.wav'
    write_metronome_midi(actual_bpm, metronome_midi_path, midi_data_1, midi_data)

    temp_midi_path_1 = BASE_DIR + artist + '_' + song + '_' + section + '_temp_1.mid'
    temp_midi_path_2 = BASE_DIR + artist + '_' + song + '_' + section + '_temp_2.mid'

    midi_data_1.write(temp_midi_path_1)
    chord_midi_object.write(temp_midi_path_2)

    # Convert the first MIDI file to audio
    output_path_1 = BASE_DIR + artist + '_' + song + '_' + section + '_temp_1.wav'
    !fluidsynth -ni music_app/SalC5Light2.sf2 "$temp_midi_path_1" -F "$output_path_1" -r 44100

    # Convert the second MIDI file to audio
    output_path_2 = BASE_DIR + artist + '_' + song + '_' + section + '_temp_2.wav'
    !fluidsynth -ni music_app/SalC5Light2.sf2 "$temp_midi_path_2" -F "$output_path_2" -r 44100

    !fluidsynth -ni music_app/PVCDrum.sf2 "$metronome_midi_path" -F "$metronome_wav_path" -r 44100

    time.sleep(1)

    # Concatenate the audio files
    output_path = BASE_DIR + artist + '_' + song + '_' + section + '.mp3'
    sound1 = AudioSegment.from_file(output_path_1)
    sound2 = AudioSegment.from_file(output_path_2)
    metronome = AudioSegment.from_file(metronome_wav_path)
    combined = sound1.overlay(sound2)
    combined = combined.overlay(metronome)

    combined.export(output_path, format='mp3')
    #########################################################

    os.remove(output_path_1)
    os.remove(output_path_2)
    os.remove(temp_midi_path_1)
    os.remove(temp_midi_path_2)
    os.remove(metronome_midi_path)
    os.remove(metronome_wav_path)

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

row_start = 127 # the actual row number on the spreadsheet
row_end = 422   # the actual row number on the spreadsheet

dir_list = glob.glob('pdf_to_midi/checked_midi/**/*')
special_characters = ['&', '!', '\'']
banana=0
chord_list = []
note_list = []
vocal_data_dict = {}
bpm_dict = {}
song_count=0
for row,x in enumerate(song_list):

  artist = remove_end_space(x[0])
  song = remove_end_space(x[1]).replace('/', '-')

  if row not in np.arange(row_start-1,row_end):
    continue


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

      vocal_notes = [list(y) for y in midi_to_list(load_midi(vox_file, actual_bpm), actual_bpm)[0]]

      if sum([y[0]%2 for y in vocal_notes]):
        print(section, str(round(100*sum([y[0]%2 for y in vocal_notes])/len(vocal_notes)))+'% odd')

      for note in vocal_notes:
        note[2] = note_name_to_number(note_number_to_name(note[2]-key_shift(key)+12)) ## shift to C major

      notes = [note_number_to_name(y[2])[:-1] for y in vocal_notes]
      abberant_notes = []
      note_list += notes
      for note in notes:
        if note not in cmajor_scale:
          # raise Exception('abberant_note')
          abberant_notes.append(note)
      # if len(abberant_notes)>0:
      #   print(section, abberant_notes)


      abberent_chords=0
      midi_data = load_midi(chords_file, actual_bpm)
      instrument = midi_data.instruments[0]
      for chord_start_time in np.unique([note.start for note in instrument.notes]):
        chord_pitches = sorted([note.pitch for note in instrument.notes if note.start==chord_start_time])
        quality_ind, position_ind = get_triad(chord_pitches)
        quality = ['major', 'minor', 'dim'][quality_ind]

        chord = note_number_to_name(chord_pitches[position_ind])[:-1] + '_' + quality
        chord_shifted = note_number_to_name(chord_pitches[position_ind]-key_shift(key))[:-1] + '_' + quality
        chord_list.append(chord_shifted)
        if chord_shifted not in c_major_chords:
          abberent_chords += 1
      if abberent_chords/len(np.unique([note.start for note in instrument.notes]))>0.4:
          print('KEY PROBABLY WRONG', artist, song, section)

      BASE_DIR='checked_midi_wavs_Cmajor2/'
      output_path = BASE_DIR + artist + '_' + song + '_' + section + '.mp3'
      if not os.path.exists(output_path):
          midi_to_mp3(artist, song, section, actual_bpm, key, BASE_DIR='checked_midi_wavs_Cmajor2/')
  print('-----------------')

print('song_count:', song_count)
# for name_key in vocal_data_dict.keys():
#   for note in vocal_data_dict[name_key]:
#     note['duration'] += 1 ## I subtracted 1 when saving the json during data collection

plt.hist(chord_list, bins=np.arange(len(np.unique(chord_list)))-0.5)
plt.xticks(rotation='vertical');

plt.hist(note_list, bins=np.arange(len(np.unique(note_list)))-0.5)
plt.xticks(rotation='vertical');

len([x for x in note_list if x=='A#'])

section = "chorus"
vox_file = dir_name + '/' + section + '.mid'
chords_file = dir_name + '/ch_' + section + '.mid'


vocal_notes = [list(y) for y in midi_to_list(vox_file, bpm)[0]]

if sum([y[0]%2 for y in vocal_notes]):
  print(section, str(round(100*sum([y[0]%2 for y in vocal_notes])/len(vocal_notes)))+'% odd')

for note in vocal_notes:
  note[2] = note_name_to_number(note_number_to_name(note[2]-key_shift(key)+12)) ## shift to C major

vocal_data_dict[artist+'_'+song+'_'+section] = vocal_notes

bpm_dict[artist+'_'+song+'_'+section] = int(x[2])

abberent_chords=[]
midi_data = pretty_midi.PrettyMIDI(chords_file)
instrument = midi_data.instruments[0]
for chord_start_time in np.unique([note.start for note in instrument.notes]):
  chord_pitches = sorted([note.pitch for note in instrument.notes if note.start==chord_start_time])
  if (chord_pitches[1]-chord_pitches[0])==3:
    quality='minor'
  elif (chord_pitches[1]-chord_pitches[0])==4:
    quality='major'
  else:
    raise Exception('something messed up')
  chord = note_number_to_name(chord_pitches[0])[:-1] + '_' + quality
  chord_shifted = note_number_to_name(chord_pitches[0]-key_shift(key))[:-1] + '_' + quality
  if chord_shifted not in c_major_chords_strict:
    print(section, chord_shifted, chord)

notes = [note_number_to_name(y[2])[:-1] for y in vocal_notes]
abberant_notes = []
for note in notes:
  if note not in cmajor_scale_strict:
    # raise Exception('abberant_note')
    abberant_notes.append(note)
if len(abberant_notes)>0:
  print(section, abberant_notes)
print('-----------------')

pretty_midi.PrettyMIDI(chords_file).estimate_tempo()

pretty_midi.PrettyMIDI(vox_file).estimate_tempo()

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
