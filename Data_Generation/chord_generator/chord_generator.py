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

! pip install pretty_midi

import os
import glob
import json
from pretty_midi import note_number_to_name, note_name_to_number
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import random

max_steps = 17*4*4*2

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
    half_step_per_sec = sixth_step_per_sec/3

    c2_midi_note=note_name_to_number("C2")
    note_range=note_name_to_number("C7")-note_name_to_number("C2")

    midi_simple = [(round(x[0]*half_step_per_sec), x[2]-c2_midi_note) for x in midi_list]

    for q in np.unique([x[0] for x in midi_simple]):
      if len([x[0] for x in midi_simple if x[0]==q])>1:
        print('repeated note')

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

      root_shifted = chord_shifted.split('_')[0]
      quality = chord_shifted.split('_')[1]

      chords_list.append([root_shifted, quality, chord_start_time, chord_end_time])

    chords_list = sorted(chords_list, key=lambda z: (z[2], z[3]))
    return chords_list

    # bps=bpm/60
    # steps_per_sec=bps*4
    # sixth_step_per_sec=steps_per_sec*6

    # quantized_chords_list = [[z[2], z[3], round(z[0]*sixth_step_per_sec), round(z[1]*sixth_step_per_sec)] for z in chords_list]
    # quantized_chords_list = [x for x in quantized_chords_list if (x[3]-x[2])!=0] # get rid of notes with length 0

    # return quantized_chords_list
    # for i in range(len(quantized_chords_list)):
    #   for j in range(len(quantized_chords_list)):
    #     if i==j:
    #       continue
    #     if quantized_chords_list[i][2]<=quantized_chords_list[j][1]<quantized_chords_list[i][2]:
    #       raise Exception('chords overlap', artist, song, section, i, j)
    #     if quantized_chords_list[i][2]>=quantized_chords_list[j][2]>quantized_chords_list[i][1]:
    #       raise Exception('chords overlap', artist, song, section, i, j)

    # max_steps = 17*4
    # num_scale_degrees = len(c_major_chords)

    # chords_array = np.zeros((max_steps, num_scale_degrees))
    # for z in quantized_chords_list:
    #     chords_array[z[1]:z[2],z[0]]=1

    # return quantized_chords_list, chords_array

dir_list = glob.glob('pdf_to_midi/checked_midi/**/*')
special_characters = ['&', '!', '\'']

chord_list = []
note_list = []
vocal_data_dict = {}
bpm_dict = {}
song_count=0
tot_tracks=0
bad_tracks=0

row = random.choice(np.arange(2,400))
x = song_list[row]

artist = remove_end_space(x[0])
song = remove_end_space(x[1]).replace('/', '-')

actual_bpm = int(x[2])

# if x[6]=='midi_only':
#   actual_bpm = int(x[2])
#   # print(artist, song, 'midi_only')
# else:
#   continue

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

    chords_list = chords_process(chords_file, actual_bpm, key)

print('-----------------')

chords_list

C_chord_dict = {
    'C' : ['C4', 'G4', 'E5'],
    # 'Cmaj6add9' : ['C4', 'G4', 'D5', 'E5', 'A5'],
    'Caj7' : ['C4', 'G4', 'E5', 'B5'],
}

D_chord_dict = {
    'D' : ['D4', 'A4', 'F5'],
    # 'Dm6add9' : ['D4', 'A4', 'E5', 'F5', 'B5'],
    'Dm7' : ['D4', 'A4', 'F5', 'C5'],
}

E_chord_dict = {
    'E' : ['E4', 'B4', 'D5', 'E5'],
    'Em7' : ['E4', 'B4', 'D5', 'E5', 'G5'],
    # 'Em7add11' : ['E4', 'E5', 'G5', 'A5', 'D6'],
}

F_chord_dict = {
    'F' : ['F4', 'A4', 'C5'],
    # 'Fmaj6add9' : ['F3', 'C4', 'A4', 'D5', 'G5'],
    'Fmaj7' : ['F3', 'A4', 'C5', 'E5'],
}

G_chord_dict = {
    'G' : ['G3', 'D4', 'B4', 'D5'],
    'Gadd9' : ['G3', 'D4', 'B4', 'D5', 'A5'],
    # 'G13' : ['G3', 'F4', 'B4', 'A5', 'E5'],
}

A_chord_dict = {
    'Am' : ['A3', 'E4', 'C5', 'C6'],
    'Am7' : ['A3', 'G4', 'C5', 'E5'],
    'Am7_2' : ['A3', 'E4', 'C5', 'G5', 'C6'],
    # 'Am11' : ['A3', 'E4', 'C5', 'D5', 'G5', 'B5'],
}

B_chord_dict = {
    # 'Bm7add11b5' : ['B3', 'A4', 'D5', 'F5', 'B5', 'E6'],
    'Bm7b5' : ['B4', 'D5', 'F5', 'A5'],
}

def generate_chord(root,quality,start,end):
    if root=='A':
      temp_list = A_chord_dict[random.choice(list(A_chord_dict.keys()))]
    elif root=='B':
      temp_list = B_chord_dict[random.choice(list(B_chord_dict.keys()))]
    elif root=='C':
      temp_list = C_chord_dict[random.choice(list(C_chord_dict.keys()))]
    elif root=='D':
      temp_list = D_chord_dict[random.choice(list(D_chord_dict.keys()))]
    elif root=='E':
      temp_list = E_chord_dict[random.choice(list(E_chord_dict.keys()))]
    elif root=='F':
      temp_list = F_chord_dict[random.choice(list(F_chord_dict.keys()))]
    elif root=='G':
      temp_list = G_chord_dict[random.choice(list(G_chord_dict.keys()))]
    else:
      raise Exception('chord went wrong')

    previous_notes_list = []
    for note in temp_list:
      previous_notes_list.append(note_name_to_number(note))

    return np.asarray(previous_notes_list)

def remove_elements_with_small_difference(notes_list):
    for q in range(1000):
      ind = random.choice(range(len(notes_list)))
      remove_inds = [i for i,x in enumerate(notes_list) if np.abs(x-notes_list[ind])==1]
      notes_list = np.delete(notes_list, remove_inds)

    return notes_list

######################################################################################################################
# Create a PrettyMIDI object
chord_midi_object = pretty_midi.PrettyMIDI(initial_tempo=actual_bpm)
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)

#############initialize###############################
root,quality,start,end = chords_list[0]
previous_notes_list = generate_chord(root,quality,start,end)
# previous_notes_list = [x+random.choice([-12,0,12]) for x in previous_notes_list]
for note in previous_notes_list:
    temp_note = pretty_midi.Note(velocity=100, pitch=note, start=start, end=end)
    cello.notes.append(temp_note)
# Add it to our cello instrument
#######################################################

for root,quality,start,end in chords_list[1:]:
    new_notes_list = generate_chord(root,quality,start,end)

    final_notes_list = []
    for prev_note in previous_notes_list:
        octave_ind = 'zebra'
        finite_loop = 0
        while octave_ind == 'zebra':
            if finite_loop > 100:
              raise Exception('no suitable note found')

            temp_note = random.choice(new_notes_list)

            try:
              octave_ind = random.choice(np.where(np.abs((np.arange(-48,48,12)+temp_note)-prev_note)<=2)[0])
              shifted_note = np.arange(-48,48,12)[octave_ind]+temp_note
            except:
              octave_ind = 'zebra'

            finite_loop += 1
        final_notes_list.append(shifted_note)

    final_notes_list = remove_elements_with_small_difference(final_notes_list)

    for note in final_notes_list:
        temp_note = pretty_midi.Note(velocity=100, pitch=note, start=start, end=end)
        cello.notes.append(temp_note)
    # Add it to our cello instrument

    ##### add root note below chord ####################
    root_note = note_name_to_number(root+'3')
    while max(root_note - final_notes_list)>0:
      root_note -= 12
    temp_note = pretty_midi.Note(velocity=100, pitch=root_note, start=start, end=end)
    cello.notes.append(temp_note)
    ##################################################

    # previous_notes_list = final_notes_list

# Add the cello instrument to the PrettyMIDI object
chord_midi_object.instruments.append(cello)
########################################################################################################################

chord_midi_object.write('chord_temp.mid')

root_note = 62
while max(root_note - final_notes_list)>=0:
  root_note -= 12

root_note

root_note - final_notes_list

final_notes_list

previous_notes_list[i]

# for i in range(len(new_notes_list)):
#   deltas = np.arange(-5,5)*12
#   min_dists = []
#   for delta in deltas:
#     min_dists.append(min(np.abs(previous_notes_list-new_notes_list[i]+delta)))

#   new_notes_list[i] -= deltas[np.argmin(min_dists)]

# new_notes_list = remove_elements_with_small_difference(new_notes_list)

# ##### add root note below chord ####################
# root_note = note_name_to_number(root+'3')
# while min(root_note - new_notes_list)>=0:
#   root_note -= 12
# temp_note = pretty_midi.Note(velocity=100, pitch=root_note, start=start, end=end)
# cello.notes.append(temp_note)
# ##################################################
