from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/

! pip install pretty_midi

import sys

import os
import pickle
import glob
import pretty_midi
from pretty_midi.utilities import note_number_to_name, note_name_to_number
import json
import csv
import random
import numpy as np
import time
import subprocess

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


def load_midi(midi_file, new_bpm):
    old_bpm = round(pretty_midi.PrettyMIDI(midi_file).get_tempo_changes()[1][0])
    temp_midi_file = pretty_midi.PrettyMIDI(midi_file, old_bpm)

    for instrument in temp_midi_file.instruments:
      for note in instrument.notes:
        note.start = note.start*(old_bpm/new_bpm)
        note.end = note.end*(old_bpm/new_bpm)

    return temp_midi_file


def midi_to_list(midi_data, bpm, key):

    midi_list = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch-key_shift(key)
            velocity = note.velocity
            midi_list.append([start, end, pitch])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    midi_simple = [[round(x[0]*sixth_step_per_sec), round(x[1]*sixth_step_per_sec), x[2]] for x in midi_list]

    return midi_simple

chord_definitions = [
    ['C', 'E', 'G'],
    ['D', 'F', 'A'],
    ['E', 'G', 'B'],
    ['F', 'A', 'C'],
    ['G', 'B', 'D'],
    ['A', 'C', 'E'],
    ['B', 'D', 'F']
]

def chords_midi_to_array(midi_file_path, bpm, key):
  chords_midi = load_midi(midi_file_path, bpm)

  WARNING = 'Life is peaches'
  note_list = midi_to_list(chords_midi, bpm, key)

  master_chord_ind_list = []
  all_note_inds = np.arange(len(note_list))

  while len(all_note_inds)>0:
    temp_chord_inds = [random.choice(all_note_inds)]
    all_note_inds = [x for x in all_note_inds if x not in temp_chord_inds]
    ind_was_added=True
    while ind_was_added:
      ind_was_added=False
      for i in temp_chord_inds:
        inds_to_add = []
        for j in all_note_inds:
          if note_list[j][0]<=note_list[i][0]<note_list[j][1] or note_list[j][0]<note_list[i][1]<=note_list[j][1]:
            ind_was_added = True
            inds_to_add.append(j)
            temp_chord_inds.append(j)
        all_note_inds = [x for x in all_note_inds if x not in inds_to_add]

    master_chord_ind_list.append(temp_chord_inds)

  max_steps = 17*4*4*6
  num_scale_degrees = len(c_major_chords)

  chords_array = np.zeros((max_steps, num_scale_degrees))

  for chord_set in master_chord_ind_list:
    chord_start = min([note_list[i][0] for i in chord_set])
    chord_end = max([note_list[i][1] for i in chord_set])

    if len(chord_set)>=3: # can use chord definitions
      temp_chord = [note_number_to_name(note_list[i][2])[:-1] for i in chord_set]
      chord_overlaps = []
      for chord in chord_definitions:
        chord_overlaps.append(len([x for x in chord if x in temp_chord]))
      exact_matches = [i for i,x in enumerate(chord_overlaps) if x==3]
      if len(exact_matches)>0: # at least one exact match exists
        if len(exact_matches)==1: # exactly one match, nice
          chord_root = chord_definitions[exact_matches[0]][0]
        else: # for now just pick one randomly. Will update in future
          chord_root = chord_definitions[random.choice(exact_matches)][0]
      else: # no exact match; just guess bass note
        chord_root = note_number_to_name(min([note_list[i][2] for i in chord_set]))[:-1]
    else: # not enough notes to define a chord; best guess is bass note
      chord_root = note_number_to_name(min([note_list[i][2] for i in chord_set]))[:-1]

    if not chord_root in [x[0] for x in c_major_chords]:
      WARNING = 'Some chords not in scale. Double check key. Editing them and continuing'
      print('WARNING:', chord_root, 'not in C major. Editing this chord to', chord_root[0])
      chord_root = chord_root[0]

    scale_degree = [i for i,x in enumerate(c_major_chords) if x[0]==chord_root][0]

    chords_array[chord_start:chord_end,scale_degree]=1
  return chords_array, WARNING


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

        bps=bpm/60
        steps_per_sec=bps*4
        sixth_step_per_sec=steps_per_sec*6

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

midi_file_path = 'chord_bpm132_Cmajor_000033.mid'
bpm = 132
key = ['C', 'major']
# desired_section, user, vox_gen_ind

chords_array, WARNING = chords_midi_to_array(midi_file_path=midi_file_path, bpm=bpm, key=key)

chord_end = (np.max(np.where(chords_array.max(axis=1)==1)[0])+1)/96

with open('music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/note_ends.pickle', 'rb') as handle:
  note_ends = pickle.load(handle)

gen_vox_file = 'music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/' + random.choice([x for x in note_ends if (chord_end-1)<x[1]<(chord_end+1)])[0] + '.pickle'

with open(gen_vox_file, 'rb') as filehandle:
    gen_vox = pickle.load(filehandle)

chords_array.shape

def generate_vox(midi_file_path, bpm, key, desired_section, user, vox_gen_ind):
  OUTPUT_DIR = 'static/user_data/' + user

  if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)

  chords_array, WARNING = chords_midi_to_array(midi_file_path=midi_file_path, bpm=bpm, key=key)

  noise = torch.randn(1, nz, device=device)

  if desired_section == 'post_chorus':
    section = [0,0,0,1,0]
  elif desired_section == 'pre_chorus':
    section = [0,1,0,0,0]
  elif desired_section == 'chorus':
    section = [0,0,1,0,0]
  elif desired_section == 'verse1':
    section = [1,0,0,0,0]
  elif desired_section == 'verse2':
    section = [0,0,0,0,1]
  else:
    raise Exception('Bad section input')

  if bpm<100:
    bpm_array = [1,0,0] # slow
  elif 100<=bpm<140:
    bpm_array = [0,1,0] # medium
  else:
    bpm_array = [0,0,1] # fast

  chords_array, section, bpm_array = torch.from_numpy(chords_array).float(), torch.FloatTensor(section), torch.FloatTensor(bpm_array)

  chords_array = torch.unsqueeze(torch.unsqueeze(chords_array,0),0).to(device) # must add dims for batch and channel
  section = torch.unsqueeze(section, 0).to(device) # must add dim for batch
  bpm_array = torch.unsqueeze(bpm_array, 0).to(device) # must add dim for batch

  with torch.no_grad():
    generated_vox = netG(noise, chords_array, section, bpm_array)

  bps=bpm/60
  steps_per_sec=bps*4
  sixth_step_per_sec=steps_per_sec*6

  chords_midi_path = OUTPUT_DIR + 'chords.mid'
  acc_file = OUTPUT_DIR + 'chords.wav'
  write_chords_midi(chords_array[0,0], bpm, chords_midi_path)
  subprocess.run(["fluidsynth", "-ni", "data_files/SalC5Light2.sf2", chords_midi_path, "-F", acc_file, "-r", "44100"])

  vocal_midi_list = array_to_list(generated_vox[0,0])

  octave_shift=2
  velocity = 100

  pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
  inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello')
  pm.instruments.append(inst)

  for start,end,pitch in [[x[0],x[1],x[2]] for x in vocal_midi_list]:
      if pitch!=len(cmajor_small_ref):
        pitch = cmajor_small_ref[pitch][:-1]+str(int(cmajor_small_ref[pitch][-1])+octave_shift)
        inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(pitch), start/sixth_step_per_sec, end/sixth_step_per_sec))

  vocal_midi = OUTPUT_DIR + 'vox.mid'
  vocal_wav = OUTPUT_DIR + 'vox.wav'
  pm.write(vocal_midi)
  time.sleep(1)

  subprocess.run(["fluidsynth", "-ni", "data_files/SalC5Light2.sf2", vocal_midi, "-F", vocal_wav, "-r", "44100"])

  time.sleep(1)

  sound1 = AudioSegment.from_file(vocal_wav)
  sound2 = AudioSegment.from_file(acc_file)

  metronome_midi_path = OUTPUT_DIR + 'metronome.mid'
  metronome_wav_path = OUTPUT_DIR + 'metronome.wav'
  write_metronome_midi(bpm, metronome_midi_path, vocal_midi_list[-1][1]/sixth_step_per_sec, sound2.duration_seconds)
  subprocess.run(["fluidsynth", "-ni", "data_files/PVCDrum.sf2", metronome_midi_path, "-F", metronome_wav_path, "-r", "44100"])
  metronome = AudioSegment.from_file(metronome_wav_path)

  combined = sound1.overlay(sound2)
  combined = combined.overlay(metronome)

  combined.export(OUTPUT_DIR + 'combined_vox.mp3', format='mp3')

  os.remove(metronome_midi_path)
  os.remove(metronome_wav_path)

  os.remove(chords_midi_path)
  os.remove(acc_file)

  storage = S3Boto3Storage()
  local_file_path = OUTPUT_DIR + 'combined_vox.mp3'
  S3_file_path = 'static/user_data/' + user + 'combined_vox_' + str(vox_gen_ind).zfill(6) + '.mp3'
  with open(local_file_path, 'rb') as wav_file:
      wav_content = wav_file.read()
  storage.save(S3_file_path, ContentFile(wav_content))

  local_file_path = OUTPUT_DIR + 'vox.mid'
  S3_file_path = 'static/user_data/' + user + 'vox_' + str(vox_gen_ind).zfill(6) + '.mid'
  with open(local_file_path, 'rb') as midi_file:
      midi_content = midi_file.read()
  storage.save(S3_file_path, ContentFile(midi_content, 'rb'))
