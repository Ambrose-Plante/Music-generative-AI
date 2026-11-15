from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/

! pip install pickle5
! pip install pretty_midi
# ! pip install fluidsynth
# !cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2
# ! pip install pydub

import os
import pickle5
import glob
import json
from pretty_midi import note_number_to_name, note_name_to_number
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from music_app.functions.midi_functions import *
import random
# from pydub import AudioSegment

import pretty_midi

# Function to load MIDI file and adjust note timings based on BPM
def load_midi(midi_file, new_bpm):
    # Get the old BPM from the MIDI file
    old_bpm = round(pretty_midi.PrettyMIDI(midi_file).get_tempo_changes()[1][0])
    # Load the MIDI file
    temp_midi_file = pretty_midi.PrettyMIDI(midi_file)

    # Adjust the start and end times of the notes to match the new BPM
    for instrument in temp_midi_file.instruments:
        for note in instrument.notes:
            note.start = note.start * (old_bpm / new_bpm)
            note.end = note.end * (old_bpm / new_bpm)

    return temp_midi_file

# Convert MIDI to a list of notes in simplified form with time represented in "beats"
def midi_to_list(midi_data, bpm):
    midi_list = []

    # Extract all note information
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch])

    # Sort by start time and pitch
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    # Convert time to beats
    bps = bpm / 60  # Beats per second
    steps_per_sec = bps * 4  # Four beats per second
    sixth_step_per_sec = steps_per_sec * 6
    half_step_per_sec = sixth_step_per_sec / 3

    # Simplify timing and return as a list of tuples (time in beats, pitch, duration in beats)
    midi_simple = [(round(x[0] * half_step_per_sec), round(x[1] * half_step_per_sec) - round(x[0] * half_step_per_sec), x[2]) for x in midi_list]

    return midi_simple


# Create a new MIDI file using midi list
def write_midi(midi_list, bpm):
    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    # Convert beats back to time in seconds
    bps = bpm / 60
    for time_a_in_beats, duration_in_beats, pitch_b in midi_list:
        start = time_a_in_beats / (bps * 4 * 6 / 3)  # Convert beats back to seconds
        end = start + (duration_in_beats / (bps * 4 * 6 / 3))  # Convert duration back to seconds
        new_note = pretty_midi.Note(velocity=100, pitch=pitch_b, start=start, end=end)
        piano.notes.append(new_note)

    new_midi.instruments.append(piano)
    return new_midi

import glob
import random
from tqdm import tqdm

file_list = glob.glob('all_vocal_midis_cmajor_with_section/*.mid')

chorus_file_list = [x for x in file_list if '_chorus' in x and 'pre_' not in x and 'post_' not in x]
verse_file_list = [x for x in file_list if '_verse' in x]
post_file_list = [x for x in file_list if '_post' in x]
pre_file_list = [x for x in file_list if '_pre_' in x]

iternames = [
    # ['chorus', chorus_file_list],
    # ['verse', verse_file_list],
    ['postchorus', post_file_list],
    ['prechorus', pre_file_list]
]

for section, section_list in iternames:

  # Define the BPM to align both files
  new_bpm = 120  # Set a target BPM for both MIDI files

  z = 0
  total_iterations = 1000

  midi_list_object = []
  for midi_file in section_list:
    midi_list_object.append(midi_to_list(load_midi(midi_file, new_bpm=new_bpm), bpm=new_bpm))

  for i in tqdm(range(1000)):

    steps_per_bar = 4*8

    final_midi = []
    if section=='postchorus':
      bar_range = 9
    else:
      bar_range = 17

    for bar in range(bar_range):
      bar_start = bar*steps_per_bar
      bar_end = (bar+1)*steps_per_bar-1

      possible_midi = []
      while len(possible_midi)==0:
        possible_midi_sample = random.sample(midi_list_object,1)[0]
        possible_midi = [x for x in possible_midi_sample if x[0]>=bar_start and x[0]<=bar_end]

      final_midi += possible_midi

      write_midi(final_midi, 120).write('mixed_vocal_midis/bar_frankenstein/Cmajor_Aminor/' + section + '/' + str(i).zfill(5) + '.mid')

import pretty_midi
import glob
import os
from tqdm import tqdm

# Function to transpose the midi list object by a certain number of semitones
def transpose_midi(midi_file, semitone_shift):
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Transpose each note in all instruments
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.pitch += semitone_shift  # Adjust pitch by semitone shift

    return midi_data

# Define the keys and corresponding semitone shifts
keys = {
    'C#major_A#minor': 1,
    'Dmajor_Bminor': 2,
    'D#major_Cminor': 3,
    'Emajor_C#minor': 4,
    'Fmajor_Dminor': 5,
    'F#major_D#minor': 6,
    'Gmajor_Eminor': 7,
    'G#major_Fminor': 8,
    'Amajor_F#minor': 9,
    'A#major_Gminor': 10,
    'Bmajor_G#minor': 11
}

# Path to the original C major/A minor MIDI files
original_midi_dir = 'mixed_vocal_midis/bar_frankenstein/Cmajor_Aminor'

# Get all the MIDI files in the C major directory
midi_files = glob.glob(f'{original_midi_dir}/**/*.mid', recursive=True)

# Iterate over each key and perform the transposition
for key_name, semitone_shift in keys.items():

    for midi_file in tqdm(midi_files):
        # Transpose the MIDI file to the target key
        transposed_midi = transpose_midi(midi_file, semitone_shift)

        # Create a new file path for the transposed file in the target key folder
        new_midi_file = midi_file.replace('Cmajor_Aminor', key_name)

        # Ensure the target directory exists
        new_midi_dir = os.path.dirname(new_midi_file)
        os.makedirs(new_midi_dir, exist_ok=True)

        # Save the transposed MIDI file
        transposed_midi.write(new_midi_file)
