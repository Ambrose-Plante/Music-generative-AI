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

# Function to find the closest note based on timing in beats from fileB for each note in fileA
def match_notes_temporally(notesA, notesB):
    matched_notes = []
    for timeA, durationA, pitchA in notesA:
        # Find the closest note in time from notesB
        closest_noteB = min(notesB, key=lambda x: abs(x[0] - timeA))
        matched_notes.append((closest_noteB[2], timeA, durationA))  # (pitch from B, time from A, duration from A)
    return matched_notes

# Create a new MIDI file using timings from A and pitches from B
def create_midi_with_new_pitches(matched_notes, original_midi, bpm):
    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    # Convert beats back to time in seconds
    bps = bpm / 60
    for pitch_b, time_a_in_beats, duration_in_beats in matched_notes:
        start = time_a_in_beats / (bps * 4 * 6 / 3)  # Convert beats back to seconds
        end = start + (duration_in_beats / (bps * 4 * 6 / 3))  # Convert duration back to seconds
        new_note = pretty_midi.Note(velocity=100, pitch=pitch_b, start=start, end=end)
        piano.notes.append(new_note)

    new_midi.instruments.append(piano)
    return new_midi

os.chdir('pdf_to_midi/')

import glob
import random
from tqdm import tqdm

file_list = glob.glob('all_vocal_midis_cmajor/*.mid')

# Define the BPM to align both files
new_bpm = 120  # Set a target BPM for both MIDI files

z = 0
total_iterations = 1000

# Progress bar for the while loop, only updating when a file is written
with tqdm(total=total_iterations, desc="Processing MIDI files") as pbar:
    while z < total_iterations:
        selected_files = random.sample(file_list, 2)
        fileA, fileB = selected_files

        # Load the two MIDI files and adjust their timing to the new BPM
        midiA = load_midi(fileA, new_bpm)
        midiB = load_midi(fileB, new_bpm)

        # Convert MIDI files to lists of notes with simplified timing
        notesA_list = midi_to_list(midiA, new_bpm)
        notesB_list = midi_to_list(midiB, new_bpm)

        # Check if the timing of notesB is sufficient for notesA
        if notesB_list[-1][0] < notesA_list[-1][0]:
            continue
        else:
            # Match notes temporally
            matched_notes = match_notes_temporally(notesA_list, notesB_list)

            # Create a new MIDI file (fileC) with timings from A and pitches from B
            new_midi = create_midi_with_new_pitches(matched_notes, midiA, new_bpm)

            # Save the new MIDI file
            new_midi.write('mixed_vocal_midis/direct_mix/' + str(z).zfill(6) + '.mid')

            # Increment counter and update progress bar
            z += 1
            pbar.update(1)

import glob
import pretty_midi
import numpy as np

# Assuming midi_to_list function is already available
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

    bps = bpm / 60  # Beats per second
    steps_per_sec = bps * 4  # Four beats per second
    sixth_step_per_sec = steps_per_sec * 6
    half_step_per_sec = sixth_step_per_sec / 3

    # Here, we only care about timing, so we ignore pitch and return start and duration in beats
    midi_simple = [(round(x[0] * half_step_per_sec), round(x[1] * half_step_per_sec) - round(x[0] * half_step_per_sec)) for x in midi_list]

    return midi_simple

# Load all MIDI files and extract their note timings
file_list = glob.glob('all_vocal_midis_cmajor/*.mid')
new_bpm = 120  # Adjust the BPM as needed

# Store timings of all MIDI files in a list
midi_timings = []

for file in file_list:
    midi_data = pretty_midi.PrettyMIDI(file)
    midi_timings.append(midi_to_list(midi_data, new_bpm))

# Convert all timings into a flat list of numpy arrays for clustering
all_timings = [np.array(timings)[:, 0] for timings in midi_timings]  # Only using start times

from sklearn.preprocessing import StandardScaler

# Pad sequences so they all have the same length
max_length = max(len(t) for t in all_timings)
normalized_timings = np.array([np.pad(t, (0, max_length - len(t)), 'constant') for t in all_timings])

from sklearn.cluster import KMeans

# Apply K-means clustering (you can tune n_clusters as needed)
n_clusters = 10  # Define the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(normalized_timings)

# Cluster labels for each MIDI file
labels = kmeans.labels_

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_timings)

# Scatter plot of the PCA result, colored by cluster labels
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('MIDI Timing Clusters')
plt.colorbar()
plt.show()

import matplotlib.pyplot as plt

# Function to visualize a single MIDI file
def plot_midi_timings(midi_data, title="MIDI File Visualization"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through instruments and their notes
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Plot the note timing (start, duration) ignoring pitch
            ax.plot([note.start, note.end], [note.pitch, note.pitch], label=instrument.name if instrument.is_drum else None, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    ax.set_title(title)
    plt.show()

# Assuming that 'labels' contains the cluster labels for each MIDI file
# and 'file_list' contains the paths to each MIDI file

def visualize_cluster(cluster_label, labels, file_list):
    # Get the indices of files that belong to the specified cluster
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]

    # Loop through all the files in the cluster and visualize each one
    for idx in cluster_indices[:100][::5]:
        midi_file = file_list[idx]
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Plot the MIDI file's timings
        plot_midi_timings(midi_data, title=f"Cluster {cluster_label} - {midi_file}")

visualize_cluster(3, labels, file_list)  # Replace '0' with the desired cluster label
