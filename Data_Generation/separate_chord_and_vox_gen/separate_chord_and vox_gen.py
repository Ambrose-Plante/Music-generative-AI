from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/

import os
os.chdir('music_app/midi_ML_data/stable_diffusion/outputs/')

! tar -cvf processed_vox_09_30_23.tar processed_vox_09_30_23

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

with open('chords_generator/filtered_chords_list.pickle', "rb") as filename:
  chords_list = pickle.load(filename)

np.unique([round(x[-1][-1]) for x in chords_list])

for num_beats in np.unique([round(x[-1][-1]) for x in chords_list]):
  num_progressions = len([round(x[-1][-1]) for x in chords_list if round(x[-1][-1])==num_beats])
  print(num_beats, '--', num_beats/4-1, '--', num_progressions)

four_bars = []
eight_bars = []
twelve_bars = []
sixteen_bars = []

for x in chords_list:
  if round(x[-1][-1])==20:
    four_bars.append(x)
  elif round(x[-1][-1])==36:
    eight_bars.append(x)
  elif round(x[-1][-1])==52:
    twelve_bars.append(x)
  elif round(x[-1][-1])==68:
    sixteen_bars.append(x)

# with open('chords_generator/separate/four_bars.pickle', 'wb') as filename:
#   pickle.dump(four_bars, filename, pickle.HIGHEST_PROTOCOL)

# with open('chords_generator/separate/eight_bars.pickle', 'wb') as filename:
#   pickle.dump(eight_bars, filename, pickle.HIGHEST_PROTOCOL)

# with open('chords_generator/separate/twelve_bars.pickle', 'wb') as filename:
#   pickle.dump(twelve_bars, filename, pickle.HIGHEST_PROTOCOL)

# with open('chords_generator/separate/sixteen_bars.pickle', 'wb') as filename:
#   pickle.dump(sixteen_bars, filename, pickle.HIGHEST_PROTOCOL)

def unwrap_vocals_array(wrapped_vocals_array):
    vocals_array = np.zeros((544,17))
    for i in range(5):
        vocals_array[102*i:(102*i+102),:] = wrapped_vocals_array[:102,17*i:(17*i+17)]
    i=5
    vocals_array[510:,:] = wrapped_vocals_array[:34,17*i:(17*i+17)]

    return vocals_array


def array_to_list(vocals_array):
    array = unwrap_vocals_array(vocals_array)
    array[array>=0.5] = 1
    array[array<0.5] = 0

    array_list = []
    for i,col in enumerate(array):
      if sum(col>0.1):
        array_list.append([i*3, col.argmax()])

    if len(array_list)<2:
      return [1,2]

    final_list = []
    for i in range(len(array_list)-1):
      note_len = min(40, array_list[i+1][0]-array_list[i][0])
      final_list.append([array_list[i][0], array_list[i][0]+note_len, array_list[i][1]])

    i = len(array_list)-1
    final_list.append([array_list[i][0], array_list[i][0]+40, array_list[i][1]])

    return final_list

wrapped_vocals_array = np.load('music_app/midi_ML_data/stable_diffusion/outputs/output2.npy')

1632/96

# note_ends = []
# tot = len(wrapped_vocals_array)
# for i, vocals_array in enumerate(wrapped_vocals_array):
#   print(i, '/', tot)
#   filename = 'music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/' + str(i).zfill(9) + '.pickle'
#   if len(array_to_list(vocals_array))>5: # more than 5 notes in generated vox
#     note_ends.append([str(i).zfill(9), max([x[1] for x in array_to_list(vocals_array)])/96])
#     with open(filename, 'wb') as savefile:
#       pickle.dump(array_to_list(vocals_array), savefile, protocol=pickle.HIGHEST_PROTOCOL)

# filename = 'music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/note_ends.pickle'
# with open(filename, 'wb') as savefile:
#   pickle.dump(note_ends, savefile, protocol=pickle.HIGHEST_PROTOCOL)

with open('music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/note_ends.pickle', 'rb') as notesfile:
  notes_end = pickle.load(notesfile)

plt.hist([x[1] for x in notes_end], bins=np.arange(1, 18));

# import glob
# for filename in glob.glob('music_app/midi_ML_data/stable_diffusion/outputs/processed_vox_09_30_23/*.pickle'):
#   os.remove(filename)
