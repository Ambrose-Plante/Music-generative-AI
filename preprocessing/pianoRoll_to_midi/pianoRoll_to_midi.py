from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cmajor_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B']
c_major_chords = ['C_major', 'D_minor', 'D_major', 'E_minor', 'E_major', 'F_minor', 'F_major', 'G_major', 'A_minor', 'A_major', 'B_minor']

cmajor_small_ref=[]
for num in np.arange(2,4):
  cmajor_small_ref = cmajor_small_ref + [x+str(num) for x in cmajor_scale]
cmajor_small_ref = ['G1', 'A1', 'B1'] + cmajor_small_ref

filename = "./midi_ML_data/Train2/Avicii_Wake Me Up_chorus_0.npz"
data = np.load(filename)
vocals_array=data['vocals_array']

dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array.T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(vocals_array.T, interpolation='nearest', aspect='auto', origin='lower')
  # plt.xlim(0,900)
  # plt.ylim(0,30)
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);
  # plt.imshow(vocals_array.T, interpolation='nearest', origin='lower')

vocals_array.shape

trans_array = np.zeros((256,256))

for i in range(7):
  if i<6:
    trans_array[(25*i):(25*(i+1)), 0:256] = vocals_array[(256*i):(256*(i+1)), 0:25].T
  else:
    trans_array[(25*i):(25*(i+1)), 0:96] = vocals_array[(256*i):(256*(i+1)), 0:25].T

dpi = matplotlib.rcParams['figure.dpi']
height, width = trans_array.shape
figsize = 2*width/float(dpi), 2*height/float(dpi)

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(trans_array, interpolation='nearest', aspect='auto', origin='lower')

vocals_array2 = np.zeros((1632,25))

for i in range(7):
  if i<6:
    vocals_array2[(256*i):(256*(i+1)), 0:25] = trans_array[(25*i):(25*(i+1)), 0:256].T
  else:
    vocals_array2[(256*i):(256*(i+1)), 0:25] = trans_array[(25*i):(25*(i+1)), 0:96].T

dpi = matplotlib.rcParams['figure.dpi']
height, width = vocals_array2.T.shape
figsize = width/float(dpi), 5

fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0,0,1,1])

ax.imshow(vocals_array2.T, interpolation='nearest', aspect='auto', origin='lower')
  # plt.xlim(0,900)
  # plt.ylim(0,30)
ax.set_yticks(np.arange(len(cmajor_small_ref)), cmajor_small_ref);
  # plt.imshow(vocals_array.T, interpolation='nearest', origin='lower')

np.array_equal(vocals_array, vocals_array2)
