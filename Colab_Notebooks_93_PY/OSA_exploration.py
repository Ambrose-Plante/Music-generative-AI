from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/OSA_study

pip install pyEDFlib

from pyedflib import highlevel
import matplotlib.pyplot as plt

# read an edf file
signals, signal_headers, header = highlevel.read_edf('EX88TYW6SK9VWSTE.edf')

signal_headers[12]

for i,x in enumerate(signals):
  print(i, signal_headers[i]['label'], x.shape)

signals[2].shape[0]/(200*60*60)

i = 16
print(signal_headers[i])
downsample = 100
sample_frequency = int(signal_headers[i]['sample_frequency']/downsample)
start = int(0) # seconds
end = int(1500) # seconds
plt.plot(signals[i][::downsample][start*sample_frequency:end*sample_frequency])
plt.xlabel('cycles')
# plt.ylim(-20,20)

i = 1
print(signal_headers[i])
plt.plot(signals[i][::200])

i = 2
print(signal_headers[i])
plt.plot(signals[i][::200])

i = 3
print(signal_headers[i])
plt.plot(signals[i][::200])

i = 4
print(signal_headers[i])
plt.plot(signals[i][::200])

i = 5
print(signal_headers[i])
plt.plot(signals[i][::200])

5891400/736425
