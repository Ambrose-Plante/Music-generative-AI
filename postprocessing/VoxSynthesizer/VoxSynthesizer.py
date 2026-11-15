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

!conda install -c conda-forge librosa --yes
!conda install -c roebel pretty_midi --yes

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import json
from pretty_midi.utilities import note_number_to_name, note_name_to_number
import glob
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

pitched_consonents = ['W', 'M', 'N', 'L', 'V', 'Z', 'NG']

consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "P", "R", "S", "SH", "T", "TH"
              , "Y", "ZH"]

sample_rate = 22050
##########
def remainder(x):
  return x-int(x)
#############


######################
def disseminate_steps(available_steps, num_consonents):
  base = int(available_steps/num_consonents)
  addition = np.zeros(num_consonents)
  addition[:(available_steps-base*num_consonents)]=1
  num_steps = base*np.ones(num_consonents)+addition
  return num_steps.astype(int)
########################


def get_sample_signal(vowel, target_steps, target_pitch, sample_rate=sample_rate):

  filename = glob.glob('AutoVox/120bpm_C4/' + vowel + '_*')
  if len(filename)==1:
    filename = filename[0]
  else:
    raise Exception('file name issue')

  sample_steps = int(filename.split('steps.wav')[0].split(vowel+'_')[1])
  sample_bpm = int(filename.split('bpm')[0].split('AutoVox/')[1])
  sample_pitch = note_name_to_number(filename.split('bpm_')[1].split('/')[0])
  pitch_shift = target_pitch - sample_pitch

  signal, sample_rate = librosa.load(filename, sr=sample_rate)

  ### shift sample to proper bpm ###
  shift_fraction=bpm/sample_bpm
  signal_shifted = librosa.effects.time_stretch(signal, shift_fraction)
  ! rm -f 'temp.wav'
  sf.write('temp.wav', signal_shifted, sample_rate, subtype='PCM_24')
  signal, sample_rate = librosa.load('temp.wav', sr=sample_rate) 
  #########################################

  ### shift sample to num_steps ###
  shift_fraction=sample_steps/target_steps
  signal_shifted = librosa.effects.time_stretch(signal, shift_fraction)
  ! rm -f 'temp.wav'
  sf.write('temp.wav', signal_shifted, sample_rate, subtype='PCM_24')
  signal, sample_rate = librosa.load('temp.wav', sr=sample_rate) 
  ####################################

  if (vowel in pitched_consonents) or (vowel in vowels):
    print(vowel, 'in pitched consonents or vowels')
    signal = librosa.effects.pitch_shift(signal, sample_rate, pitch_shift)

  return signal

f = open('midi_data/Avicii_Wake Me Up/chorus.json')
notes = json.load(f)

bpm = 124
bps=bpm/60
steps_per_sec=bps*4
sixth_step_per_sec=steps_per_sec*6

vocals=[]
x = notes[0]

f = open('midi_data/Avicii_Wake Me Up/chorus.json')

i=0
while i<len(notes):
  x = notes[i]

  if i==len(notes)-1: #last note, just append it
    vocals.append((x['start'], x['start']+x['duration'], note_name_to_number("C2")+x['pitch'], x['syllable']))
    i+=1
  elif notes[i+1]['syllable']=='-': # slide notes coming up

    phonemes=x['syllable'].split(' ')
    temp = [(i,x) for i,x in enumerate(phonemes) if x in vowels]

    if len(temp)==1:
      vowel_index,vowel = temp[0]
    else:
      raise VowelError('not exactly one vowel in syllable')

    consonents_before_vowel = phonemes[0:vowel_index]
    consonents_after_vowel = phonemes[(vowel_index+1):]

    j=1
    while notes[j+i+1]['syllable']=='-':
      j+=1

    if consonents_before_vowel:
      syll_begin = ' '.join(consonents_before_vowel) + ' ' + vowel
    else:
      syll_begin = vowel
    
    if consonents_after_vowel:
      syll_end = vowel + ' ' + ' '.join(consonents_after_vowel)
    else:
      syll_end = vowel

    # append phonemes up through the vowel
    vocals.append((notes[i]['start'], notes[i]['start']+notes[i]['duration'], note_name_to_number("C2")+notes[i]['pitch'], syll_begin))

    if j>1: # 3 or more slide notes; append vowel as intermediate j-2 times
      for n in range(j-1):
        vocals.append((notes[i+1+n]['start'], notes[i+1+n]['start']+notes[i+1+n]['duration'], note_name_to_number("C2")+notes[i+1+n]['pitch'], vowel))

    vocals.append((notes[i+j]['start'], notes[i+j]['start']+notes[i+j]['duration'], note_name_to_number("C2")+notes[i+j]['pitch'], syll_end))
    i+=(j+1)

  else: # no slide notes
    vocals.append((x['start'], x['start']+x['duration'], note_name_to_number("C2")+x['pitch'], x['syllable']))
    i+=1

vocals[:8]

def syllable_to_signal(syll):
  start=syll[0]
  end=syll[1]
  duration=end-start
  pitch=syll[2]

  if syll[3]=='-':
    print('slide note')
  else:
    phonemes=syll[3].split(' ')
    temp = [(i,x) for i,x in enumerate(phonemes) if x in vowels]

    if len(temp)==1:
      vowel_index,vowel = temp[0]
    else:
      raise VowelError('not exactly one vowel in syllable')

    consonents_before_vowel = phonemes[0:vowel_index]
    consonents_after_vowel = phonemes[(vowel_index+1):]

    before_signals=[]
    if len(consonents_before_vowel)>0:
      consonent_before_steps = int(np.ceil(duration*0.2))
      if consonent_before_steps<len(consonents_before_vowel):
        raise Exception('not enough space for consonents before vowel')
      elif consonent_before_steps>8:
        consonent_before_steps=8
      num_steps = disseminate_steps(consonent_before_steps, len(consonents_before_vowel))
      for q,consonent in enumerate(consonents_before_vowel):
        before_signals.append(get_sample_signal(consonent, num_steps[q], pitch, sample_rate))
    else:
      consonent_before_steps=0

    after_signals=[]
    if len(consonents_after_vowel)>0:
      consonent_after_steps = int(np.ceil(duration*0.2))
      if consonent_after_steps<len(consonents_after_vowel):
        consonent_after_steps = int(np.ceil(duration*0.3))
        if consonent_after_steps<len(consonents_after_vowel):
          raise Exception('not enough space for consonents after vowel')
      elif consonent_after_steps>8:
        consonent_after_steps=8
      num_steps = disseminate_steps(consonent_after_steps, len(consonents_after_vowel))
      for q,consonent in enumerate(consonents_after_vowel):
        after_signals.append(get_sample_signal(consonent, num_steps[q], pitch, sample_rate))
    else:
      consonent_after_steps=0

    vowel_steps = duration-consonent_before_steps-consonent_after_steps
    vowel_signal = get_sample_signal(vowel, vowel_steps, pitch, sample_rate)

    if len(after_signals)>0:
      if len(before_signals)>0:
        syll_signal = np.concatenate((np.hstack(before_signals), vowel_signal, np.hstack(after_signals)), axis=None)
      else:
        syll_signal = np.concatenate((vowel_signal, np.hstack(after_signals)), axis=None)
    else:
      if len(before_signals)>0:
        syll_signal = syll_signal = np.concatenate((np.hstack(before_signals), vowel_signal), axis=None)
      else:
        syll_signal = vowel_signal

    return syll_signal

signal=np.zeros(round((1632/sixth_step_per_sec)*sample_rate))

for i, syllable in enumerate(vocals):
  start=round((syllable[0]/sixth_step_per_sec)*sample_rate)
  new_signal=syllable_to_signal(syllable)
  signal[start:(start+new_signal.shape[0])] = new_signal

syllable

plt.plot(signal)

! rm -f 'temp.wav'
sf.write('temp.wav', signal, sample_rate, subtype='PCM_24')

from IPython.display import Audio
Audio('temp.wav')
