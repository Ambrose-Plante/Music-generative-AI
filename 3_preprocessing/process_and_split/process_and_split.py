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
conda install --channel defaults conda python=3.7 --yes
conda update --channel defaults --all --yes

!conda --version # now returns 4.8.3
!python --version # now returns Python 3.6.10 :: Anaconda, Inc.
import sys
sys.path

_ = (sys.path
        .append("/usr/local/lib/python3.7/site-packages"))

!conda install -c conda-forge pickle5 --yes
!conda install -c roebel pretty_midi --yes
!conda install -c conda-forge librosa --yes

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

import os
import pickle5
import glob
import librosa
import librosa.display
import soundfile as sf
import IPython.display
import pretty_midi
from pretty_midi.utilities import note_number_to_name, note_name_to_number
import json
import csv
import random
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

datafile = '/content/drive/My Drive/integrated/data/train/The Weeknd_Love Me Harder_0_0_chorus.npz'

data = np.load(datafile)
key = note_number_to_name(np.where(data['key_array']==1)[0][0] + note_name_to_number("C0"))[:-1]
bpm_table = np.load('bpm_table/bpm_table.npy')
# bpm = np.where(data['bpm_array']==1)[0][0] + 50
bpm=62
bpm_row = bpm_table[np.where(bpm_table[:,0]==bpm)[0][0]]
NFFT = int(bpm_row[1])
sample_rate = int(bpm_row[2])
hop = int(bpm_row[3])

import pickle
with open('/content/drive/MyDrive/integrated/val_data.pickle', 'rb') as f:
    val_data = pickle.load(f)

index=0
bpm_table = np.load('bpm_table/bpm_table.npy')
bpm = 90
bpm_row = bpm_table[np.where(bpm_table[:,0]==bpm)[0][0]]
NFFT = int(bpm_row[1])
sample_rate = int(bpm_row[2])
hop = int(bpm_row[3])

print(val_data['names'][index])
audio = librosa.feature.inverse.mel_to_audio(val_data['vox'][index], sr=sample_rate, n_fft=NFFT, hop_length=hop, center=False)
audio[audio>0.5]=0.5
audio[audio<-0.5]=-0.5

audio = librosa.feature.inverse.mel_to_audio(data['vox_input'], sr=sample_rate, n_fft=NFFT, hop_length=hop, center=False)

plt.plot(audio)

import soundfile as sf
sf.write('vocal_reconstruction_streets.wav', audio, sample_rate)

plt.figure(figsize=(15,15))
plt.imshow(val_data['midi'][index].sum(axis=0), origin='lower')

val_data['bpm'].shape

plt.imshow(data['vox_input'])

def color_assign(artist, song):
    if os.path.exists("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl"):
        if os.path.exists("saved_pronunciations/"+artist+"_"+song.replace('/', '-')+".pkl"):

            file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
            loaded_dictionary = pickle5.load(file_to_read)

            complete=True
            rdy=True
            for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:
                if len(loaded_dictionary[x])>2:
                    if x=='verse1':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_verse1.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse1_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='pre_chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_prechorus.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/prechorus_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_chorus.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/chorus_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='post_chorus':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_chorus.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_chorus.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_post.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/post_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

                    elif x=='verse2':
                        vocals_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2.json')
                        vox_midi_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2.mid')
                        chords_exist = os.path.exists('midi_data/' + artist + '_' + song + '/ch_verse2.mid')
                        acc_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2_acc.wav')
                        vox_wav_exist = os.path.exists('midi_data/' + artist + '_' + song + '/verse2_vocals.wav')
                        if not (vocals_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                            complete=False
                            if not (vox_midi_exist and chords_exist and acc_wav_exist and vox_wav_exist):
                                rdy=False

            if complete==True:
                return (artist, song)

def chords_process(filename, bpm, pitch_shift):
    midi_data = pretty_midi.PrettyMIDI(filename)

    midi_list = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            if velocity<50:
                quality=0 #minor
            else:
                quality=1 #major
            midi_list.append([start, end, pitch+pitch_shift, quality])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6

    c0_midi_note=note_name_to_number("C0")

    midi_simple = [(round(x[0]*sixth_step_per_sec), round(x[1]*sixth_step_per_sec), (x[2]-c0_midi_note)%12, x[3]) for x in midi_list]
    
    max_steps = 17*4*4*6
    highest_note = 11

    chords_array = np.zeros((max_steps, highest_note+1, 2))
    for x in midi_simple:
        chords_array[x[0]:x[1],x[2],x[3]]=1
        
    return midi_simple, chords_array

# complete_list=[]
# with open('song_list.csv', newline='', mode='r', encoding='utf-8-sig') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if color_assign(row[0],row[1]):
#             complete_list.append(row)

complete_list = [['Beyoncé', '6 Inch', '138', 'A#', 'minor'],
 ['The Weeknd', 'All I Know', '120', 'G#', 'minor'],
 ['The Weeknd', 'Love Me Harder', '99', 'C#', 'minor'],
 ['Beyoncé', 'Halo', '80', 'A', 'major'],
 ['Beyoncé', 'Blow', '120', 'G', 'minor'],
 ['Doja Cat', 'Streets', '90', 'E', 'minor'],
 ['Dua Lipa', 'New Rules', '116', 'A', 'minor'],
 ['Dua Lipa', 'Break My Heart', '113', 'E', 'minor'],
 ['Dua Lipa', 'Levitating', '103', 'B', 'minor'],
 ['Rihanna', 'Consideration', '145', 'G', 'minor']]

complete_list

# split=[0.6, 0.2, 0.2]
# num_train = int(len(complete_list)*split[0])
# num_val = int(len(complete_list)*split[1])
# num_test = len(complete_list) - num_train - num_val

# random.shuffle(complete_list)
# train_set = complete_list[:num_train]
# val_set = complete_list[num_train:(num_train+num_val)]
# test_set = complete_list[(num_train+num_val):]

train_set = [['Beyoncé', '6 Inch', '138', 'A#', 'minor'],
['Beyoncé', 'Blow', '120', 'G', 'minor'],
['The Weeknd', 'Love Me Harder', '99', 'C#', 'minor'],
['Rihanna', 'Consideration', '145', 'G', 'minor'],
['Dua Lipa', 'Levitating', '103', 'B', 'minor'],
['Beyoncé', 'Halo', '80', 'A', 'major']]

val_set = [['Dua Lipa', 'New Rules', '116', 'A', 'minor'],
['The Weeknd', 'All I Know', '120', 'G#', 'minor']]

test_set = [['Dua Lipa', 'Break My Heart', '113', 'E', 'minor'],
['Doja Cat', 'Streets', '90', 'E', 'minor']]

# print("num_train:", num_train, "num_val:", num_val, "num_test:", num_test)
[print(x) for x in train_set];
print("-------------------------")
[print(x) for x in val_set];
print("-------------------------")
[print(x) for x in test_set];

def get_syllables(syll_file, section):
    if section=='prechorus':
        section='pre_chorus'
    stanza = pickle5.load( open( syll_file, "rb" ) )[section]
    stanza = stanza.replace('0', '')
    stanza = stanza.replace('1', '')
    stanza = stanza.replace('2', '')
    stanza = stanza.replace('3', '')
    stanza = stanza.replace('4', '')
    stanza = stanza.replace('5', '')

    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

    consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
                  , "V", "W", "Y", "Z", "ZH"]

    syllables = []

    for line in stanza.split('\r\n'):
        for word in line.split('  '):
    #         print(word)
            vowel_list=[]
            for phoneme in word.split(' '):
                if phoneme in vowels:
                    vowel_list.append(phoneme)
                elif phoneme not in consonants:
                    raise Exception("uh oh weird phoneme: ", phoneme, 'in line: ', stanza)
            if len(vowel_list)==1:
                syllables.append(word)
            else:
                temp_word=word
                for i in range(len(vowel_list)):
                    if i==(len(vowel_list)-2):
                        syllables.append(temp_word.split(vowel_list[i], 1)[0] + vowel_list[i])
                        syllables.append(temp_word.split(vowel_list[i] + " ", 1)[1])
                        break
    #                 print(vowel_list[i])
                    syllables.append(temp_word.split(vowel_list[i], 1)[0] + vowel_list[i])
                    temp_word = temp_word.split(vowel_list[i]+" ", 1)[1]
    return syllables

def process_vocals(vocals_json, syll_file, section, pitch_shift):
    
    with open(vocals_json) as f:
        dic = json.load(f)

    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

    consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH"
          , "V", "W", "Y", "Z", "ZH"]

    ref_syll_vec = vowels+consonants

    input_syll_vec = np.zeros((num_syllables, num_pieces_per_syllable, len(ref_syll_vec)))
    for i,syll in enumerate(get_syllables(syll_file, section)):
        temp = []
        for j,part in enumerate(syll.split(' ')):
            input_syll_vec[i,j,[z for z,x in enumerate(ref_syll_vec) if x==part]]=1

    max_steps = 17*4*4*6
    note_range = note_name_to_number("C7")-note_name_to_number("C2")+1

    vocals_array = np.zeros((max_steps, note_range, num_syllables))

    z=0
    for note in dic:
        note_start = note['start']
        note_end = note['start'] + note['duration'] + 1
        vocals_array[note_start:note_end, note['pitch']+pitch_shift, z]=1
        if note['syllable']!='-':
            z+=1
    return vocals_array, input_syll_vec

def max_samples(sample_rate, bpm):
    return sample_rate*17*4*60/bpm

def frame_step(sample_rate, bpm):
    bps=bpm/60
    steps_per_sec=bps*4
    sixth_step_per_sec=steps_per_sec*6
    sec_per_sixth_step=(1/sixth_step_per_sec)
    samples_per_sixth_step=sample_rate*sec_per_sixth_step

    return samples_per_sixth_step/2

def remainder(x):
    return x-int(x)

def key_to_array(key, pitch_shift):
    key_array=np.zeros((12,2))
    key_number=note_name_to_number(key[0]+'0')+pitch_shift
    q=(key_number-note_name_to_number("C0"))%12
    if key[1].lower()=='minor':
        key_array[q,1]=1
    elif key[1].lower()=='major':
        key_array[q,0]=1
    else:
        raise Exception("key naming error")
    return key_array

def bpm_to_array(bpm, bpm_shift):
    max_bpm=200
    min_bpm=50
    bpm_array=np.zeros(max_bpm-min_bpm)
    bpm_array[bpm+bpm_shift-min_bpm]=1
    return bpm_array

def wav_to_spec(nfilt, track, wav, pitch_shift, bpm_shift):

    bpm_table = np.load('bpm_table/bpm_table.npy')
    artist = track[0]
    song = track[1]
    bpm = int(track[2])
    key = [track[3], track[4]]
    
    if bpm_shift!=0:
        signal, sample_rate = librosa.load(wav)
        shift_fraction=(bpm+bpm_shift)/bpm
        signal_shifted = librosa.effects.time_stretch(signal, shift_fraction)
        ! rm -f 'temp.wav'
        sf.write('temp.wav', signal_shifted, sample_rate, subtype='PCM_24')
        bpm = bpm + bpm_shift
        i = np.where(bpm_table[:,0]==bpm)[0][0]
        NFFT = int(bpm_table[i][1])
        sample_rate = int(bpm_table[i][2])
        hop = int(bpm_table[i][3])
        signal, sample_rate = librosa.load('temp.wav', sr=sample_rate)
    else:
        i = np.where(bpm_table[:,0]==bpm)[0][0]
        NFFT = int(bpm_table[i][1])
        sample_rate = int(bpm_table[i][2])
        hop = int(bpm_table[i][3])
        signal, sample_rate = librosa.load(wav, sr=sample_rate)

    if pitch_shift!=0:
        signal = librosa.effects.pitch_shift(signal, sample_rate, pitch_shift)


    if signal.shape[0]>max_samples(sample_rate, bpm):
        signal = signal[:int(max_samples(sample_rate, bpm))]
    elif signal.shape[0]<max_samples(sample_rate, bpm):
        signal = np.append(signal, np.zeros(int(max_samples(sample_rate, bpm)) - signal.shape[0]))

#     sf.write('/Users/agp2004/Desktop/test.wav', signal, sample_rate, subtype='PCM_24')
    
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=nfilt, n_fft=NFFT, hop_length=hop, center=False)

    return S

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def upload_file(filename, file_loc, dir_id):
  uploaded = drive.CreateFile({'title': filename, 'parents': [{'id': dir_id}]})
  uploaded.SetContentFile(file_loc)
  uploaded.Upload()
  # print('Uploaded file with ID {}'.format(uploaded.get('id')))

# ! rm -rf data
# ! mkdir data
# ! mkdir data/train
# ! mkdir data/val
# ! mkdir data/test

from google.colab import files

uploaded = files.upload() # upload clients_secret.json

!mv client_secret_529386604270-3eelg2ggsjf47hgbj6ugb5d1ebnccnbl.apps.googleusercontent.com.json client_secrets.json

num_syllables = 200
num_pieces_per_syllable = 7
nfilt = 120

for track in train_set:
    for bpm_shift in np.arange(-10, 11):
        for pitch_shift in np.arange(-6, 7):
            artist = track[0]
            song = track[1]
            bpm = int(track[2])
            key = [track[3], track[4]]
            
            syll_file = "saved_pronunciations/"+artist+"_"+song.replace('/', '-')+".pkl"
            
            if bpm_shift<0:
                bpm_shift_name = 'neg' + str(abs(bpm_shift))
            else:
                bpm_shift_name = str(abs(bpm_shift))
            
            if pitch_shift<0:
                pitch_shift_name = 'neg' + str(abs(pitch_shift))
            else:
                pitch_shift_name = str(abs(pitch_shift))
                
            
            file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
            loaded_dictionary = pickle5.load(file_to_read)

            for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:

                check_file = 'data/train/' + artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'

                if os.path.exists(check_file):
                  continue

                if len(loaded_dictionary[x])>2:
                    if x=='verse1':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/verse1.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse1.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/verse1_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/verse1_vocals.wav'

                    elif x=='pre_chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/prechorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_prechorus.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/prechorus_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/prechorus_vocals.wav'

                    elif x=='chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/chorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/chorus_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/chorus_vocals.wav'

                    elif x=='post_chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/post_chorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_post.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/post_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/post_vocals.wav'

                    elif x=='verse2':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/verse2.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse2.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/verse2_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/verse2_vocals.wav'
                  
                        
                    output_array, input_syll_vec = process_vocals(vocals_json, syll_file, x, pitch_shift)
                    vox_input = wav_to_spec(nfilt, track, vox_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
                    acc_input = wav_to_spec(nfilt, track, acc_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
                    midi_simple, chords_array = chords_process(chords_file, bpm, pitch_shift=pitch_shift)
                    key_array = key_to_array(key, pitch_shift)
                    bpm_array = bpm_to_array(bpm, bpm_shift)
                    
                    outfile = artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'
                    print(outfile)
                    np.savez_compressed('/content/temp/temp.npz', output_array=output_array, input_syll_vec=input_syll_vec, vox_input=vox_input, acc_input=acc_input, chords_array=chords_array, key_array=key_array, bpm_array=bpm_array)
                    upload_file(outfile, '/content/temp/temp.npz', '1-1qyo35BBB6jmD2IxqY1rnN4UbLKB7y4')
                    !rm /content/temp/temp.npz
                    

!ls /content/drive/MyDrive/integrated/data/train/*.npz | wc -l

num_syllables = 200
num_pieces_per_syllable = 7
nfilt = 120

for track in val_set:
    for bpm_shift in np.arange(-2, 3):
        for pitch_shift in np.arange(-2, 3):
            artist = track[0]
            song = track[1]
            bpm = int(track[2])
            key = [track[3], track[4]]
            
            syll_file = "saved_pronunciations/"+artist+"_"+song.replace('/', '-')+".pkl"
            
            if bpm_shift<0:
                bpm_shift_name = 'neg' + str(abs(bpm_shift))
            else:
                bpm_shift_name = str(abs(bpm_shift))
            
            if pitch_shift<0:
                pitch_shift_name = 'neg' + str(abs(pitch_shift))
            else:
                pitch_shift_name = str(abs(pitch_shift))
                
            
            file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
            loaded_dictionary = pickle5.load(file_to_read)

            for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:

                check_file = 'data/val/' + artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'

                if os.path.exists(check_file):
                  continue

                if len(loaded_dictionary[x])>2:
                    if x=='verse1':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/verse1.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse1.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/verse1_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/verse1_vocals.wav'

                    elif x=='pre_chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/prechorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_prechorus.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/prechorus_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/prechorus_vocals.wav'

                    elif x=='chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/chorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/chorus_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/chorus_vocals.wav'

                    elif x=='post_chorus':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/post_chorus.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_post.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/post_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/post_vocals.wav'

                    elif x=='verse2':
                        vocals_json = 'midi_data/' + artist + '_' + song + '/verse2.json'
                        chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse2.mid'
                        acc_wav = 'midi_data/' + artist + '_' + song + '/verse2_acc.wav'
                        vox_wav = 'midi_data/' + artist + '_' + song + '/verse2_vocals.wav'
                  
                        
                    output_array, input_syll_vec = process_vocals(vocals_json, syll_file, x, pitch_shift)
                    vox_input = wav_to_spec(nfilt, track, vox_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
                    acc_input = wav_to_spec(nfilt, track, acc_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
                    midi_simple, chords_array = chords_process(chords_file, bpm, pitch_shift=pitch_shift)
                    key_array = key_to_array(key, pitch_shift)
                    bpm_array = bpm_to_array(bpm, bpm_shift)
                    
                    outfile = artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'
                    print(outfile)
                    np.savez_compressed('/content/temp/temp.npz', output_array=output_array, input_syll_vec=input_syll_vec, vox_input=vox_input, acc_input=acc_input, chords_array=chords_array, key_array=key_array, bpm_array=bpm_array)
                    upload_file(outfile, '/content/temp/temp.npz', '1-1Xn6JcIWMVRxmsoN2RqZVKc91q6sKHN')
                    !rm /content/temp/temp.npz
                    

!ls /content/drive/MyDrive/integrated/data/val/*.npz | wc -l

num_syllables = 200
num_pieces_per_syllable = 7
nfilt = 120

for track in test_set:
    bpm_shift=0
    pitch_shift=0

    artist = track[0]
    song = track[1]
    bpm = int(track[2])
    key = [track[3], track[4]]
    
    syll_file = "saved_pronunciations/"+artist+"_"+song.replace('/', '-')+".pkl"
    
    if bpm_shift<0:
        bpm_shift_name = 'neg' + str(abs(bpm_shift))
    else:
        bpm_shift_name = str(abs(bpm_shift))
    
    if pitch_shift<0:
        pitch_shift_name = 'neg' + str(abs(pitch_shift))
    else:
        pitch_shift_name = str(abs(pitch_shift))
        
    
    file_to_read = open("saved_lyrics/"+artist+"_"+song.replace('/', '-')+".pkl",'rb')
    loaded_dictionary = pickle5.load(file_to_read)

    for x in ["verse1", "pre_chorus", "chorus", "post_chorus", "verse2"]:

        check_file = 'data/train/' + artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'

        if os.path.exists(check_file):
          continue

        if len(loaded_dictionary[x])>2:
            if x=='verse1':
                vocals_json = 'midi_data/' + artist + '_' + song + '/verse1.json'
                chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse1.mid'
                acc_wav = 'midi_data/' + artist + '_' + song + '/verse1_acc.wav'
                vox_wav = 'midi_data/' + artist + '_' + song + '/verse1_vocals.wav'

            elif x=='pre_chorus':
                vocals_json = 'midi_data/' + artist + '_' + song + '/prechorus.json'
                chords_file = 'midi_data/' + artist + '_' + song + '/ch_prechorus.mid'
                acc_wav = 'midi_data/' + artist + '_' + song + '/prechorus_acc.wav'
                vox_wav = 'midi_data/' + artist + '_' + song + '/prechorus_vocals.wav'

            elif x=='chorus':
                vocals_json = 'midi_data/' + artist + '_' + song + '/chorus.json'
                chords_file = 'midi_data/' + artist + '_' + song + '/ch_chorus.mid'
                acc_wav = 'midi_data/' + artist + '_' + song + '/chorus_acc.wav'
                vox_wav = 'midi_data/' + artist + '_' + song + '/chorus_vocals.wav'

            elif x=='post_chorus':
                vocals_json = 'midi_data/' + artist + '_' + song + '/post_chorus.json'
                chords_file = 'midi_data/' + artist + '_' + song + '/ch_post.mid'
                acc_wav = 'midi_data/' + artist + '_' + song + '/post_acc.wav'
                vox_wav = 'midi_data/' + artist + '_' + song + '/post_vocals.wav'

            elif x=='verse2':
                vocals_json = 'midi_data/' + artist + '_' + song + '/verse2.json'
                chords_file = 'midi_data/' + artist + '_' + song + '/ch_verse2.mid'
                acc_wav = 'midi_data/' + artist + '_' + song + '/verse2_acc.wav'
                vox_wav = 'midi_data/' + artist + '_' + song + '/verse2_vocals.wav'
          
                
            output_array, input_syll_vec = process_vocals(vocals_json, syll_file, x, pitch_shift)
            vox_input = wav_to_spec(nfilt, track, vox_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
            acc_input = wav_to_spec(nfilt, track, acc_wav, pitch_shift=pitch_shift, bpm_shift=bpm_shift)
            midi_simple, chords_array = chords_process(chords_file, bpm, pitch_shift=pitch_shift)
            key_array = key_to_array(key, pitch_shift)
            bpm_array = bpm_to_array(bpm, bpm_shift)
            
            outfile = artist + "_" + song + '_' + pitch_shift_name + "_" + bpm_shift_name + "_" + x + '.npz'
            print(outfile)
            np.savez_compressed('/content/temp/temp.npz', output_array=output_array, input_syll_vec=input_syll_vec, vox_input=vox_input, acc_input=acc_input, chords_array=chords_array, key_array=key_array, bpm_array=bpm_array)
            upload_file(outfile, '/content/temp/temp.npz', '1-16Jj8FTAysBuig4OtVVJWrvq0mZPu4D')
            !rm /content/temp/temp.npz
                    

for file in np.load('/content/drive/MyDrive/integrated/data/test/' + outfile).files:
  print(file, np.load('/content/drive/MyDrive/integrated/data/test/' + outfile)[file].shape)
