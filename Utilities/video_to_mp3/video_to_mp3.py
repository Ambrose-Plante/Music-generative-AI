from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/MED_ED/GYN

# !pip install git+https://github.com/openai/whisper.git
# ! pip install python-Levenshtein
# ! pip install pdfplumber

import glob
import time
import os
import pickle
import numpy as np
# from Levenshtein import distance as levenshtein_distance

import re

# Define a function to extract numeric part from the string
def extract_number_and_text(s):
    # Use regular expression to find the numeric part and the rest of the string
    match = re.match(r'(\d+)\. (.+)', s)
    if match:
        return int(match.group(1)), match.group(2)
    else:
        return float('inf'), s  # If no numeric part found, put it at the end

# Sort the list using the custom function
sorted_list = sorted(glob.glob('*'), key=extract_number_and_text)
print(sorted_list)

# ! whisper "audio2/temp_movie.mp3" --model small --language English
# ! rm ./*.json ./*.srt ./*.tsv ./*.vtt

os.chdir('/content/drive/My Drive/MED_ED')

!ls

# os.mkdir('GYN_audio')

for i in range(len(sorted_list)):
  filename = 'GYN/' + sorted_list[i]
  outfile = 'GYN_audio/' + str(i).zfill(3) + '.mp3'
  # ! ffmpeg -y -i "$filename" -map 0:a -acodec pcm_s16le "$outfile" ## wav
  ! ffmpeg -y -i "$filename" -map 0:a -acodec libmp3lame "$outfile" ## mp3

os.chdir('GYN_audio/')

!pip install pydub

# import wave

# infiles = sorted(glob.glob('*'))
# outfile = "PEDs_combined_OME_audio.wav"

# with wave.open(outfile, 'wb') as wav_out:
#     for wav_path in infiles:
#         with wave.open(wav_path, 'rb') as wav_in:
#             if not wav_out.getnframes():
#                 wav_out.setparams(wav_in.getparams())
#             wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

import os
from pydub import AudioSegment

# Get the list of MP3 files
mp3_files = sorted(glob.glob('*'))

# Create an empty AudioSegment object
merged_audio = AudioSegment.empty()

# Iterate through the list of MP3 files and append them to the merged_audio object
for mp3_file in mp3_files:
    merged_audio += AudioSegment.from_file(mp3_file)

# Export the merged audio to a new MP3 file
merged_audio.export("GYN_combined_OME_audio.mp3", format="mp3")

!ls

filename = '2023-06-27_M5_Pentose_Phosphate_Pathway_Pike_AY_Edits.mp4'
text_file = filename.split('.mp4')[0] + '.txt'

! ffmpeg -y -i "$filename" -map 0:a -acodec copy "audio2/$filename"
! whisper "audio2/$filename" --model small --language English
! rm ./*.json ./*.srt ./*.tsv ./*.vtt
! mv $text_file transcripts/
