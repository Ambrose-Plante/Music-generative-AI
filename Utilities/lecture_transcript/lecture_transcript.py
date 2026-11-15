from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/lecture_transcripts

# !pip install git+https://github.com/openai/whisper.git
# ! pip install python-Levenshtein
# ! pip install pdfplumber

import glob
import time
import os
import pickle
import numpy as np
# from Levenshtein import distance as levenshtein_distance

! ffmpeg -y -i "$filename" -map 0:a -acodec copy "audio2/$filename"

# ! whisper "audio2/temp_movie.mp3" --model small --language English
# ! rm ./*.json ./*.srt ./*.tsv ./*.vtt

filename = '2023-06-27_M5_Pentose_Phosphate_Pathway_Pike_AY_Edits.mp4'
text_file = filename.split('.mp4')[0] + '.txt'

! ffmpeg -y -i "$filename" -map 0:a -acodec copy "audio2/$filename"
! whisper "audio2/$filename" --model small --language English
! rm ./*.json ./*.srt ./*.tsv ./*.vtt
! mv $text_file transcripts/
