from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/music_app

! pip install pickle5

import nltk
from nltk.corpus import cmudict
import glob
import pickle5
nltk.download('cmudict')

from functions.functions import clean, lite_clean
arpabet = cmudict.dict()

for filename in glob.glob('lyrics_data/archive/*'):
  save_filename = 'lyrics_data/cleaned_lyrics/' + filename.split('/')[-1]

  f = open(filename)

  cleaned_lines = []
  for line in f.readlines():
    line = line.replace('\n', '')
    line = clean(line)
    line = lite_clean(line)
    line = line.lower()
    good_words = []
    for word in line.split(' '):
      if word in arpabet:
        good_words.append(word)
    if len(good_words)>2:
      cleaned_lines.append(' '.join(good_words))

  with open(save_filename, 'wb') as saveeee_filename:
    pickle5.dump(cleaned_lines, saveeee_filename)

import glob
import random
lyrics = []
for filename in glob.glob('lyrics_data/archive/*'):
  save_filename = 'lyrics_data/cleaned_lyrics/' + filename.split('/')[-1]

  f = open(filename)

  cleaned_lines = []
  for line in f.readlines():
    line = line.replace('\n', '')
    if line!='':
      lyrics.append(line)

song = []
for i in range(100):

random.sample(lyrics, 50)
