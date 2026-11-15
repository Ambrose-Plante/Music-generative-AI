from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/BnB_project/

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

# Open our new sheet and add some data.
gsheet_url = 'https://docs.google.com/spreadsheets/d/1tNKM7Gwq2s_YIQzXbPJgudkv0etkah5CosZy-br2z3Q/edit#gid=0' ## Noah
worksheet1 = gc.open_by_url(gsheet_url).sheet1
worksheet2 = gc.open_by_url(gsheet_url).get_worksheet(1)

# import gensim.models.keyedvectors as word2vec

# model = word2vec.KeyedVectors.load_word2vec_format('biowordvec.vec.bin', binary=True)

# !pip install git+https://github.com/openai/whisper.git
! pip install python-Levenshtein
! pip install pdfplumber

# ! apt install tesseract-ocr
# ! apt install libtesseract-dev

# ! pip install Pillow
# ! pip install pytesseract

import glob
import time
import os
import pickle
import numpy as np
from Levenshtein import distance as levenshtein_distance
# import pytesseract

# transcript_dict = {}
# file_list = np.unique(glob.glob('../Boards and Beyond STEP 2/**/*.m*', recursive=True))
# tot_num_files = len(file_list)

# for i,filename in enumerate(file_list):
#     print(i,'/',tot_num_files)

#     out_name = filename.replace(' ', '_').split('../Boards_and_Beyond_STEP_2/')[1]
#     out_name = out_name.replace('/', '___')
#     if '.mkv' in out_name:
#       text_file = out_name.split('.mkv')[0] + '.txt'
#     elif '.mp4' in out_name:
#       text_file = out_name.split('.mp4')[0] + '.txt'

#     if os.path.exists('transcripts_step2/'+text_file):
#       continue
#     else:
#       ! ffmpeg -y -i "$filename" -map 0:a -acodec copy "audio2/$out_name"
#       time.sleep(10)
#       ! whisper "audio2/$out_name" --model small --language English
#       time.sleep(10)

#       with open(text_file) as f:
#           lines = f.readlines()

#       transcript_dict[out_name] = lines

#       ! mv $text_file transcripts_step2/
#       ! rm ./*.json ./*.srt ./*.tsv ./*.vtt
#       time.sleep(5)

# # with open('transcript_dict.pickle', 'wb') as handle:
# #     pickle.dump(transcript_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


transcript_dict = {}
for filename in glob.glob('transcripts/*.txt'):
  video_name = filename.split('transcripts/')[1].split('.txt')[0]

  with open(filename) as f:
      lines = f.readlines()

  transcript_dict[video_name] = lines

for filename in glob.glob('transcripts_step2/*.txt'):
  video_name = filename.split('transcripts_step2/')[1].split('.txt')[0]

  with open(filename) as f:
      lines = f.readlines()

  transcript_dict[video_name] = lines

# for i,filename in enumerate(glob.glob('../Boards and Beyond/**/*.mp4', recursive=True)):

#     out_name = filename.replace(' ', '_').split('../Boards_and_Beyond/')[1]
#     out_name = out_name.replace('/', '___')
#     text_file = out_name.split('.mp4')[0] + '.txt'
#     if not os.path.exists('transcripts/'+text_file):
#       print(filename)

# query = 'carcinoid'
# query = query.lower()
# len(query)

# query_matches = []
# for video_name in transcript_dict.keys():
#   for sentance in ' '.join(transcript_dict[video_name]).replace('\n', '').split('. '):
#       for char in [',','?','!']:
#         sentance.replace(char,'')

#       for word in sentance.split(' '):

#         if len(word)>=12:
#           lev_dist_tolerance=2
#         elif len(word)>=9:
#           lev_dist_tolerance=1
#         elif len(word)>=6:
#           lev_dist_tolerance=0
#         else:
#           lev_dist_tolerance=0

#         if levenshtein_distance(query, word.lower())<=lev_dist_tolerance:
#           # print(sentance)
#           query_matches.append([video_name, sentance])
#           break

# query_matches

with open('unigram_freq.csv') as f:
    lines = f.readlines()
    
filter_words=[]
freaks = []
for x in lines:
  temp_word = x.split(',')[0]
  try:
    temp_freq = int(x.split(',')[1].replace('\n', ''))
  except:
    continue
  freaks.append(temp_freq)
  if temp_freq>14840226:
    filter_words.append(temp_word.lower())

import pdfplumber
from tqdm import tqdm
filter_chars = ['.', ',', '(', ')', '?', ':', ';', '!',']','[']
space_chars = ['/', '-', '\n', '‐']
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']

for lec_ind,lecture in enumerate(sorted(glob.glob('./lectures/Week 7 pdf/*.pdf'))):

  lecture_name = lecture.split('/')[-1].split('.pdf')[0]
  print(lecture_name)

  temp_worksheet = gc.open_by_url(gsheet_url).add_worksheet(title=lecture_name, rows="100", cols="100")

  col_name = alphabet[lec_ind]+'1'
  worksheet1.update(col_name, lecture_name)
  lecture_text = []
  with pdfplumber.open(lecture) as pdf:
      for page in pdf.pages:
          temp_text = page.extract_text()

          for char in filter_chars:
            temp_text = temp_text.replace(char, '')

          for char in space_chars:
            temp_text = temp_text.replace(char, ' ')

          for word in temp_text.split(' '):
            word = word.lower()

            if len(word)>=5:
              if word not in filter_words:
                  lecture_text.append(word)

  print(len(lecture_text), 'words fouund')

  video_match_dict = {}
  hit_list_dict = {}
  len_vids = len(transcript_dict.keys())
  for q,video_name in enumerate(tqdm(transcript_dict.keys())):
    # print(q,'/',len_vids)
    query_dict = {}
    hit_list=[]
    for query in np.unique(lecture_text):
      query_dict[query]=0
      banana=0
      for sentance in ' '.join(transcript_dict[video_name]).replace('\n', '').split('. '):
        if not banana:

          for char in [',','?','!']:
            sentance.replace(char,'')

          for word in sentance.split(' '):

            if len(word)>=12:
              lev_dist_tolerance=2
            elif len(word)>=9:
              lev_dist_tolerance=1
            elif len(word)>=6:
              lev_dist_tolerance=0
            else:
              lev_dist_tolerance=0

            if levenshtein_distance(query, word.lower())<=lev_dist_tolerance:
              # print(query, '--', sentance)
              query_dict[query]=1
              hit_list.append(query)
              banana=1
              break
    video_match_dict[video_name] = 100*sum([query_dict[x] for x in query_dict.keys()])/len(query_dict)
    hit_list_dict[video_name] = hit_list


  for item_ind,item in enumerate(sorted(video_match_dict.items(), key=lambda x:x[1], reverse=True)[:30]):
    row_name = alphabet[lec_ind]+str(item_ind+2)
    worksheet1.update(row_name, '%'+str(round(item[1]))+' -- '+item[0])
    temp_worksheet.update(alphabet[item_ind]+'1', '%'+str(round(item[1]))+' -- '+item[0])
    temp_worksheet.update(alphabet[item_ind]+'2', ','.join(hit_list_dict[item[0]]))
    # for q,hit in enumerate(hit_list_dict[item[0]]):
    #   hit_row_name = alphabet[item_ind]+str(q+2)
    #   temp_worksheet.update(hit_row_name, hit)

import time

flat_list = []
for i in np.arange(1,len(worksheet1.row_values(1))+2): 
  flat_list += worksheet1.col_values(i)[:11] # only count top ten BnB matches for each lecture
 
video_match_dict = {}
for video_name in transcript_dict.keys():
  video_match_dict[video_name] = len([x.split(' -- ')[-1] for x in flat_list if x.split(' -- ')[-1]==video_name])

worksheet2.update('A1', 'BnB video')
worksheet2.update('B1', 'number of top ten lecture matches')

for item_ind,item in enumerate(sorted(video_match_dict.items(), key=lambda x:x[1], reverse=True)):
  if item_ind!=0 and item_ind%100==0: ## write-limit per minute of 300
    time.sleep(60)
  worksheet2.update('A'+str(item_ind+2), item[0])
  worksheet2.update('B'+str(item_ind+2), item[1])

flat_list

# for item in sorted(video_match_dict.items(), key=lambda x:x[1], reverse=True)[:50]:
#   print('%'+str(round(item[1], 1)), '--', item)
#   print(item[1])
#   print('---------------------------------------------------------------------------')

video_name = 'Cardiology___Cardiology___08_Hypertension___03_Hypertension_Drugs'
for query in lecture_text:
  banana=0
  for sentance in ' '.join(transcript_dict[video_name]).replace('\n', '').split('. '):
    if not banana:
      for i in range(len(sentance)-len(query)):
        if levenshtein_distance(query, sentance[i:(i+len(query))].lower())<=0:
          print(query, '--', sentance)
          banana=1
          break
