!pip install genanki
!pip install pydub
!pip install --upgrade gspread oauth2client

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default

creds, _ = default()
gc = gspread.authorize(creds)

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Mandarin_anki

import genanki
import random
from pydub import AudioSegment
import pandas as pd
import os
import csv

row_start = 1
row_end = 76
mp3 = 'jocelyn_recordings/mandarin_02_09_25.mp3'
bpm = 60
beats_per_phrase = 4
savefile_name = "mandarin_english_deck_" + str(row_start) + '_' + str(row_end) + '.apkg'
deck_name = "mandarin_english_deck_" + str(row_start) + '_' + str(row_end)

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1-5xfs9tpOPD5cwE1EEOvrEVLgwPwuqUnplBMb2Oz984/edit?gid=0#gid=0')
phrases = gsheets.worksheet('Sheet1').get_all_values()

# Create a DataFrame with one column: "Phrase" (English text)
df = pd.DataFrame([x[0] for x in phrases], columns=["Phrase"]).iloc[row_start-1:row_end]
num_phrases = row_end - row_start + 1

# Load the audio file
audio = AudioSegment.from_mp3(mp3)

# Calculate chunk length based on beats per minute
ms_per_beat = (60 / bpm) * 1000
chunk_length_ms = beats_per_phrase * ms_per_beat

# Create a directory to save the audio chunks
audio_dir = "audio_dir"
os.makedirs(audio_dir, exist_ok=True)

# Clear the directory if it already contains files
for file in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file)
    if os.path.isfile(file_path):
        os.unlink(file_path)

# Extract audio chunks
audio_files = []
for i in range(num_phrases):
    start_time = i * chunk_length_ms
    end_time = start_time + chunk_length_ms
    chunk = audio[start_time:end_time]
    file_name = os.path.join(audio_dir, f"chunk_{i}.mp3")
    chunk.export(file_name, format="mp3")
    audio_files.append(file_name)

# Ensure we have the same number of phrases and audio chunks
if not len(audio_files) == len(df):
    raise Exception("The number of audio chunks must match the number of phrases.")

def create_audio_anki_deck(df, audio_files, deck_name="Mandarin-English Deck"):
    # Create a unique model ID
    model_id = random.randrange(1 << 30, 1 << 31)

    # Define the model with TWO fields
    my_model = genanki.Model(
        model_id,
        'Audio Model',
        fields=[
            {'name': 'English'},
            {'name': 'Audio'},
        ],
        templates=[
            {
                # Card 1: English → Mandarin Audio
                'name': 'English Prompt - Mandarin Audio',
                'qfmt': '''
                <div style="font-size: 24px;">
                  {{English}}
                </div>
                ''',
                'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                {{Audio}}
                ''',
            },
            {
                # Card 2: Mandarin Audio → English
                'name': 'Mandarin Audio - English Prompt',
                'qfmt': '''
                {{Audio}}
                ''',
                'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div style="font-size: 24px;">
                  {{English}}
                </div>
                ''',
            },
        ],
        css="""
        .card {
          font-family: arial;
          font-size: 24px;
          text-align: left;
          color: black;
          background-color: white;
        }
        """
    )

    # Create the deck
    deck_id = random.randrange(1 << 30, 1 << 31)
    my_deck = genanki.Deck(deck_id, deck_name)

    # Create ONE note per phrase
    for i in range(len(df)):
        english_text = df.iloc[i]["Phrase"]
        audio_file = audio_files[i]
        audio_filename = os.path.basename(audio_file)

        # The fields for each note:
        #   English = The English phrase
        #   Audio   = [sound:chunk_i.mp3]
        note = genanki.Note(
            model=my_model,
            fields=[
                english_text,                 # Field: English
                f"[sound:{audio_filename}]"   # Field: Audio
            ]
        )

        my_deck.add_note(note)

    return my_deck

def save_deck_to_file(deck, file_name="mandarin_english_deck.apkg", media_files=None):
    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files
    package.write_to_file(file_name)
    print(f"Deck saved to {file_name}")

# Create the Anki deck
my_deck = create_audio_anki_deck(df, audio_files, deck_name=deck_name)

# Save the deck to a file, including media
save_deck_to_file(my_deck, file_name=savefile_name, media_files=audio_files)

audio_files

! ls audio_dir/chunk_0.mp3

os.mkdir('audio_dir')
