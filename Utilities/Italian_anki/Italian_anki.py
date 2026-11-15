!pip install genanki
!pip install pydub
!pip install --upgrade gspread oauth2client
!pip install --upgrade google-cloud-texttospeech
! pip install openai==0.27.0

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default
from google.cloud import texttospeech
import glob
import os
import random
import time

creds, _ = default()
gc = gspread.authorize(creds)

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Italian_anki

# Make sure your Google Cloud JSON credentials are set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Create a single global client
tts_client = texttospeech.TextToSpeechClient()

def italian_text_to_speech(text, output_filename, voice="it-IT-Standard-A", speaking_rate=0.7):
    """Generate an MP3 in Italian for the given text using Google Cloud TTS."""
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Voice parameters for Italian
    voice = texttospeech.VoiceSelectionParams(
        language_code="it-IT",
        name=voice,  # Or another valid Italian voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Audio configuration (MP3 + adjustable speaking rate)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate
    )

    # Perform synthesis
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Save output
    with open(output_filename, 'wb') as out:
        out.write(response.audio_content)

import os
import openai
import requests

def generate_dalle_image(prompt, filename="generated_image.png", size="512x512"):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    """
    Generates an image using OpenAI's DALL·E API and saves it locally.

    :param prompt: Text prompt describing the image you want to generate.
    :param filename: Name of the image file to save locally (PNG or JPG).
    :param size: The size of the generated image. Options: '256x256', '512x512', '1024x1024'.
    """
    try:
        # print(f"Generating image for prompt: '{prompt}'")

        # Call the OpenAI Image API
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=1,            # number of images to generate
            size=size,      # image dimensions
            quality="hd",
        )

        # Extract the URL from the response
        image_url = response["data"][0]["url"]
        # print(f"Image URL: {image_url}")

        # Download the image
        img_data = requests.get(image_url).content

        # Save the image
        with open(filename, "wb") as handler:
            handler.write(img_data)

        # print(f"Image saved to: {filename}")

    except Exception as e:
        print(f"Error generating image: {e}")

voices = tts_client.list_voices()
italian_voices = []
for voice in voices.voices:
    for language_code in voice.language_codes:
      if "it-IT" in language_code:
          print(f"Name: {voice.name}, Language code: {language_code}")
          italian_voices.append(voice.name)

random.choice(italian_voices)

import gspread
import pandas as pd
from google.cloud import texttospeech
import genanki
import random
import os

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/14EGNL6GlQwqYOM-uZmv9hY4227MlOq7TlhW7I-qxTRw/edit?gid=0#gid=0')
rows = gsheets.worksheet('Sheet1').get_all_values()


# Convert to a DataFrame with explicit column names
df_all = pd.DataFrame(rows, columns=['English', 'Italian'])

# Set the range of rows you want
row_start = 324
row_end = 459
df = df_all.iloc[row_start-1:row_end].reset_index(drop=True)
num_phrases = len(df)

print(df.head())  # Sanity check

# Directory where we’ll store the TTS audio
audio_dir = "audio_dir"
img_dir = "img_dir"

# Clear the directory first if you want a clean slate
for file in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file)
    if os.path.isfile(file_path):
        os.unlink(file_path)

# # Clear the directory first if you want a clean slate
# for file in os.listdir(img_dir):
#     file_path = os.path.join(img_dir, file)
#     if os.path.isfile(file_path):
#         os.unlink(file_path)

audio_files = []
for i, row in df.iterrows():
    english_text = row["English"]
    italian_text = row["Italian"]

    # Build a filename for each row
    file_name = os.path.join(audio_dir, f"italian_{i}.mp3")

    # Generate TTS in Italian
    italian_text_to_speech(text=italian_text, output_filename=file_name, speaking_rate=0.7)

    if os.path.exists(file_name):
        audio_files.append(file_name)

# Make sure we have the same number of MP3s as rows in df
assert len(audio_files) == num_phrases, "Mismatch in number of audio files and phrases."

# img_files = []
# for i, row in df.iterrows():
#     english_text = row["English"]
#     italian_text = row["Italian"]

#     # Build a filename for each row
#     file_name = os.path.join(img_dir, f"italian_{i}.png")

#     # Generate TTS in Italian
#     user_prompt = "generate a studio ghibli style image of a scene: \"" + english_text + '\"'
#     generate_dalle_image(prompt=user_prompt, filename=file_name, size="1024x1024")
#     img_files.append(file_name)
#     time.sleep(60)

# # Make sure we have the same number of MP3s as rows in df
# assert len(img_files) == num_phrases, "Mismatch in number of image files and phrases."

import genanki
import pandas as pd
import os
import random

def create_audio_anki_deck(df, audio_files, deck_name="Italian-English Deck"):
    # Create a unique model ID
    model_id = random.randrange(1 << 30, 1 << 31)

    # Define the Model with THREE fields
    my_model = genanki.Model(
        model_id,
        'Audio Model',
        fields=[
            {'name': 'English'},
            {'name': 'Italian'},
            {'name': 'Audio'},
        ],
        templates=[
            {
                # Card 1: English -> Italian
                'name': 'English -> Italian',
                # Front uses English only
                'qfmt': '''
                <div style="font-size: 24px;">
                  {{English}}
                </div>
                ''',
                # Back: repeat the front (English) plus the audio and Italian
                'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                {{Audio}}<br>
                <div style="font-size: 24px;">
                  {{Italian}}
                </div>
                ''',
            },
            {
                # Card 2: Italian (Audio) -> English
                'name': 'Italian -> English',
                # Front uses Audio only
                'qfmt': '''
                {{Audio}}
                ''',
                # Back: show the Italian text and the English text
                'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div style="font-size: 24px;">
                  {{Italian}}<br>
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

    # For each row in df, add ONE note with three fields
    for i in range(len(df)):
        english_text = df.iloc[i]["English"]
        italian_text = df.iloc[i]["Italian"]
        audio_file = audio_files[i]
        audio_filename = os.path.basename(audio_file)

        # Field 1: English
        # Field 2: Italian
        # Field 3: Audio (just the [sound:...] tag)
        note_fields = [
            english_text,
            italian_text,
            f"[sound:{audio_filename}]"
        ]

        note = genanki.Note(
            model=my_model,
            fields=note_fields
        )
        my_deck.add_note(note)

    return my_deck

def save_deck_to_file(deck, file_name="italian_english_deck.apkg", media_files=None):
    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files
    package.write_to_file(file_name)
    print(f"Deck saved to {file_name}")

deck_name = f"italian_english_deck_{row_start}_{row_end}"
savefile_name = f"{deck_name}.apkg"

my_deck = create_audio_anki_deck(df, audio_files, deck_name=deck_name)

save_deck_to_file(
    deck=my_deck,
    file_name=savefile_name,
    media_files=audio_files
)
