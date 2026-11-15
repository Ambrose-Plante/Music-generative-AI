# lecture_transcript

## Overview

This notebook automates transcription of lecture-video files. FFmpeg is used to extract an audio stream from video into an audio format suitable for transcription, after which the OpenAI Whisper command-line tool is invoked. Whisper's official repository documents this direct audio-file transcription workflow. 10 The notebook removes unwanted sidecar output formats and moves the plain-text transcript into a designated transcript directory. Its algorithm is therefore a batch media-conversion and speech-recognition pipeline rather than a custom speech model.

## Algorithmic workflow

1. Enumerate videos
2. FFmpeg audio extraction
3. Invoke Whisper
4. Retain text output
5. Clean sidecars
6. Move transcript files

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `lecture_transcript.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
