# Untitled1

## Overview

This notebook is an integration prototype joining chord-MIDI parsing, trained generator inference, and output rendering. Chord MIDI notes that overlap in time are grouped into note sets, compared against chord templates, and reduced to roots or scale-degree representations when an exact template is unavailable. Those chord arrays, song-section codes, and BPM conditions can be supplied to a trained vocal generator. The generated vocal probabilities are converted back into note events and written to MIDI, with utilities for chords and metronome tracks and subsequent audio synthesis. Rather than introducing one new learning algorithm, the notebook assembles several earlier algorithms into an end-to-end generation workflow.

## Algorithmic workflow

1. Parse/group MIDI chords
2. Map to scale-degree timeline
3. Form section/BPM conditions
4. Run generator
5. Decode vocal array
6. Render MIDI/audio
