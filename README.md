# Maximus | Music Generative AI

Research code for **Maximus**, a research platform for generating symbolic vocal melodies from harmony, tempo, song structure, and lyrical context. This repository contains individual experiments and their algorithm descriptions, together with some unrelated computational research and utilities.

## Research pipeline

```text
Score PDFs → heuristic optical music recognition → MIDI
                                                ↘
MIDI + chords + BPM + song sections + lyrics
                      ↓
     normalization and feature encoding
                      ↓
  DCGAN / conditional GAN / Transformer / diffusion
                      ↓
       vocal pitch/probability arrays
                      ↓
       note-event decoding → MIDI → audio QC
```

**Research trajectory:** the models evolve from an unconditional MIDI DCGAN to chord conditioning, section/BPM control, dense-block variants with auxiliary losses, Transformer-based context, explicit timing objectives and diffusion/image-grid experiments. The separate optical-music-recognition pipeline uses score rasterization, OpenCV symbol-template matching, and geometric musical reconstruction.

## Browse the project

| Directory | Contents | Research notebooks |
| --- | --- | ---: |
| [`Generative_AI/`](Generative_AI/) | GANs, conditional GANs, diffusion, Transformer, supervised sequence, and architectural experiments for generating symbolic vocal melodies. | 37 |
| [`preprocessing/`](preprocessing/) | Converting raw MIDI, score-derived MIDI, lyrics, phonetic annotations, and song metadata into aligned, normalized model inputs. | 17 |
| [`Data_Generation/`](Data_Generation/) | Optical music recognition from score PDFs, rule-based chord voicing, and augmentation of symbolic-MIDI data. | 12 |
| [`postprocessing/`](postprocessing/) | Decoding model output arrays, generating MIDI note events, and synthesizing or mixing audio for qualitative evaluation. | 7 |
| [`spatial_transcriptomics/`](spatial_transcriptomics/) | Exploratory neural representation learning relating histology imagery to spatial gene expression. | 2 |
| [`Utilities/`](Utilities/) | Independent work on transcription, clinical ETL, language-learning decks, data handling, and smaller experiments. | 18 |

Each project folder contains a README documenting its algorithmic purpose and processing stages. The original names, including revision letters and `Copy of` experiments, are retained to preserve experimental lineage.

## Source status

**Documentation is published first; the remaining Python exports are staged in a separate import package.** The GitHub connector does not accept an uploaded ZIP as a filesystem source for bulk commits. Two empty historical placeholders (`Network_TPU` and `Untitled3`) are included; other archive scripts need to be transferred from the prepared import package. The two existing project scripts (`AutoVox_GAN` and `pianoRoll_to_midi`) remain in place.

- **Do not publish embedded secrets.** Source files `Italian_anki.py` and `Mandarin_anki_TTS.py` are intentionally excluded because the analytical report identified hard-coded credentials. Rotate/revoke and sanitize first.
- **Research snapshots, not turnkey packages.** The report documents code statically. Original datasets, Google Drive paths, historical dependencies, model checkpoints, and sound fonts may be required.
- **Reference material.** The annotated Transformer notebook is educational/reference code; check its original license and attribution.
- The earlier Maximus project description also referenced a Django lyrics-to-notes interface; that web app is not among the 93 notebooks analyzed here.

## Documentation provenance

Descriptions are grounded in the supplied 36-page *Analytical Report: Concepts and Algorithms in Colab Notebooks.zip*, covering 93 original Colab notebooks. This repository already contained the two completed examples `AutoVox_GAN` and `pianoRoll_to_midi`; their code and individual README files have been preserved.
