# Maximus | Music Generative AI

Research code for **Maximus**, a platform for generating symbolic vocal melodies from harmony, tempo, song structure, and lyrical context. Experiments are organized by the stages of the symbolic-music pipeline.

## Research pipeline

```text
Score PDFs / chord and vocal MIDI
               ↓
      1. Data generation
               ↓
      2. Data augmentation
               ↓
      3. Preprocessing
               ↓
      4. Generative AI models
               ↓
      5. Postprocessing
               ↓
      6. Post-generation utilities
```

The model experiments span MIDI DCGANs, chord conditioning, section/BPM control, dense-block variants, Transformer-based context, timing objectives, and diffusion/image-grid representations. The optical-music-recognition work uses score rasterization, OpenCV template matching, and musical-structure reconstruction.

## Browse the project

| Directory | Contents | Project folders |
| --- | --- | ---: |
| [`1_Data_Generation/`](1_Data_Generation/) | Score PDF-to-MIDI conversion and phoneme-to-note alignment editor. | 2 |
| [`2_Data_Augmentation/`](2_Data_Augmentation/) | Markov-chain MIDI augmentation, vocal-bar recombination with chord context, and MIDI pitch/timing mixing. | 4 |
| [`3_preprocessing/`](3_preprocessing/) | Alignment, normalization, quality checks, and encoding of symbolic musical and lyric inputs. | 17 |
| [`4_Generative_AI/`](4_Generative_AI/) | Conditional GANs, diffusion, Transformer, and supervised symbolic-melody experiments. | 37 |
| [`5_postprocessing/`](5_postprocessing/) | Decoding generated model outputs into note events, MIDI, and audio. | 7 |
| [`6_post_generation_utilities/`](6_post_generation_utilities/) | Post-generation chord-voicing utility experiment. | 1 |
