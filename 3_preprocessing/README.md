# Preprocessing and dataset construction

Converting raw MIDI, score-derived MIDI, lyrics, phonetic annotations, and song metadata into aligned, normalized model inputs.

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Projects

- [`256x256`](pianoRoll_to_midi/) — The notebook develops a reversible representation transform that packs a long, narrow vocal piano-roll array into a square 256×256 matrix.
- [`analyze_half_step_midi_only_data_strict_Cmajor`](analyze_half_step_midi_only_data_strict_Cmajor/) — This notebook performs quality assurance on the half-step, MIDI-only, strict-C-major version of the music dataset.
- [`check_and_analyze_data`](check_and_analyze_data/) — This notebook combines dataset quality control with empirical statistical modeling of vocal-note behavior.
- [`check_and_analyze_midi_only_data_strict_Cmajor`](check_and_analyze_midi_only_data_strict_Cmajor/) — This notebook applies the MIDI-only validation workflow specifically to the strict-C-major normalized corpus.
- [`check_and_analyze_midi_only_data`](check_and_analyze_midi_only_data/) — This notebook validates the version of the dataset built directly from MIDI rather than JSON-style vocal annotation.
- [`clean_lyrics_database`](clean_lyrics_database/) — This notebook cleans a lyrics corpus so that downstream pronunciation and syllable algorithms receive words they can process.
- [`midi_process_full_2`](midi_process_full_2/) — This notebook converts raw song annotations, chord MIDI, tempo metadata, and vocal-note information into fixed-size arrays for model training.
- [`export_vocals_midi`](export_vocals_midi/) — This notebook exports a standardized collection of vocal MIDI files from the project's checked source data.
- [`midi_process_Cmajor_strict`](midi_process_Cmajor_strict/) — This preprocessing notebook builds a corpus whose pitch vocabulary is deliberately restricted to C-major-compatible states.
- [`midi_process_full`](midi_process_full/) — This notebook is the principal full-resolution preprocessing path for the symbolic-vocal project.
- [`midi_process_half_step_Cmajor_strict`](midi_process_half_step_Cmajor_strict/) — This notebook combines strict-C-major pitch normalization with a coarser half-step temporal representation.
- [`midi_process_half_step_midi_only_Cmajor_strict`](midi_process_half_step_midi_only_Cmajor_strict/) — This notebook produces the half-step strict-C-major dataset directly from MIDI files, eliminating dependence on a separate JSON vocal-annotation source.
- [`midi_process_midi_only_full`](midi_process_midi_only_full/) — This notebook is the full-resolution counterpart to the MIDI-only preprocessing branch.
- [`midi_to_sequence`](midi_to_sequence/) — This notebook converts musical and linguistic inputs into a sequence-oriented representation for models that should process events or tokens rather than raw multidimensional grids.
- [`NLP_vocal_preprocess_no_glove`](NLP_vocal_preprocess_no_glove/) — This notebook is a simplified preprocessing branch that removes dependence on external GloVe-style word vectors.
- [`NLP_vocal_preprocess`](NLP_vocal_preprocess/) — This notebook turns raw lyrics, pronunciation information, and vocal-note annotations into aligned numerical inputs for NLP-aware music models.
- [`process_and_split`](process_and_split/) — This notebook builds multimodal examples containing vocal output, phonetic input, audio-derived accompaniment/vocal features, chords, key, and BPM information.

See individual project READMEs for detailed algorithm steps and historical reproducibility caveats.
