# Data generation and optical music recognition

Optical music recognition from score PDFs, rule-based chord voicing, and augmentation of symbolic-MIDI data.

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Projects

- [`chord_generator_single_function`](chord_generator_single_function/) — This notebook consolidates the chord-generation logic into a more reusable workflow.
- [`chord_generator`](chord_generator/) — This notebook generates playable chord voicings from symbolic chord progressions while attempting to keep neighboring voices smooth.
- [`data_augment_midi_generator`](data_augment_midi_generator/) — This notebook generates or augments vocal MIDI using empirical distributions learned from the real corpus.
- [`efficient_pdf_to_midi`](efficient_pdf_to_midi/) — This notebook implements heuristic optical music recognition while reducing repeated work found in the earlier PDF pipeline.
- [`mix_vocals_by_bar`](mix_vocals_by_bar/) — This notebook creates synthetic vocal MIDI by recombining fixed-length bars from different performances.
- [`mix_vocals_midi`](mix_vocals_midi/) — This notebook transfers pitches between vocal MIDIs while preserving another performance's temporal skeleton.
- [`pdf_to_midi`](pdf_to_midi/) — This notebook is the original large heuristic optical-music-recognition pipeline in the archive.
- [`scaled_efficient_pdf_to_midi`](scaled_efficient_pdf_to_midi/) — This notebook optimizes the template-based OMR pipeline by performing recognition at a reduced image scale.
- [`separate_chord_and vox_gen`](separate_chord_and_vox_gen/) — This notebook experiments with generating chord and vocal material separately and then bringing the two streams into a compatible representation.
- [`true_size_efficient_pdf_to_midi`](true_size_efficient_pdf_to_midi/) — This is the most production-oriented version of the archive's heuristic PDF-to-MIDI system.
- [`vocals_by_bar_with_chord`](vocals_by_bar_with_chord/) — This notebook develops another bar-oriented synthetic-vocal corpus workflow, with the file positioned around vocal material that will be used alongside harmonic context.

See individual project READMEs for detailed algorithm steps and historical reproducibility caveats.
