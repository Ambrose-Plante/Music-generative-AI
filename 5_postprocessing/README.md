# Inference, decoding, and audio rendering

Decoding model output arrays, generating MIDI note events, and synthesizing or mixing audio for qualitative evaluation.

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Projects

- [`array_to_midi_loop`](array_to_midi_loop/) — This notebook batch-converts real or generated piano-roll arrays into MIDI and listening examples.
- [`array_to_midi_stable_diffusion`](array_to_midi_stable_diffusion/) — This notebook adapts the array-to-MIDI decoding workflow to image-like outputs produced by the diffusion experiments.
- [`array_to_midi`](array_to_midi/) — This notebook decodes neural-network vocal arrays into explicit MIDI notes and then renders them for listening.
- [`generator_to_midi`](generator_to_midi/) — This notebook takes a trained generator checkpoint and turns a single or small set of generated vocal arrays into audible music.
- [`half_step_array_to_midi_loop`](half_step_array_to_midi_loop/) — This notebook is the batch renderer for models trained on the lower-resolution half-step representation.
- [`integrated_midi_export`](integrated_midi_export/) — This notebook is an integration prototype joining chord-MIDI parsing, trained generator inference, and output rendering.
- [`VoxSynthesizer`](VoxSynthesizer/) — This notebook implements sample-based vocal synthesis from phoneme recordings rather than a neural text-to-speech model.

See individual project READMEs for detailed algorithm steps and historical reproducibility caveats.
