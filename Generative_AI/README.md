# Generative models and neural networks

GANs, conditional GANs, diffusion, Transformer, supervised sequence, and architectural experiments for generating symbolic vocal melodies.

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Projects

- [`AutoVox_GAN`](AutoVox_GAN/) — This notebook implements a multimodal conditional GAN intended to generate vocal-output arrays from symbolic musical context.
- [`Copy of dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A`](Copy_of_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A/) — This is a dense-block conditional GAN for generating strict-C-major vocal piano rolls from chord context plus song-section and tempo information.
- [`Copy of dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_C`](Copy_of_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_C/) — This notebook is another revision of the dense conditional GAN family rather than an unrelated algorithm.
- [`Crazy_idea_Cmajor_strict_A`](Crazy_idea_Cmajor_strict_A/) — This notebook extends the strict-C-major dense conditional GAN with an explicit objective related to how many distinct note events the generated sequence contains.
- [`Crazy_idea_Cmajor_strict_B`](Crazy_idea_Cmajor_strict_B/) — This experiment carries the preceding conditional-GAN idea further by adding Transformer-based sequence processing and positional information.
- [`Crazy_idea_Cmajor_strict_C`](Crazy_idea_Cmajor_strict_C/) — This notebook is a further variant of the Transformer/dense conditional-GAN experiments for strict-C-major vocal generation.
- [`dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B`](dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B/) — This notebook is revision B of the dense strict-C-major conditional GAN series.
- [`dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_D`](dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_D/) — This is a later revision of the same strict-C-major, section-and BPM-conditioned dense GAN.
- [`dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_E`](dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_E/) — Revision E continues the same dense conditional GAN lineage for strict-C-major vocal arrays.
- [`half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A`](half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A/) — This notebook ports the dense chord/section/BPM-conditioned GAN concept to the project's half-step temporal representation.
- [`half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B`](half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B/) — This notebook is a second architectural revision of the half-step dense conditional GAN.
- [`list_conditional_DCGAN_network_pytorch`](list_conditional_DCGAN_network_pytorch/) — This notebook experiments with conditional adversarial generation using a list/sequence-style symbolic representation instead of relying exclusively on the large two-dimensional piano-roll layout.
- [`midi_conditional_DCGAN_network_pytorch_with_section_and_bpm`](midi_conditional_DCGAN_network_pytorch_with_section_and_bpm/) — This notebook extends the section-conditioned GAN with explicit tempo information.
- [`midi_conditional_DCGAN_network_pytorch_with_section`](midi_conditional_DCGAN_network_pytorch_with_section/) — This notebook adds song-section identity to the chord-conditioned GAN.
- [`midi_conditional_DCGAN_network_pytorch`](midi_conditional_DCGAN_network_pytorch/) — This notebook is an early chord-conditioned extension of the unconditional MIDI DCGAN.
- [`midi_conditional_DCGAN_TransD`](midi_conditional_DCGAN_TransD/) — This notebook modifies the conditional MIDI GAN by introducing Transformer-based processing on the discriminator side.
- [`midi_DCGAN_network_pytorch`](midi_DCGAN_network_pytorch/) — This notebook is the archive's clearest baseline GAN for vocal piano-roll generation.
- [`monitor_temp`](monitor_temp/) — Despite its filename, this notebook does not primarily monitor computer temperature; its saved content is a dense conditional GAN evaluation/model-monitoring experiment.
- [`network_A_full_pitch`](network_A_full_pitch/) — This notebook is a larger conditional GAN experiment that retains a fuller pitch representation and mixes convolutional processing with Transformer-style context modeling.
- [`network_B`](network_B/) — This notebook explores a Transformer-heavy version of the conditional vocal GAN.
- [`Network_batches`](Network_batches/) — This notebook revises the supervised Network approach to make training operate more naturally on batches.
- [`network_C`](network_C/) — This notebook tests a more convolution-centric alternative within the same experimental network family.
- [`network_D`](network_D/) — This notebook tests a substantially more fully connected architecture for the same vocal-generation problem.
- [`Network_TPU`](Network_TPU/) — This file is zero bytes and therefore does not contain a valid Jupyter notebook document.
- [`Network`](Network/) — This notebook trains a supervised multimodal neural network to predict a structured vocal output rather than using a GAN.
- [`NLP_GAN_Wasserstein`](NLP_GAN_Wasserstein/) — This notebook replaces the standard probability-style GAN discriminator objective with a Wasserstein-style critic formulation.
- [`NLP_GAN`](NLP_GAN/) — This notebook combines linguistic sequence representations with adversarial vocal-generation machinery.
- [`NLP_seq2seq`](NLP_seq2seq/) — This notebook implements a Transformer-based supervised sequence model for the linguistic/vocal side of the project.
- [`run_NN_Cmaj_strict_full_step`](run_NN_Cmaj_strict_full_step/) — This notebook packages a trained strict-C-major conditional generator into an end-to-end inference workflow.
- [`softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section`](softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/) — This notebook changes the dense section-conditioned GAN's vocal output toward an explicit softmax formulation.
- [`stable_diffusion_2D`](stable_diffusion_2D/) — This is an experimental notebook for treating symbolic vocal music as a two-dimensional generative object suitable for diffusion modeling.
- [`tiered_vanilla_midi_conditional_DCGAN_network_pytorch_with_section`](tiered_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/) — This notebook modifies section-conditioned adversarial training so that multiple generated “tiers” or variants contribute to the discriminator objective.
- [`timing_net_A`](timing_net_A/) — This notebook shifts attention toward note timing and event structure while retaining the conditional adversarial framework.
- [`timing_net_B`](timing_net_B/) — This notebook is an alternate timing-oriented architecture that relies more heavily on channel-wise and fully connected transformations.
- [`trans_vanilla_midi_conditional_DCGAN_network_pytorch_with_section`](trans_vanilla_midi_conditional_DCGAN_network_pytorch_with_section/) — This notebook augments the vanilla section-conditioned GAN with explicit positional encoding and Transformer encoder layers.
- [`vanilla_midi_conditional_DCGAN_network_pytorch_with_section`](vanilla_midi_conditional_DCGAN_network_pytorch_with_section/) — This notebook is a relatively direct section-conditioned chord GAN that serves as a baseline for the dense, softmax, Transformer, and tiered variants.
- [`WAS_midi_conditional_DCGAN_network_pytorch_with_section`](WAS_midi_conditional_DCGAN_network_pytorch_with_section/) — Despite the Wasserstein-suggestive filename and comments, the active saved training path is most accurately described as a section-conditioned GAN using binary cross-entropy.

See individual project READMEs for detailed algorithm steps and historical reproducibility caveats.
