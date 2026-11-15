This notebook implements a multimodal conditional GAN intended to generate vocal-output arrays from
symbolic musical context. Its dataset loader reads the desired output together with streams including syllable, vocal/accompaniment, chord, key, and BPM-related arrays. The generator combines convolutional/dense processing of
conditioning information with latent noise and then upsamples toward the target vocal representation,
while the discriminator evaluates real versus generated samples and includes an auxiliary conditional-
classification head. Dense blocks concatenate intermediate feature maps, and spectral normalization is
used to constrain layers. Training combines adversarial binary cross-entropy
with an auxiliary categorical loss, followed by checkpointing and generated-array export.

Key algorithmic steps:
1. load multimodal NPZ data
2. encode conditioning
3. concatenate latent noise
4. dense/transposed-convolution generation
5. discriminator real/fake + class prediction
6. optimize, checkpoint, export
