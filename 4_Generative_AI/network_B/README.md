# network_B

## Overview

This notebook explores a Transformer-heavy version of the conditional vocal GAN. Temporal features are passed through Transformer encoders so authenticity and/or generation decisions can depend on long-range relationships across the complete musical phrase. Section conditioning remains part of the adversarial problem, and an auxiliary one-value constraint encourages sparse categorical pitch behavior. Binary cross-entropy and section classification provide the main optimization signals around those sequence modules. It serves as an architecture comparison against the more convolution-dominated networks A and C and the fully connected network D.

## Algorithmic workflow

1. Form sequence features
2. Transformer contextualization
3. Condition on section/ music
4. Generate/discriminate
5. One-value and BCE/CE losses
6. Compare validation behavior
