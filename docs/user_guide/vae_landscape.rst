.. _vae_landscape:

=========================
VAE Latent Landscape
=========================

The Variational Autoencoder (VAE) is the generative backbone of ALSEBO.
It learns a smooth, continuous latent space from a protein MSA that can be
sampled to produce novel, biologically plausible sequences [Ziegler2023]_.

Conceptual background
---------------------

A standard autoencoder maps an input **x** (a one-hot encoded sequence) to a
compressed representation **z** (latent code) and back.
A VAE adds a probabilistic twist: the encoder outputs a *distribution*
over **z** (a Gaussian parameterised by μ and σ) rather than a single point.
During training the model is forced to keep that distribution close to a
standard normal, which makes the latent space smooth and interpolatable.

.. code-block:: text

    Sequence (one-hot)
          │
          ▼
    ┌─────────────┐     ┌─────────────┐
    │   Encoder   │────►│  μ,  σ      │  (latent parameters)
    └─────────────┘     └──────┬──────┘
                               │  reparameterisation trick:  z = μ + ε·σ
                               ▼
                        ┌─────────────┐
                        │   Decoder   │────► Reconstructed sequence
                        └─────────────┘

The **reparameterisation trick** (z = μ + ε·σ, ε ~ 𝒩(0,1)) makes the
sampling step differentiable so the whole network can be trained end-to-end
with gradient descent.

Architecture
------------

The ALSEBO VAE (``alsebo.VAE.model``) uses a fully-connected architecture:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Details
   * - **Input**
     - One-hot encoded sequence flattened to shape
       ``(num_aa_types × sequence_length,)``; 23 amino acid categories
       (20 standard + selenocysteine, gap, pyrrolysine).
   * - **Encoder**
     - Two dense hidden layers (2 × hidden_units → hidden_units) with
       BatchNormalization and configurable activation (default ``relu``),
       followed by two linear heads for μ and σ.
   * - **Latent space**
     - 2-D by default (``dim_latent_vars=2``), making it directly
       visualisable as a 2-D map of sequence diversity.
   * - **Decoder**
     - Mirror of the encoder; output reshaped to
       ``(num_aa_types, sequence_length)`` and passed through a Softmax
       to produce a per-position amino acid probability distribution.
   * - **Loss**
     - Reconstruction loss (binary cross-entropy) + KL divergence
       between the encoder distribution and 𝒩(0, I).

Training
--------

Train the VAE on an MSA FASTA file using the ``run_vae.py`` script:

.. code-block:: bash

    python src/alsebo/VAE/run_vae.py \
        path/to/msa.fasta \
        path/to/save/model \
        path/to/logs/

Key hyperparameters in ``run_vae.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Effect
   * - ``dim_latent_vars``
     - 2
     - Dimensionality of the latent space.
       2-D is recommended for visualisation and latent featurization.
   * - ``num_hidden_units``
     - ``seq_len × 3``
     - Capacity of the encoder/decoder.
       Increase for longer or more diverse MSAs.
   * - ``EPOCHS``
     - 1000
     - Training iterations; ``EarlyStopping(patience=10)`` terminates
       early if loss plateaus.
   * - ``BATCH_SIZE``
     - 16
     - Sequences per gradient update.

Generating the sequence space
------------------------------

After training, sample uniformly from the 2-D latent space and decode each
point to obtain a pool of candidate sequences.
The resulting FASTA file should have the latent coordinates (z0, z1) written
into each sequence header so that the ``"latent"`` featurization method can
read them directly:

.. code-block:: text

    >1.5991983967935868 -2.1042084168336674
    MKVLILGAGFIGSELTARLHESTGDNVKVFCLVRDN...

This is the ``generated_seqs.fasta`` consumed by :func:`alsebo.seq_space.generate_seq_space`.

Latent landscape as a functional map
--------------------------------------

The key insight from Ziegler *et al.* [Ziegler2023]_ is that the VAE latent
space is not merely a compression — it organises sequences by phylogenetic
relationships and functional properties.
Regions of high Hamiltonian energy (as computed by DCA) correspond to
fitness barriers, while low-energy basins correspond to functional clusters.
ALSEBO exploits this structure by initialising the BO search from a
diverse, landscape-covering set of sequences rather than a random one.

References
----------

.. [Ziegler2023] Ziegler, C., Martin, J., Sinner, C., & Morcos, F. (2023).
   Latent generative landscapes as maps of functional diversity in protein
   sequence space. *Nature Communications*, **14**, 2222.
   https://doi.org/10.1038/s41467-023-37958-z
