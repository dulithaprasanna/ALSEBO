.. _overview:

========
Overview
========

What is ALSEBO?
---------------

Protein engineering requires searching through an astronomically large
sequence space — there are 20\ :sup:`N` possible sequences of length *N*.
Exhaustive screening is impossible; random mutagenesis is inefficient.
ALSEBO (*Active Learning Sequence Exploration via Bayesian Optimisation*)
addresses this by combining two ideas:

1. A **Variational Autoencoder (VAE)** that learns a continuous, low-dimensional
   *latent generative landscape* from a multiple sequence alignment (MSA), then
   samples diverse, biologically plausible candidate sequences from that
   landscape [Ziegler2023]_.

2. A **Bayesian Optimisation (BO)** loop that builds a probabilistic surrogate
   model from a small number of experimental measurements and uses it to
   intelligently select the next sequences most likely to improve the objective.

Together they reduce the number of wet-lab experiments needed to find
high-performing variants by orders of magnitude compared to random or
exhaustive approaches.

The ALSEBO pipeline
-------------------

.. code-block:: text

                        ┌─────────────────────────────────┐
                        │         MSA  (input)            │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   VAE — latent landscape        │
                        │   • encodes sequences → z       │
                        │   • samples new sequences       │
                        └────────────────┬────────────────┘
                                         │  generated_seqs.fasta
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Sequence Space                │
                        │   featurize: DCA · ESM · latent │
                        │   → seq_space.csv               │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Initial Training Set          │
                        │   t-SNE/PCA + k-means           │
                        │   → training_seqs.csv           │
                        └────────────────┬────────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   Wet-lab / in silico        │
                          │   evaluation                 │
                          │   measure objective(s)       │
                          └──────────────┬──────────────┘
                                         │  seq_exp_data.csv
                    ┌────────────────────▼────────────────────────┐
                    │              Active Learning Loop            │
                    │                                             │
                    │   GPR surrogate  ──►  UCB acquisition       │
                    │        ▲                    │               │
                    │        │           top-k candidates         │
                    │        │                    │               │
                    │   append results  ◄─  evaluate batch        │
                    └─────────────────────────────────────────────┘

Modules at a glance
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Module
     - Key functions
     - Responsibility
   * - :mod:`alsebo.seq_space`
     - :func:`~alsebo.seq_space.generate_seq_space`
     - Featurize generated sequences into ``seq_space.csv``
   * - :mod:`alsebo.training_space`
     - :func:`~alsebo.training_space.sample_initial_training_sequnces`,
       :func:`~alsebo.training_space.generate_sequence_training_file`
     - Diverse initial batch selection and data file creation
   * - :mod:`alsebo.optimizer`
     - :func:`~alsebo.optimizer.gpr`,
       :func:`~alsebo.optimizer.get_next_seq_bo`,
       :func:`~alsebo.optimizer.save_next_batch_results`
     - GPR surrogate, UCB scoring, result persistence

Choosing a featurization method
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Method
     - When to use
     - Notes
   * - ``"DCA"``
     - You have a deep MSA (>1 000 sequences) and want co-evolutionary
       features that capture residue–residue interactions.
     - Computationally cheap at inference time; requires the ``dca`` package.
   * - ``"ESM"``
     - You have a shallow MSA or want richer, context-aware embeddings
       from a pre-trained protein language model.
     - Requires GPU for large batches; uses
       ``facebook/esm2_t30_150M_UR50D`` by default.
   * - ``"latent"``
     - You want to work entirely within the VAE latent space — the
       2-D coordinates stored in the FASTA headers are used directly.
     - Fastest option; only valid when sequences were generated by the
       ALSEBO VAE with latent coordinates written into the FASTA headers.

References
----------

.. [Ziegler2023] Ziegler, C., Martin, J., Sinner, C., & Morcos, F. (2023).
   Latent generative landscapes as maps of functional diversity in protein
   sequence space. *Nature Communications*, **14**, 2222.
   https://doi.org/10.1038/s41467-023-37958-z
