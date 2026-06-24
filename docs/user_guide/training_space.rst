.. _training_space:

================
Training Space
================

Before running the Bayesian optimisation loop, ALSEBO selects a small
*initial training set* — a diverse subset of the candidate pool that is
sent for experimental measurement first.
The quality of this initial batch directly affects how quickly the
surrogate model learns the fitness landscape.

Why diversity matters
---------------------

A GP surrogate trained on a clustered (non-diverse) initial set will have
high uncertainty everywhere except in the region it was trained on, causing
the acquisition function to explore only that region in early rounds.
By spreading the initial measurements across the full sequence landscape,
the surrogate gains a coarse but global picture of the fitness surface
from the very first round.

Sampling strategy
-----------------

:func:`alsebo.training_space.sample_initial_training_sequnces` implements
a two-stage approach:

**Stage 1 — Dimensionality reduction**

The high-dimensional feature vectors from ``seq_space.csv`` are projected
to 2-D using either t-SNE or PCA:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Method
     - Behaviour
     - When to use
   * - ``"TSNE"``
     - Non-linear; preserves local neighbourhood structure.
       Sequences that are functionally similar cluster together.
     - Default. Best when the feature space has non-linear structure
       (e.g. DCA or ESM features).
   * - ``"PCA"``
     - Linear; preserves global variance.
       Fast and deterministic.
     - Good starting point when the feature space is already
       low-dimensional (e.g. ``"latent"`` featurization with 2-D VAE).

**Stage 2 — k-means clustering + centroid selection**

The 2-D projection is partitioned into *k* clusters (where *k* =
``training_seq_size``).
For each cluster, the sequence closest to the centroid is selected as the
representative:

.. code-block:: python

    kmeans = KMeans(n_clusters=k, random_state=42).fit(gen_seq_X_2d)
    for cluster_id in range(k):
        cluster_points = np.where(labels == cluster_id)[0]
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(gen_seq_X_2d[cluster_points] - centroid, axis=1)
        best_idx = cluster_points[np.argmin(dists)]

This guarantees that the *k* selected sequences are as spread out as possible
across the projected landscape.

.. code-block:: python

    from alsebo.training_space import sample_initial_training_sequnces

    sample_initial_training_sequnces(
        exp_dir="./experiment/",
        training_seq_size=20,
        manipold="TSNE",          # or "PCA"
    )
    # writes experiment/training_seqs.csv

Recording experimental results
-------------------------------

After measuring the selected sequences, attach the objective values to create
the first training CSV with
:func:`alsebo.training_space.generate_sequence_training_file`:

.. code-block:: python

    from alsebo.training_space import generate_sequence_training_file

    obj_config = {
        "names":      ["fitness"],
        "directions": ["max"],
    }

    obj_values = [
        [0.45], [0.67], [0.31], ...   # one inner list per sequence
    ]

    generate_sequence_training_file(
        exp_dir="./experiment/",
        obj_config=obj_config,
        obj_values=obj_values,
    )
    # writes experiment/seq_exp_data.csv

The ``obj_config`` dictionary
------------------------------

``obj_config`` is passed through the entire ALSEBO pipeline and controls
how objectives are interpreted:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``"names"``
     - List of column names for each objective in the CSV,
       e.g. ``["fitness"]`` or ``["fitness", "thermostability"]``.
   * - ``"directions"``
     - List of ``"max"`` or ``"min"`` per objective.
       Minimisation objectives are sign-flipped internally before UCB scoring.
   * - ``"weights"``
     - *(optional)* List of floats summing to 1 for multi-objective
       scalarisation. Defaults to equal weights if omitted.
