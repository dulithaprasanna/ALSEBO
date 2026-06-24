.. _quickstart:

==========
Quickstart
==========

This page walks through a complete ALSEBO run — from a raw sequence alignment
to a recommended next experimental batch — in five steps.

Assumed directory layout
------------------------

All inputs and outputs live under a single experiment directory:

.. code-block:: text

    experiment/
    ├── msa.fasta              ← multiple sequence alignment (input)
    ├── generated_seqs.fasta   ← sequences sampled from the VAE (input)
    ├── seq_space.csv          ← featurized sequence space  (generated in Step 1)
    ├── training_seqs.csv      ← initial training batch     (generated in Step 2)
    └── seq_exp_data.csv       ← experimental data log      (generated in Step 3+)

Step 1 — Featurize the generated sequence space
------------------------------------------------

After training the VAE and sampling sequences into ``generated_seqs.fasta``,
compute features for the full candidate space.
Three featurization methods are available: ``"DCA"`` (default),
``"ESM"``, and ``"latent"`` (raw VAE latent coordinates).

.. code-block:: python

    from alsebo.seq_space import generate_seq_space

    EXP_DIR = "./experiment/"

    generate_seq_space(
        exp_dir=EXP_DIR,
        msa_fname="msa.fasta",
        featuarization_method="DCA",   # or "ESM" / "latent"
    )
    # writes experiment/seq_space.csv

Step 2 — Sample a diverse initial training set
-----------------------------------------------

Select a small, diverse batch of sequences to measure experimentally first.
Diversity is achieved by projecting the feature space to 2-D (t-SNE or PCA)
and picking the sequence closest to each k-means centroid.

.. code-block:: python

    from alsebo.training_space import sample_initial_training_sequnces

    sample_initial_training_sequnces(
        exp_dir=EXP_DIR,
        training_seq_size=20,   # number of sequences to select
        manipold="TSNE",        # or "PCA"
    )
    # writes experiment/training_seqs.csv

Step 3 — Record initial experimental results
--------------------------------------------

After measuring the selected sequences in the lab (or via simulation),
create the first training CSV by attaching objective values to the sequences.

.. code-block:: python

    from alsebo.training_space import generate_sequence_training_file

    obj_config = {
        "names":      ["fitness"],   # column name(s) in the CSV
        "directions": ["max"],       # "max" to maximise, "min" to minimise
    }

    # One inner list per sequence, one value per objective
    obj_values = [
        [0.45], [0.67], [0.31], [0.88], [0.52],
        # ... 20 rows total matching training_seq_size
    ]

    generate_sequence_training_file(
        exp_dir=EXP_DIR,
        obj_config=obj_config,
        obj_values=obj_values,
    )
    # writes experiment/seq_exp_data.csv

Step 4 — Run Bayesian Optimisation to select the next batch
------------------------------------------------------------

Train a Gaussian Process surrogate on the measured data, score the remaining
candidate sequences with the UCB acquisition function, and retrieve the top-k
recommendations.

.. code-block:: python

    from alsebo.optimizer import (
        read_seq_files,
        gpr,
        seq_space_prediction,
        get_next_seq_bo,
    )

    # Load training data and unexplored candidate space
    x_train, y_train, x_space, seq_ids = read_seq_files(EXP_DIR, obj_config)

    # Fit one GPR model per objective
    models = gpr(x_train, y_train)

    # Predict mean and uncertainty over the full candidate space
    preds = seq_space_prediction(models, x_space)

    # Rank candidates and return the top 5
    next_seqs, scores, idx = get_next_seq_bo(
        seq_ids,
        obj_config,
        preds,
        top_k=5,
        strategy="UCB",
        beta=2.0,           # higher = more exploration
    )

    print(next_seqs)

Step 5 — Append new results and repeat
---------------------------------------

Measure the recommended sequences and append the results to the training
CSV to close the active learning loop.

.. code-block:: python

    from alsebo.optimizer import save_next_batch_results

    new_obj_values = [
        [0.91], [0.84], [0.76], [0.93], [0.88],
    ]

    save_next_batch_results(
        exp_dir=EXP_DIR,
        next_best_seqs=next_seqs,
        idx=idx,
        x_space=x_space,
        obj_config=obj_config,
        obj_values=new_obj_values,
    )
    # appends 5 new rows to experiment/seq_exp_data.csv

Repeat Steps 4–5 for as many rounds as needed.
The GPR surrogate improves with each new measurement,
steering the search towards high-performing regions of the landscape.

Multi-objective example
-----------------------

To optimise two objectives simultaneously (e.g. fitness and thermostability),
extend ``obj_config`` with a second entry and provide two values per sequence:

.. code-block:: python

    obj_config = {
        "names":      ["fitness", "thermostability"],
        "directions": ["max",     "max"],
        "weights":    [0.6,       0.4],   # optional — equal weights by default
    }

    obj_values = [
        [0.91, 0.73],
        [0.84, 0.81],
        # ...
    ]
