.. _bayesian_opt:

====================
Bayesian Optimisation
====================

Bayesian Optimisation (BO) is the engine that drives the ALSEBO active
learning loop.
Rather than evaluating sequences randomly, it builds a cheap-to-query
*surrogate model* of the fitness landscape and uses it to decide which
sequences to measure next.

The surrogate model — Gaussian Process Regression
---------------------------------------------------

ALSEBO uses a **Gaussian Process Regressor (GPR)** as its surrogate.
A GP places a probability distribution over functions: given a set of
training points it returns a posterior *mean* (best estimate) and
*standard deviation* (uncertainty) at every candidate point.

**Kernel**

The covariance structure is defined by a composite kernel:

.. math::

    k(x, x') = C \cdot k_{\text{Matérn}}(x, x') + k_{\text{noise}}

- **ConstantKernel** (C) — overall output scale.
- **Matérn** (ν = 2.5) — smooth but not infinitely differentiable; a good
  default for biological fitness landscapes.
- **WhiteKernel** — models observation noise (experimental measurement error).

All kernel hyperparameters are optimised by maximising the log marginal
likelihood during :func:`~alsebo.optimizer.gpr`.

**Multi-objective case**

One independent GPR is fitted per objective.
This is handled automatically: if ``y_train`` has multiple columns,
:func:`~alsebo.optimizer.gpr` returns a list of models — one per column.

.. code-block:: python

    from alsebo.optimizer import read_seq_files, gpr, seq_space_prediction

    x_train, y_train, x_space, seq_ids = read_seq_files(EXP_DIR, obj_config)
    models = gpr(x_train, y_train)          # list of GPR models
    preds  = seq_space_prediction(models, x_space)  # [(mean, std), ...]

The acquisition function — UCB
-------------------------------

The **Upper Confidence Bound (UCB)** acquisition function balances
*exploration* (high uncertainty) with *exploitation* (high predicted value):

.. math::

    \text{UCB}_i(x) = \mu_i(x) + \beta \cdot \sigma_i(x)

where μ\ :sub:`i` and σ\ :sub:`i` are the posterior mean and standard
deviation for objective *i*.

For multiple objectives, per-objective UCB scores are combined via
weighted scalarisation:

.. math::

    \text{score}(x) = \sum_{i} w_i \cdot \text{UCB}_i(x)

Minimisation objectives are sign-flipped (μ → −μ) before scoring so the
acquisition function always maximises.

.. code-block:: python

    from alsebo.optimizer import get_next_seq_bo

    next_seqs, scores, idx = get_next_seq_bo(
        seq_ids,
        obj_config,
        preds,
        top_k=5,
        strategy="UCB",
        beta=2.0,       # exploration–exploitation trade-off
    )

Tuning ``beta``
----------------

``beta`` is the single most important BO hyperparameter:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - Low (e.g. 0.5)
     - Exploitative — recommends sequences near already-known good regions.
       Converges fast but may miss the global optimum.
   * - High (e.g. 5.0)
     - Explorative — recommends sequences in uncertain, unexplored regions.
       More robust to multi-modal landscapes but slower to converge.
   * - **2.0** *(default)*
     - Good general-purpose starting point for protein fitness landscapes.

Appending results and closing the loop
---------------------------------------

After measuring the recommended batch, append results to the training CSV
and repeat:

.. code-block:: python

    from alsebo.optimizer import save_next_batch_results

    save_next_batch_results(
        exp_dir=EXP_DIR,
        next_best_seqs=next_seqs,
        idx=idx,
        x_space=x_space,
        obj_config=obj_config,
        obj_values=new_obj_values,   # list[list[float]], shape (top_k, n_objectives)
    )

:func:`~alsebo.optimizer.save_next_batch_results` appends the new rows to
``seq_exp_data.csv`` (creating it if missing) and the next call to
:func:`~alsebo.optimizer.read_seq_files` will automatically exclude all
already-tested sequences from the candidate pool.

Round-by-round improvement
----------------------------

Each round the surrogate benefits from more data and the acquisition
function's recommendations become increasingly targeted:

.. code-block:: text

    Round 0  →  diverse random-ish initial batch (k-means sampling)
    Round 1  →  GPR trained on 20 points; UCB identifies promising regions
    Round 2  →  GPR trained on 25 points; uncertainty shrinks in good regions
    Round N  →  surrogate converges; recommendations cluster near the optimum
