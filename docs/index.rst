======
ALSEBO
======

**Active Learning Sequence Exploration via Bayesian Optimization**

ALSEBO is a Python framework for navigating protein sequence space using a
closed-loop active learning strategy. A Variational Autoencoder (VAE) generates
a continuous latent landscape of sequences; Bayesian Optimisation (BO) then
iteratively proposes the most promising candidates for experimental testing.

.. code-block:: text

    VAE latent landscape
          │
          ▼
    Sequence Space  (DCA · ESM · latent features)
          │
          ▼
    Initial Training Set  (t-SNE / PCA + k-means diversity sampling)
          │
          ▼
    Experiment  →  GPR surrogate  →  UCB acquisition  →  Next Batch
          └──────────────────────────────────────────────────┘
                         active learning loop

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation <installation>
   Quickstart <quickstart>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   Overview <readme>
   Contributions & Help <contributing>
   License <license>
   Authors <authors>
   Changelog <changelog>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
