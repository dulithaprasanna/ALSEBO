.. _installation:

============
Installation
============

Requirements
------------

- Python **3.10** or newer
- Git (required to install the DCA dependency from source)

Standard install
----------------

Clone the repository and install in editable mode so local changes are
reflected immediately:

.. code-block:: bash

    git clone https://github.com/dulithaprasanna/ALSEBO.git
    cd ALSEBO
    pip install -e .

This installs all dependencies declared in ``setup.cfg``, including the
``dca`` package which is fetched directly from GitHub:

.. code-block:: text

    dca @ git+https://github.com/utdal/py-mfdca.git@vae_dependency

.. note::

    Because ``dca`` is a Git-only dependency, make sure Git is available in
    the environment where you run ``pip install``.

Virtual environment (recommended)
----------------------------------

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -e .

Dependencies
------------

The following packages are installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - ``numpy`` / ``pandas``
     - Array operations and data I/O
   * - ``scikit-learn``
     - Gaussian Process Regression, t-SNE, PCA, k-means
   * - ``biopython``
     - FASTA parsing in sequence space and VAE data loading
   * - ``transformers`` / ``torch``
     - ESM protein language model embeddings
   * - ``tensorflow``
     - VAE training (encoder–decoder architecture)
   * - ``matplotlib`` / ``seaborn``
     - Plotting utilities
   * - ``dca``
     - Mean-field Direct Coupling Analysis features

Verify the installation
-----------------------

.. code-block:: python

    import alsebo
    print(alsebo.__version__)

Building the documentation locally
------------------------------------

.. code-block:: bash

    pip install tox
    tox -e viewdocs   # builds HTML and serves at http://localhost:8000
