import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import os

def read_seq_files(
    exp_dir: str,
    obj_config: dict,
    file_name: str = "seq_exp_data.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads training data and the candidate sequence space from CSV files.

    Reads the experimental results CSV to extract sequence encodings and
    objective values, then loads the full candidate sequence space and removes
    sequences that have already been tested.

    :param exp_dir: Path to the experiment directory containing the CSV files.
    :type exp_dir: str
    :param obj_config: Objective configuration dict with at least a ``names``
        key listing the objective column names.
    :type obj_config: dict
    :param file_name: Filename of the experimental data CSV relative to
        ``exp_dir``, defaults to ``"seq_exp_data.csv"``.
    :type file_name: str
    :return: Tuple of (x_train, y_train, x_space, seq_ids) where x_train and
        x_space are feature encodings, y_train holds objective values for tested
        sequences, and seq_ids lists of untested candidate sequences.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]
    """
    # Training data (sequences + experimental outcomes)

    exp_data_fpath = os.path.join(exp_dir+file_name)
    df_train = pd.read_csv(exp_data_fpath)

    train_seq_ids = df_train["seq_id"].values
    df_train = df_train.drop('seq_id',axis=1)
    obj_cols = obj_config['names']

    y_train = df_train[obj_cols] #multi objective dataframe
    x_train = df_train.drop(obj_cols, axis=1) #sequnce encodings


    # Full candidate sequence space
    seq_space_fpath = os.path.join(exp_dir+'seq_space.csv')
    df_seq = pd.read_csv(seq_space_fpath)

    # Delete the experimental sequnce from canditate sequnces to remove redundancy
    # Remove already tested sequences
    df_seq = df_seq[~df_seq["seq_id"].isin(train_seq_ids)].reset_index(drop=True)

    seq_ids = df_seq["seq_id"] # candidate sequences
    x_space = df_seq.drop('seq_id', axis=1)    # sequence encodings

    return x_train, y_train, x_space, seq_ids

def gpr(
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
) -> list[GaussianProcessRegressor]:
    """Trains Gaussian Process Regression model(s) on sequence data.

    Fits one GPR model per objective using a composite kernel
    (ConstantKernel * Matern + WhiteKernel). Handles both single-objective
    and multi-objective cases by fitting one model per output column.

    :param x_train: Feature encodings of the training sequences.
    :type x_train: pd.DataFrame or np.ndarray, shape (n_samples, n_features)
    :param y_train: Objective values for the training sequences. A 1-D array
        or single-column 2-D array is treated as single-objective; a
        multi-column 2-D array triggers one model per column.
    :type y_train: pd.DataFrame or np.ndarray, shape (n_samples,) or (n_samples, n_objectives)
    :return: List of fitted GaussianProcessRegressor models, one per objective.
    :rtype: list[GaussianProcessRegressor]
    """
    # Ensure numpy array
    y_train = np.asarray(y_train)

    # Kernel choice
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel()

    # --- Single-objective case ---
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        if y_train.ndim == 2:  # shape (n_samples, 1)
            y_train = y_train.ravel()  # flatten to (n_samples,)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        model.fit(x_train, y_train)
        return [model]

    # --- Multi-objective case ---
    else:
        models = []
        for i in range(y_train.shape[1]):  # loop over objectives
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            model.fit(x_train, y_train[:, i])
            models.append(model)
        return models


def seq_space_prediction(
    models: list[GaussianProcessRegressor],
    x_space: pd.DataFrame | np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generates posterior mean and standard deviation predictions over the candidate space.

    Runs each trained GPR model over all candidate sequences and collects
    the predictive mean and standard deviation for use in the acquisition function.

    :param models: List of fitted GPR models, one per objective.
    :type models: list[GaussianProcessRegressor]
    :param x_space: Feature encodings of the untested candidate sequences.
    :type x_space: pd.DataFrame or np.ndarray, shape (n_candidates, n_features)
    :return: List of (mean, std) tuples, one tuple per objective.
    :rtype: list[tuple[np.ndarray, np.ndarray]]
    """
    preds = []
    for model in models:
        mean, std = model.predict(x_space, return_std=True)
        preds.append((mean, std))
    return preds

def acquisition_function(
    preds: list[tuple[np.ndarray, np.ndarray]],
    obj_config: dict,
    strategy: str = "UCB",
    beta: float = 2.0,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Computes a scalar acquisition score for each candidate sequence.

    Combines per-objective UCB scores using weighted scalarization. Objective
    directions (``"min"`` or ``"max"``) are read from ``obj_config`` so that
    minimization objectives are flipped before scoring.

    :param preds: List of (mean, std) prediction tuples, one per objective,
        as returned by :func:`seq_space_prediction`.
    :type preds: list[tuple[np.ndarray, np.ndarray]]
    :param obj_config: Objective configuration dict containing ``directions``
        (list of ``"min"``/``"max"`` strings) and optionally ``weights``
        (list of floats summing to 1). Equal weights are used if not provided.
    :type obj_config: dict
    :param strategy: Acquisition strategy to use. Currently supports
        ``"UCB"`` (Upper Confidence Bound), defaults to ``"UCB"``.
    :type strategy: str
    :param beta: Exploration-exploitation trade-off parameter for UCB.
        Higher values favour exploration, defaults to ``2.0``.
    :type beta: float
    :param weights: Per-objective weights for scalarization. Overridden by
        ``obj_config["weights"]`` if present, defaults to ``None``.
    :type weights: list[float] or None
    :return: Acquisition scores for each candidate sequence.
    :rtype: np.ndarray, shape (n_candidates,)
    """
    n = len(preds[0][0])
    n_obj = len(preds)
    scores = np.zeros(n)
    weights = obj_config.get("weights", [1.0 / n_obj] * n_obj)
    directions = obj_config['directions']

    if strategy == "UCB":
        for i, (mean, std) in enumerate(preds):

            # Flip sign if objective is minimization
            if directions[i] == "min":
                mean = -mean

            w = weights[i]
            scores += w * (mean + beta * std)

    # TODO: extend with Pareto-based EI, Thompson sampling, diversity clustering ....
    return scores

def get_next_seq_bo(
    seq_ids: pd.Series,
    obj_config: dict,
    preds: list[tuple[np.ndarray, np.ndarray]],
    top_k: int = 10,
    strategy: str = "UCB",
    weights: list[float] | None = None,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Selects the top-k candidate sequences recommended for the next experiment.

    Scores all candidates using the acquisition function and returns the
    highest-scoring sequences along with their scores and indices.

    :param seq_ids: Sequences of all untested candidate sequences.
    :type seq_ids: pd.Series
    :param obj_config: Objective configuration dict passed through to
        :func:`acquisition_function`.
    :type obj_config: dict
    :param preds: List of (mean, std) prediction tuples, one per objective,
        as returned by :func:`seq_space_prediction`.
    :type preds: list[tuple[np.ndarray, np.ndarray]]
    :param top_k: Number of top candidates to return, defaults to ``10``.
    :type top_k: int
    :param strategy: Acquisition strategy passed to :func:`acquisition_function`,
        defaults to ``"UCB"``.
    :type strategy: str
    :param weights: Per-objective weights passed to :func:`acquisition_function`,
        defaults to ``None``.
    :type weights: list[float] or None
    :return: Tuple of (next_best_seqs, top_scores, top_idx) where
        next_best_seqs are the recommended sequences, top_scores are their
        acquisition scores, and top_idx are their positions in x_space.
    :rtype: tuple[pd.Series, np.ndarray, np.ndarray]
    """
    scores = acquisition_function(preds, obj_config, strategy=strategy, weights=weights)
    idx = np.argsort(scores)[::-1][:top_k]
    next_best_seqs = seq_ids.iloc[idx].reset_index(drop=True)

    return next_best_seqs, scores[idx], idx

def save_next_batch_results(
    exp_dir: str,
    next_best_seqs: pd.Series,
    idx: np.ndarray,
    x_space: pd.DataFrame,
    obj_config: dict,
    obj_values: list[list],
    file_name: str = "seq_exp_data.csv",
) -> None:
    """Appends the experimental results of the recommended batch to the training CSV.

    Combines the sequence feature encodings, sequences, and measured objective
    values into a single row per sequence and appends them to the experiment data
    file. Raises ``ValueError`` if the batch size or number of objectives does not
    match ``obj_values``.

    :param exp_dir: Path to the experiment directory where the CSV is stored.
    :type exp_dir: str
    :param next_best_seqs:  recommended sequences, as returned
        by :func:`get_next_seq_bo`.
    :type next_best_seqs: pd.Series
    :param idx: Integer indices of the recommended sequences within ``x_space``,
        as returned by :func:`get_next_seq_bo`.
    :type idx: np.ndarray
    :param x_space: Feature encodings of the full untested candidate space.
    :type x_space: pd.DataFrame
    :param obj_config: Objective configuration dict with a ``names`` key listing
        the objective column names.
    :type obj_config: dict
    :param obj_values: Measured objective values for the recommended batch.
        Shape must be (top_k, n_objectives).
    :type obj_values: list[list[float]]
    :param file_name: Filename of the experimental data CSV relative to
        ``exp_dir``, defaults to ``"seq_exp_data.csv"``.
    :type file_name: str
    :return: None
    :rtype: None
    """
    obj_array = np.array(obj_values)
    obj_cols = obj_config['names']
    if obj_array.ndim == 1:
        obj_array = obj_array.reshape(-1, 1)

    if len(idx) != obj_array.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {len(idx)} sequences "
            f"but {obj_array.shape[0]} objective rows."
        )

    if obj_array.shape[1] != len(obj_cols):
        raise ValueError(
            f"Objective column mismatch: expected {len(obj_cols)}, "
            f"got {obj_array.shape[1]}"
        )

    obj_df = pd.DataFrame(obj_array, columns=obj_cols)

    next_best_seq_features = x_space.iloc[idx].reset_index(drop=True)
    next_best_exp_data_df = pd.concat(
        [next_best_seq_features,next_best_seqs, obj_df],
           axis=1
        )
    exp_data_fpath = os.path.join(exp_dir+file_name)

    next_best_exp_data_df.to_csv(
        exp_data_fpath,
        mode='a',
        header=not os.path.exists(exp_data_fpath),
        index=False
    )
