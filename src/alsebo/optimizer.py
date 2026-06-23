import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import os

def read_seq_files(exp_dir,obj_config,file_name="seq_exp_data.csv"):
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

def gpr(x_train, y_train):
    """
    Train Gaussian Process Regression model(s).
    Handles both single-objective (SO) and multi-objective (MO).
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


def seq_space_prediction(models, x_space):
    preds = []
    for model in models:
        mean, std = model.predict(x_space, return_std=True)
        preds.append((mean, std))
    return preds

def acquisition_function(preds, obj_config, strategy="UCB", beta=2.0, weights=None):
    """
    preds: list of (mean, std) tuples for each objective
    strategy: UCB (Upper Confidence Bound) or EI
    weights: objective weights for scalarization
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

def get_next_seq_bo(seq_ids:pd.Series,obj_config, preds, top_k=10, strategy="UCB", weights=None):

    scores = acquisition_function(preds, obj_config, strategy=strategy, weights=weights)
    idx = np.argsort(scores)[::-1][:top_k]
    next_best_seqs = seq_ids.iloc[idx].reset_index(drop=True)

    return next_best_seqs, scores[idx],idx

def save_next_batch_results(exp_dir,next_best_seqs,idx,x_space,obj_config,obj_values: list[list],file_name="seq_exp_data.csv"):

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


