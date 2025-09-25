import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def read_seq_files(exp_dir,obj):
    # Training data (sequences + experimental outcomes)

    df_train = pd.read_csv(exp_dir+'seq_exp_data.csv')
    df_train = df_train.drop('seq_id',axis=1)
    obj_col = obj['names']

    y_train = df_train[obj_col] #multi objective dataframe
    x_train = df_train.drop(obj_col, axis=1) #sequnce encodings
    

    # Full candidate sequence space
    df_seq = pd.read_csv(exp_dir+'seq_space.csv')
    # Todo delete the experimental sequnce from canditate sequnces to remove redundancy 
    seq_ids = df_seq["seq_id"].values # actual sequences
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

def acquisition_function(preds, strategy="UCB", beta=2.0, weights=None):
    """
    preds: list of (mean, std) tuples for each objective
    strategy: UCB (Upper Confidence Bound) or EI
    weights: objective weights for scalarization
    """
    n = len(preds[0][0])  # number of sequences
    scores = np.zeros(n)

    if strategy == "UCB":
        for i, (mean, std) in enumerate(preds):
            w = weights[i] if weights is not None else 1.0 / len(preds)
            scores += w * (mean + beta * std)

    # TODO: extend with Pareto-based EI, Thompson sampling, diversity clustering ....
    return scores

def get_next_seq_bo(seq_ids, preds, top_k=10, strategy="UCB", weights=None):

    scores = acquisition_function(preds, strategy=strategy, weights=weights)
    idx = np.argsort(scores)[::-1][:top_k]

    return seq_ids[idx], scores[idx],idx
