import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def sample_initial_training_sequnces(
    exp_dir: str,
    training_seq_size: int,
    manipold: str = "TSNE",
) -> None:
    """Samples a diverse initial training set from the candidate sequence space.

    Projects the sequence feature space down to 2-D using the chosen manifold
    method, then runs k-means clustering and picks the sequence closest to each
    cluster centroid. The selected sequences are saved to ``training_seqs.csv``
    in ``exp_dir``.

    :param exp_dir: Path to the experiment directory containing ``seq_space.csv``
        and where ``training_seqs.csv`` will be written.
    :type exp_dir: str
    :param training_seq_size: Number of diverse sequences to select (equals the
        number of k-means clusters).
    :type training_seq_size: int
    :param manipold: Dimensionality reduction method to use before clustering.
        ``"TSNE"`` uses t-SNE and ``"PCA"`` uses PCA, defaults to ``"TSNE"``.
    :type manipold: str
    :return: None
    :rtype: None
    """
    seq_space_fpath = os.path.join(exp_dir,'seq_space.csv')
    gen_seq_features = pd.read_csv(seq_space_fpath)
    gen_seqs = gen_seq_features['seq_id']
    gen_seq_features = gen_seq_features.drop('seq_id',axis=1)

    if manipold=="TSNE":
        gen_seq_X_2d = TSNE(n_components=2, random_state=42).fit_transform(gen_seq_features)
    elif manipold=="PCA":
        gen_seq_X_2d = PCA(n_components=2, random_state=42).fit_transform(gen_seq_features)
    
    k = training_seq_size
    # k-means clustering on t-SNE coords
    kmeans = KMeans(n_clusters=k, random_state=42).fit(gen_seq_X_2d)
    labels = kmeans.labels_

    # find representative sequence for each cluster
    selected_indices = []
    for cluster_id in range(k):
        cluster_points = np.where(labels == cluster_id)[0]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # pick point closest to centroid
        dists = np.linalg.norm(gen_seq_X_2d[cluster_points] - centroid, axis=1)
        best_idx = cluster_points[np.argmin(dists)]
        selected_indices.append(best_idx)

    training_df = gen_seq_features.iloc[selected_indices]
    training_df['seq_id'] = gen_seqs.iloc[selected_indices]
    training_seq_fpath = os.path.join(exp_dir,'training_seqs.csv')
    training_df.to_csv(training_seq_fpath,index=False)

def generate_sequence_training_file(
    exp_dir: str,
    obj_config: dict,
    obj_values: list[list],
    file_name: str = "seq_exp_data.csv",
) -> None:
    """Combines the initial training sequences with their measured objective values.

    Reads the sampled training sequences from ``training_seqs.csv``, attaches
    the provided objective values as new columns, and writes the combined
    DataFrame to the experiment data CSV. Raises ``ValueError`` if the number
    of sequences does not match the number of objective rows.

    :param exp_dir: Path to the experiment directory containing
        ``training_seqs.csv`` and where the output CSV will be written.
    :type exp_dir: str
    :param obj_config: Objective configuration dict with a ``names`` key listing
        the objective column names.
    :type obj_config: dict
    :param obj_values: Measured objective values for each training sequence.
        Shape must be (n_training_sequences, n_objectives).
    :type obj_values: list[list[float]]
    :param file_name: Output filename for the experiment data CSV relative to
        ``exp_dir``, defaults to ``"seq_exp_data.csv"``.
    :type file_name: str
    :return: None
    :rtype: None
    """

    training_seq_fpath = os.path.join(exp_dir,'training_seqs.csv')
    seq_exp_data_fpath = os.path.join(exp_dir,file_name)

    training_df = pd.read_csv(training_seq_fpath)

    if len(training_df) != len(obj_values):
        raise ValueError(
            f"Mismatch: {len(training_df)} rows but "
            f"{len(obj_values)} objective entries."
        )
    
    obj_cols = obj_config['names']
    obj_df = pd.DataFrame(obj_values, columns=obj_cols)

    training_df = pd.concat([training_df, obj_df], axis=1)
    training_df.to_csv(seq_exp_data_fpath,index=False)








    

