import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

def sample_initial_training_sequnces(exp_dir,training_seq_size,manipold="TSNE" ):

    seq_space_fpath = os.path.join(exp_dir,'seq_space.csv')
    gen_seq_features = pd.read_csv(seq_space_fpath)
    gen_seqs = gen_seq_features['seq_id']
    gen_seq_features = gen_seq_features.drop('seq_id',axis=1)

    if manipold=="TSNE":
        gen_seq_X_2d = TSNE(n_components=2, random_state=42).fit_transform(gen_seq_features)
    
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

def generate_sequence_training_file(exp_dir, obj_config, obj_values: list[list],file_name="seq_exp_data.csv"):

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








    

