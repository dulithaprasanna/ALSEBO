import numpy as np
from dca.dca_class import dca
import pandas as pd 
from Bio import SeqIO
from Bio.SeqRecord  import SeqRecord
from Bio.Seq import Seq
import multiprocessing as mp
import os


def generate_seq_space(exp_dir, msa_fname,gen_seq_fasta_fname="generated_seqs.fasta", featuarization_method="DCA"):

    gen_seq_path = os.path.join(exp_dir,gen_seq_fasta_fname)
    gen_sequences = [str(i.seq) for i in SeqIO.parse(gen_seq_path, "fasta")]

    if featuarization_method=="DCA":
        gen_seq_features = generate_dca_features(exp_dir,msa_fname,gen_sequences)
    
    #create seq space csv file with dca features
    seq_space_fpath = os.path.join(exp_dir,'seq_space.csv')
    gen_seq_features_df = pd.DataFrame(gen_seq_features)
    gen_seq_features_df['seq_id'] = gen_sequences
    gen_seq_features_df.to_csv(seq_space_fpath, index=False)

    print("featuarize generated sequence space saved!")



def generate_dca_features(exp_dir,msa_fname, gen_sequences):

    msa_fpath = os.path.join(exp_dir, msa_fname)
    mfdcamodel = dca(msa_fpath)
    mfdcamodel.mean_field()
    hamiltonian = mfdcamodel.compute_Hamiltonian(msa_fpath)

    n = mfdcamodel.N
    localfields = mfdcamodel.localfields
    couplings = mfdcamodel.couplings    

    # Encode sequence (e.g., mapping AAs to 0–20)
    aa_to_index = {aa: i for i, aa in enumerate('-ACDEFGHIKLMNPQRSTVWY')}

    dca_features = []
    for sequence in gen_sequences:
        seq_indices = [aa_to_index[aa] for aa in sequence]
        dca_feature = compute_dca_features(seq_indices, localfields, couplings)
        dca_features.append(dca_feature)
    gen_seq_features = np.array(dca_features)

    return gen_seq_features

def compute_dca_features(seq_indices, localfields, couplings):
    """
    Vectorized DCA feature computation.
    """
    n = len(seq_indices)
    features = np.zeros(n)

    # Precompute array of partner amino acids
    seq_idx = np.array(seq_indices)

    for i in range(n):
        aa_i = seq_idx[i]
        # Local field contribution
        f_i = localfields[aa_i, i]

        # All pairwise couplings with i (broadcast)
        aa_j = seq_idx
        f_i += 0.5 * np.sum(couplings[i, np.arange(n), aa_i, aa_j]) - 0.5 * couplings[i, i, aa_i, aa_i]

        features[i] = f_i

    return features

def generate_esm_features(exp_dir,msa_fname, gen_sequences):
    pass

