import numpy as np
from dca.dca_class import dca
import pandas as pd 
from Bio import SeqIO
from Bio.SeqRecord  import SeqRecord
from Bio.Seq import Seq
import multiprocessing as mp
import os

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


def generate_seq_space(exp_dir, msa_fname,gen_seq_fasta_fname="generated_seqs.fasta", featuarization_method="DCA"):

    gen_seq_path = os.path.join(exp_dir,gen_seq_fasta_fname)
    gen_sequences = [str(i.seq) for i in SeqIO.parse(gen_seq_path, "fasta")]

    if featuarization_method=="DCA":
        gen_seq_features = generate_dca_features(exp_dir,msa_fname,gen_sequences)
    elif featuarization_method=="ESM":
        gen_seq_features = generate_esm_features(gen_sequences)
    elif featuarization_method=="latent":
        gen_seq_features = generate_latent_features(gen_seq_path)
    else:
        gen_seq_features = generate_dca_features(exp_dir,msa_fname,gen_sequences)
    
    
    #create seq space csv file with dca features
    seq_space_fpath = os.path.join(exp_dir,'seq_space.csv')
    gen_seq_features_df = pd.DataFrame(gen_seq_features)
    gen_seq_features_df['seq_id'] = gen_sequences
    gen_seq_features_df.to_csv(seq_space_fpath, index=False)

    print("featuarize generated sequence space saved an!")

    return gen_seq_features



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

def generate_esm_features(gen_sequences, batch_size=32,model_name="facebook/esm2_t30_150M_UR50D"):
    """
    Generates mean-pooled ESM embeddings for a list of sequences.
    Structured to serve as a drop-in alternative to generate_dca_features.

    """
    # Initialize the Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    esm_features = []
    
    # Extract Features
    # We use batching here because running transformer inference one-by-one is too slow
    for i in range(0, len(gen_sequences), batch_size):
        batch_seqs = gen_sequences[i:i + batch_size]
        
        # Tokenize sequence (maps AAs to tokens automatically)
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        token_reps = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        for j in range(len(batch_seqs)):
            # Find actual sequence length ignoring the padding
            seq_len = attention_mask[j].sum().item()
            
            # Slice [1:seq_len-1] to remove <cls> and <eos> tokens
            actual_tokens = token_reps[j, 1:seq_len-1]
            
            # Mean pooling to get a single vector per sequence
            seq_embedding = actual_tokens.mean(dim=0)
            esm_features.append(seq_embedding.cpu().numpy())
            
    gen_seq_features = np.array(esm_features)
    
    return gen_seq_features

def generate_latent_features(fasta_path):
    
    latent_features = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        # Biopython stores the full header (minus the '>') in record.description
        # Example header: "1.5991983967935868 -2.1042084168336674"
        parts = record.description.split()
        
        try:
            # Extract the z0 and z1 coordinates
            z0 = float(parts[0])
            z1 = float(parts[1])
            
            latent_features.append([z0, z1])
            
        except (ValueError, IndexError):
            print(f"Warning: Could not parse coordinates from header: {record.description}")
            continue

    return np.array(latent_features)
