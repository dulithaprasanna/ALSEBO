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


def generate_seq_space(
    exp_dir: str,
    msa_fname: str,
    gen_seq_fasta_fname: str = "generated_seqs.fasta",
    featuarization_method: str = "DCA",
) -> np.ndarray:
    """Featurizes the generated sequence space and saves it to a CSV file.

    Reads generated sequences from a FASTA file, computes features using the
    specified method (DCA, ESM, or latent), and writes a ``seq_space.csv`` to
    ``exp_dir`` with one row per sequence.

    :param exp_dir: Path to the experiment directory where inputs are read from
        and ``seq_space.csv`` is written to.
    :type exp_dir: str
    :param msa_fname: Filename of the multiple sequence alignment file relative
        to ``exp_dir``, used by the DCA featurization method.
    :type msa_fname: str
    :param gen_seq_fasta_fname: Filename of the generated sequences FASTA file
        relative to ``exp_dir``, defaults to ``"generated_seqs.fasta"``.
    :type gen_seq_fasta_fname: str
    :param featuarization_method: Featurization method to use. One of
        ``"DCA"``, ``"ESM"``, or ``"latent"``. Defaults to ``"DCA"``.
    :type featuarization_method: str
    :return: Feature matrix for all generated sequences.
    :rtype: np.ndarray, shape (n_sequences, n_features)
    """

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



def generate_dca_features(
    exp_dir: str,
    msa_fname: str,
    gen_sequences: list[str],
) -> np.ndarray:
    """Computes DCA (Direct Coupling Analysis) features for a list of sequences.

    Fits a mean-field DCA model on the provided MSA, then computes per-position
    DCA feature vectors for each generated sequence via :func:`compute_dca_features`.

    :param exp_dir: Path to the experiment directory containing the MSA file.
    :type exp_dir: str
    :param msa_fname: Filename of the multiple sequence alignment file relative
        to ``exp_dir``.
    :type msa_fname: str
    :param gen_sequences: List of amino acid sequences to featurize.
    :type gen_sequences: list[str]
    :return: DCA feature matrix, one row per sequence.
    :rtype: np.ndarray, shape (n_sequences, sequence_length)
    """

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

def compute_dca_features(
    seq_indices: list[int],
    localfields: np.ndarray,
    couplings: np.ndarray,
) -> np.ndarray:
    """Computes per-position DCA feature values for a single sequence.

    For each position ``i``, combines the local field contribution with the
    sum of pairwise coupling terms across all other positions.

    :param seq_indices: Amino acid indices for each position in the sequence,
        mapped from the standard alphabet ``'-ACDEFGHIKLMNPQRSTVWY'``.
    :type seq_indices: list[int]
    :param localfields: Local field parameters from the DCA model, indexed as
        ``localfields[aa_index, position]``.
    :type localfields: np.ndarray, shape (n_aa, sequence_length)
    :param couplings: Pairwise coupling parameters from the DCA model, indexed as
        ``couplings[i, j, aa_i, aa_j]``.
    :type couplings: np.ndarray, shape (sequence_length, sequence_length, n_aa, n_aa)
    :return: DCA feature vector for the sequence.
    :rtype: np.ndarray, shape (sequence_length,)
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

def generate_esm_features(
    gen_sequences: list[str],
    batch_size: int = 32,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
) -> np.ndarray:
    """Generates mean-pooled ESM embeddings for a list of protein sequences.

    Loads the specified ESM model, runs batched inference, and returns a single
    mean-pooled embedding vector per sequence by averaging over non-padding tokens
    (excluding ``<cls>`` and ``<eos>`` special tokens).

    :param gen_sequences: List of amino acid sequences to embed.
    :type gen_sequences: list[str]
    :param batch_size: Number of sequences to process per forward pass,
        defaults to ``32``.
    :type batch_size: int
    :param model_name: HuggingFace model identifier for the ESM model to load,
        defaults to ``"facebook/esm2_t30_150M_UR50D"``.
    :type model_name: str
    :return: ESM embedding matrix, one mean-pooled vector per sequence.
    :rtype: np.ndarray, shape (n_sequences, embedding_dim)
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

def generate_latent_features(
    fasta_path: str,
) -> np.ndarray:
    """Extracts latent coordinates from FASTA sequence headers.

    Parses each record's header as a pair of space-separated floats
    (``z0`` and ``z1``) representing 2-D latent space coordinates.
    Records with unparseable headers are skipped with a warning.

    :param fasta_path: Path to the FASTA file whose headers contain latent
        coordinates in the format ``"z0 z1"``.
    :type fasta_path: str
    :return: Latent coordinate matrix, one row per successfully parsed sequence.
    :rtype: np.ndarray, shape (n_sequences, 2)
    """
    
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
