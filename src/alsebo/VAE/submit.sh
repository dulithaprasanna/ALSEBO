#!/bin/bash
#SBATCH --time=20:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=nova    # gpu node(s)

module load cuda

conda activate base
cd path_to_vae/VAE/
python run_vae.py path_to_fasta path_to_output_directory/saved_model.keras path_to_output_directory
