#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -N javascript32

# Request one GPU in the gpu queue:

#$ -cwd
#$ -l h_rt=40:10:00
#$ -l h_vmem=40G
#$ -m bea -M s2584336@ed.ac.uk

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda

export HF_HOME="/exports/eddie/scratch/s2584336/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2584336/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2584336/huggingface_cache/datasets"


# Load Python
module load anaconda # this loads a specific version of anaconda
conda activate /exports/eddie/scratch/s2584336/anaconda/envs_dirs/mem # this starts the environment

# Run the program
python /exports/eddie/scratch/s2584336/combined/bertandt5/main.py --prog_lang "javascript"
