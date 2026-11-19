#!/bin/bash
# Global-anchor (full-dataset gradient) runs.
# Usage: sbatch [--gres=gpu:a30:1] [--time=..] ga.slurm <run_name> [extra clip.py args...]
#   e.g. sbatch ga.slurm adamw_lr1e-3 --opt adamW --lr 1e-3 --opt_betas 0.9 0.95 --epochs 300
#        sbatch ga.slurm lbfgs --opt lbfgs --epochs 30
#SBATCH --job-name=ga
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a30:1
#SBATCH --partition=gpu

cd /scratch/user/tahaismail424/coursework/cs633/clip-project
source .venv/bin/activate
cd csce689_iSogCLR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_name=$1; shift

python ./bimodal_exps/clip.py \
    --data_path ../data \
    --ann_path ../data/clip_train \
    --train_file cc3m_train_subset.json \
    --train_image_root cc3m_subset_100k/ \
    --init_model \
    --no-distributed \
    --ita_type clip \
    --temp 0.01 \
    --global-anchor \
    --batch_size_train 64 \
    --num-workers 16 \
    --output_dir output/ga_${run_name} \
    "$@"
