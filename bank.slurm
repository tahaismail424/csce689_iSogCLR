#!/bin/bash
# Memory-bank (cached global negatives, minibatch updates) runs.
# Usage: sbatch [--gres=gpu:a30:1] [--time=..] bank.slurm <run_name> [extra clip.py args...]
#   e.g. sbatch bank.slurm r781 --bank-refresh-steps 781 --epochs 30
#        sbatch --time=01:00:00 bank.slurm smoke_r20 --train_frac 0.05 --epochs 2 --bank-refresh-steps 20
# Defaults match run.slurm (baseline CLIP): AMP, lr 2e-4, batch 128, tau 0.01, cosine.
#SBATCH --job-name=bank
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu

cd /scratch/user/tahaismail424/coursework/cs633/clip-project
source .venv/bin/activate
cd csce689_iSogCLR
export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_name=$1; shift

python ./bimodal_exps/clip.py \
    --data_path ../data \
    --ann_path ../data/clip_train \
    --train_file cc3m_train_subset.json \
    --train_image_root cc3m_subset_100k/ \
    --init_model \
    --use_amp \
    --no-distributed \
    --ita_type clip \
    --tau_init 0.01 \
    --temp 0.01 \
    --sched cosine \
    --memory-bank \
    --num-workers 16 \
    --output_dir output/bank_${run_name} \
    "$@"
