#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=clip_clip
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu

#First Executable Line
module load WebProxy
source ~/.bashrc
cd /scratch/user/tahaismail424/coursework/cs633/clip-project
source .venv/bin/activate
cd csce689_iSogCLR

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"

data_path=../data
ann_path=../data/clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30
ita_type=clip

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path ${data_path} \
    --ann_path ${ann_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/smoke_${ita_type}_${data}_g${gamma}_e${epochs} \
    --init_model \
    --use_amp \
    --ita_type ${ita_type} \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} \
    --eta_init 0.03 --sched cosine \
    --no-distributed \
    --num-workers 16 \
    --epochs ${epochs}
