#!/bin/bash
#SBATCH --job-name=clip_eval
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu

module load WebProxy

cd /scratch/user/tahaismail424/coursework/cs633/clip-project
source .venv/bin/activate
cd csce689_iSogCLR

for ita_type in clip sogclr; do
  for ckpt in checkpoint_30 checkpoint_best; do
    python ./bimodal_exps/clip.py \
      --data_path ../data --ann_path ../data/clip_train \
      --train_file cc3m_train_subset.json --train_image_root cc3m_subset_100k/ \
      --init_model --no-distributed --ita_type ${ita_type} --sogclr_gamma 0.8 \
      --evaluate --checkpoint output/smoke_${ita_type}_cc3m_g0.8_e30/${ckpt}.pth \
      --zs_dataset imagenet --zs_datafolder ../data/imagenet/val \
      --output_dir output/eval_${ita_type}_${ckpt}
  done
done

