#! /bin/bash
#
#SBATCH --partition=general
#SBATCH --gres=gpu:8
#SBATCH --job-name=compsec
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --exclude=g[006,009]
#SBATCH --time=11:00:00

output_dir=/net/scratch/jiaminy/compsec/outputs/attack_outputs/all_possible_30
mkdir -p ${output_dir}

python train.py \
    --train_data /net/scratch/jiaminy/compsec/outputs/data/adversial_train_data_all_possible.json \
    --dev_data /net/scratch/jiaminy/compsec/outputs/data/dev_correct.json \
    --test_data /net/scratch/jiaminy/compsec/outputs/data/test_correct.json \
    --learning_rate 2e-5 \
    --train_data_size 30 \
    --per_device_train_batch_size 30 \
    --num_train_epochs 1 \
    --output_dir ${output_dir}