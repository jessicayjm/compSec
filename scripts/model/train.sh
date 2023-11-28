#! /bin/bash
#
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --job-name=compsec
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --exclude=g[006,009]
#SBATCH --time=11:00:00

python train.py \
    --train_data /net/scratch/jiaminy/compsec/outputs/data/train.json \
    --dev_data /net/scratch/jiaminy/compsec/outputs/data/dev_correct.json \
    --test_data /net/scratch/jiaminy/compsec/outputs/data/test_correct.json \
    --train_data_size 1 \
    --num_train_epochs 1 \
    --output_dir model1