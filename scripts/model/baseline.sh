#! /bin/bash
#
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --job-name=compsec
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --exclude=g[006,009]
#SBATCH --time=11:00:00

python baseline.py \
    --train_data /net/scratch/jiaminy/compsec/outputs/data/train.json \
    --dev_data /net/scratch/jiaminy/compsec/outputs/data/dev.json \
    --test_data /net/scratch/jiaminy/compsec/outputs/data/test.json