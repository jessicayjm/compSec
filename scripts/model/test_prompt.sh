#! /bin/bash
#
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --job-name=test_prompt
#SBATCH --nodes=2
#SBATCH --mem=300gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:00:00

python test_prompt.py \
    --data /net/scratch/jiaminy/compsec/outputs/data/train_is_possible.json