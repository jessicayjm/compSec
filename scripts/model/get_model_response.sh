#! /bin/bash
#
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --job-name=get_model_response
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --exclude=g[006,009]
#SBATCH --time=11:00:00

python get_model_response.py \
    --train_data /net/scratch/jiaminy/compsec/outputs/data/train.json \
    --num_instances 60