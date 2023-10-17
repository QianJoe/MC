#!/bin/bash
#SBATCH --gres=gpu:2
#module load apps/anaconda3/2021.05
source activate py3.8

python -u main.py
