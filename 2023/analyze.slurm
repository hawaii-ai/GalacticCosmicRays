#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=kill-shared # sadow, gpu, kill-shared
##SBATCH --partition=sadow
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12gb ## max amount of memory per node you require
#SBATCH --time=03-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=analyze_hmc
##SBATCH --job-name=train_nn
#SBATCH --output=logs/job-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr
#jupyter nbconvert train_nn.ipynb --to python
#python train_nn.py

python analyze_hmc.py