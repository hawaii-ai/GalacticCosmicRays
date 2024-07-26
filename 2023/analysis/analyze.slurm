#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=sadow # sadow, gpu, kill-shared, shared
#SBATCH --partition=sadow
#SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12gb ## max amount of memory per node you require
#SBATCH --time=03-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=analyze_hmc
#SBATCH --output=../logs/analysis-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile and run python script.
source ~/profiles/auto.profile
conda activate gcr

python analyze_hmc.py