#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=kill-shared # sadow, gpu, kill-shared, shared
##SBATCH --partition=sadow
##SBATCH --account=sadow

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb ## max amount of memory per node you require
#SBATCH --time=03-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on sadow

#SBATCH --job-name=gcr_nn
#SBATCH --output=./logs/analysis-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr

# Create TRAIN_SIZE_PERCENT array, and run for different percentages of the training set size
TRAIN_SIZE_PERCENT=( 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 )
for i in "${TRAIN_SIZE_PERCENT[@]}"
do
    python nn_train_size_analysis.py --TRAIN_SIZE_PERCENT $i --POLARITY "pos"
    python nn_train_size_analysis.py --TRAIN_SIZE_PERCENT $i --POLARITY "neg"
done