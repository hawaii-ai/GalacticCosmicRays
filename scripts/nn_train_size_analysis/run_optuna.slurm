#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=sadow ##, gpu, kill-shared, shared
#SBATCH --account=sadow

##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb ## max amount of memory per node you require
#SBATCH --time=07-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on sadow

#SBATCH --job-name=optuna
#SBATCH --output=./logs/optuna-%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu
# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate gcr

# nvidia-smi
# echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Running on node: $(hostname)"

echo "Now running optuna_nn.py..."
python optuna_nn.py 
