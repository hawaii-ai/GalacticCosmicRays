#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu # sadow, gpu, kill-shared
##SBATCH --partition=sadow
##SBATCH --account=sadow
##SBATCH --nodelist=gpu-0008

##SBATCH --array=0-110   # Array range
##SBATCH --array=0-47  # Pamela 2013
##SBATCH --array=42 # PRL2021, Need to do 42
##SBATCH --array=40-49 # PRL2021, Need to do 42
#SBATCH --array=0-35  # Pamela 2018, needed to redo lots
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=15000 ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=hmc
##SBATCH --output=job.out
#SBATCH --output=logs/job-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=psadow@hawaii.edu

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate jax_2023 

# UPDATE ME
#'AMS02_H-PRL2021' # 'AMS02_H-PRL2018', 'AMS02_H-PRL2021', 'PAMELA_H-ApJ2013', 'PAMELA_H-ApJL2018'
export EXPERIMENT_NAME=PAMELA_H-ApJL2018
echo $EXPERIMENT_NAME

python hmc_distributed.py

