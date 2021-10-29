#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu # sadow, gpu, kill-shared
##SBATCH --account=sadow

#SBATCH --array=1-50                # Array range
##SBATCH --output=array_%A-%a.log    # Standard output and error log

##SBATCH --nodelist=gpu-0008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8400 ## max amount of memory per node you require
#SBATCH --time=0-08:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=hmc
#SBATCH --output=job.out
##SBATCH --output=job-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=myemail@hawaii.edu

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate jax
mkdir -p ../../results/job_$SLURM_ARRAY_JOB_ID
python hmc_distributed.py
