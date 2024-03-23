#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=koa # koa, sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=koa
##SBATCH --nodelist=gpu-0008

#SBATCH --array=0-132 ## Negative intervals only
##SBATCH --array=0-209
##BATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12000 ## max amount of memory per node you require
#SBATCH --time=30-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=hmc_vspoles400_parequalsperr
#SBATCH --output=job.out
#SBATCH --output=logs/job-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr

# UPDATE ME
#'AMS02_H-PRL2021' # 'AMS02_H-PRL2018', 'AMS02_H-PRL2021', 'PAMELA_H-ApJ2013', 'PAMELA_H-ApJL2018'
#export EXPERIMENT_NAME=PAMELA_H-ApJL2018
#echo $EXPERIMENT_NAME

python hmc_distributed.py