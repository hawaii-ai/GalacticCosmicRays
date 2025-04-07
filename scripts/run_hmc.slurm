#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=shared,gpu # koa, sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow

#SBATCH --array=0-132 ## Negative intervals only 0-132 and for new yearly data that's 0-14, for old data negative + positive intervals it's 0-209
##BATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12000 ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=hmc_train_fraction ## Job name
#SBATCH --output=job.out
#SBATCH --output=logs/hmc-train-size-analysis-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu
#SBATCH --begin=now+6hours ## 12 hours from now

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)
echo Running hmc_version $HMC_VERSION with train_size $TRAIN_SIZE and model_version $MODEL_VERSION

# Load python profile, the`n call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr

python hmc_distributed.py