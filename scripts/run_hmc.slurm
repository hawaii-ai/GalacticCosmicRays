#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=shared # koa, sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow

#SBATCH --array=0-132 ## Negative intervals only 0-132 and for new yearly data that's 0-14, for old data negative + positive intervals it's 0-209
##BATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=best_hmc ## Job name
#SBATCH --output=job.out
#SBATCH --output=logs/hmc-best-optuna-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu
##SBATCH --begin=now+3hour ## x hours from now

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)
echo Running hmc_version $HMC_VERSION with train_size $TRAIN_SIZE model_version $MODEL_VERSION data version $DATA_VERSION bootstrap $BOOTSTRAP regularizer $REGULARIZER
echo Integrate $INTEGRATE par_equals_perr $PAR_EQUALS_PERR constant_vspoles $CONSTANT_VSPOLES File version $FILE_VERSION

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr-gpu

python hmc_distributed.py
