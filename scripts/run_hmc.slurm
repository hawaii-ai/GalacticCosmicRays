#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

# For other data versions, we have 133 negative intervals (0-132)
#SBATCH --array=0-132 ## Negative intervals only 0-132 and for new yearly data that's 0-14, for old data negative + positive intervals it's 0-209
#SBATCH --job-name=ppc-sim-err ## Job name
#SBATCH --output=logs/ppc-sim-obs-trial5-test-10000-%A-%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID

#SBATCH --partition=shared,kill-shared #shared,kill-shared ## koa, sadow, gpu, shared, kill-shared, exclusive-long
##SBATCH --account=sadow
##SBATCH --gres=gpu:1 ## Not sure if this is necessary or not. Might speed up computation, but often GPU utilization is super low for HMC sampling
#SBATCH --cpus-per-task=1
#SBATCH --mem=6gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --mail-type=FAIL ##START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu
##SBATCH --begin=now+24hour ## x hours from now

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)
echo Running hmc_version $HMC_VERSION with train_size $TRAIN_SIZE model_version $MODEL_VERSION data version $DATA_VERSION bootstrap $BOOTSTRAP regularizer $REGULARIZER
echo Integrate $INTEGRATE par_equals_perr $PAR_EQUALS_PERR constant_vspoles $CONSTANT_VSPOLES File version $FILE_VERSION NN Model save dir $MODEL_SAVE_DIR

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr-gpu

python hmc_distributed.py
