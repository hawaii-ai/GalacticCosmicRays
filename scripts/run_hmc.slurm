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

#SBATCH --job-name=hmc_0.0001
#SBATCH --output=job.out
#SBATCH --output=logs/hmc-train-size-analysis-%A_%a.out ## Filled with %A=SLURM_JOB_ID, %a=SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

echo This is job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID running on $HOSTNAME
echo $(date +%F-%H-%M-%S)

# Load python profile, the`n call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
conda activate gcr

# Arguments (update me)
train_size='0.0001' # 0.1, 0.01, 0.001, 0.0001, fulldataset
model_version='v3.0' # v2.0 is MSE NN, v3.0 is MAE NN
hmc_version="v26/$train_size.1"
file_version='2023' # 2024 is the yearly data, 2023 is the old data
integrate='false' # If False, Chi2 is interpolated. If True, Chi2 is integrated.
par_equals_perr='false' # If True, only 3 parameters will be sampled by the HMC and pwr1par==pwr1perr and pwr2par==pwr2perr
constant_vspoles='false' # If True, vspoles is fixed to 400.0. If False, vspoles is specified in the data file.

# Run 
echo "running hmc_version $hmc_version with train_size $train_size"
python hmc_distributed.py --model_version $model_version --hmc_version $hmc_version --file_version $file_version --integrate $integrate --par_equals_perr $par_equals_perr --constant_vspoles $constant_vspoles --train_size $train_size
