#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=sadow ## koa, sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=sadow

##SBATCH --gres=gpu:1 ## Not sure if this is necessary or not. Might speed up computation, but often GPU utilization is super low for HMC sampling
#SBATCH --cpus-per-task=2
#SBATCH --mem=6gb ## max amount of memory per node you require
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=plot ## Job name
#SBATCH --output=../logs/analysis-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu
##SBATCH --begin=now+24hour ## x hours from now

# Load python profile and run python script.
source ~/profiles/auto.profile
conda activate gcr-gpu

# python plot_hmc_train_size_individual_runs.py
python $FILE