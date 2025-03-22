#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=sadow # sadow, gpu, kill-shared, shared
#SBATCH --account=sadow

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb ## max amount of memory per node you require
#SBATCH --time=14-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on sadow

#SBATCH --job-name=gcr_nn
#SBATCH --output=./logs/analysis-%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate gcr

# Create TRAIN_SIZE_PERCENT array, and run for different percentages of the training set size
# currently logspaced 100 points between 0.00001 and 1
size_percents=( 0.00001  0.000011 0.000013 0.000014 0.000016 0.000018 0.00002  0.000023 0.000025 0.000028 0.000032 0.000036 0.00004  0.000045 0.000051 0.000057 0.000064 0.000072 0.000081 0.000091 0.000102 0.000115 0.000129 0.000145 0.000163 0.000183 0.000206 0.000231 0.00026  0.000292 0.000327 0.000368 0.000413 0.000464 0.000521 0.000586 0.000658 0.000739 0.00083  0.000933 0.001048 0.001177 0.001322 0.001485 0.001668 0.001874 0.002105 0.002364 0.002656 0.002984 0.003352 0.003765 0.004229 0.004751 0.005337 0.005995 0.006734 0.007565 0.008498 0.009545 0.010723 0.012045 0.01353  0.015199 0.017074 0.019179 0.021544 0.024201 0.027186 0.030539 0.034305 0.038535 0.043288 0.048626 0.054623 0.061359 0.068926 0.077426 0.086975 0.097701 0.10975  0.123285 0.138489 0.155568 0.174753 0.196304 0.220513 0.247708 0.278256 0.312572 0.351119 0.394421 0.443062 0.497702 0.559081 0.628029 0.70548  0.792483 0.890215 1)
for i in "${size_percents[@]}"
do
    echo "train size percentage: $i"
    python nn_train_size_analysis.py --polarity "pos" --train_size_percent $i
    python nn_train_size_analysis.py --polarity "neg" --train_size_percent $i 
done
