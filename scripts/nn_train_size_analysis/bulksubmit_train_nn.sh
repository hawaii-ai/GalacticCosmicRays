#!/bin/bash

# Arguments (update me)
sizes=( 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
polarity='neg' # 'pos' or 'neg'

# Loop over each size and submit for various model versions and bootstrap options
for train_size in "${sizes[@]}"
do
    echo "Submitting jobs for train_size_fraction: $train_size"

    # Submit with bootstrap = True
    sbatch --export=ALL,POLARITY=$polarity,MODEL_VERSION='v1',TRAIN_SIZE=$train_size,BOOTSTRAP=true run_train_nn.slurm
    sbatch --export=ALL,POLARITY=$polarity,MODEL_VERSION='v1.1',TRAIN_SIZE=$train_size,BOOTSTRAP=true run_train_nn.slurm
    sbatch --export=ALL,POLARITY=$polarity,MODEL_VERSION='v2',TRAIN_SIZE=$train_size,BOOTSTRAP=true run_train_nn.slurm

    # Submit with bootstrap = False
    sbatch --export=ALL,POLARITY=$polarity,MODEL_VERSION='v1.1',TRAIN_SIZE=$train_size,BOOTSTRAP=false run_train_nn.slurm
done
