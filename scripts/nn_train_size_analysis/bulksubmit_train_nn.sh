#!/bin/bash

# # Arguments (update me)
sizes=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.0001 0.001 0.01 )
polarity='neg' # 'pos' or 'neg'
save_dir='../../models/model_size_investigation_optuna_10082025_trial5_full'

# Loop over each size and submit for various model versions and bootstrap options
# for train_size in "${sizes[@]}"
# do
#     echo "Submitting jobs for train_size_fraction: $train_size"

#     # Submit with bootstrap = True (b1)
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init1',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init2',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init3',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init4',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init5',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d2',MODEL_VERSION='init1',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d3',MODEL_VERSION='init1',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d4',MODEL_VERSION='init1',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
#     sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d5',MODEL_VERSION='init1',TRAIN_SIZE=$train_size,BOOTSTRAP='b1',SAVE_DIR=$save_dir run_train_nn.slurm
# done

# # Submit with bootstrap = False (b0) for train_size = 1.0
# sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init1',TRAIN_SIZE='1.0',BOOTSTRAP='b0',SAVE_DIR=$save_dir run_train_nn.slurm
# sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init2',TRAIN_SIZE='1.0',BOOTSTRAP='b0',SAVE_DIR=$save_dir run_train_nn.slurm
sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init3',TRAIN_SIZE='1.0',BOOTSTRAP='b0',SAVE_DIR=$save_dir run_train_nn.slurm
sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init4',TRAIN_SIZE='1.0',BOOTSTRAP='b0',SAVE_DIR=$save_dir run_train_nn.slurm
sbatch --export=ALL,POLARITY=$polarity,DATA_VERSION='d1',MODEL_VERSION='init5',TRAIN_SIZE='1.0',BOOTSTRAP='b0',SAVE_DIR=$save_dir run_train_nn.slurm