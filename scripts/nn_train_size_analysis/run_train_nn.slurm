#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu # sadow, gpu, kill-shared, shared
##SBATCH --account=sadow

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36gb ## max amount of memory per node you require
#SBATCH --time=03-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on sadow

#SBATCH --job-name=nn_besthp
#SBATCH --output=./logs/optuna-september12-%A.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate gcr-gpu

# nvidia-smi
# echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Running on node: $(hostname)"

echo "Running training with:"
echo "  Polarity: $POLARITY"
echo "  Data version: $DATA_VERSION"
echo "  Model version: $MODEL_VERSION"
echo "  Train size: $TRAIN_SIZE"
echo "  Bootstrap: $BOOTSTRAP"
echo "  Save directory: $SAVE_DIR"

echo "Now running the training script..."
python nn_train_size_analysis.py \
    --polarity "$POLARITY" \
    --data_version "$DATA_VERSION" \
    --model_version "$MODEL_VERSION" \
    --train_size_fraction "$TRAIN_SIZE" \
    --bootstrap "$BOOTSTRAP" \
    --save_dir "$SAVE_DIR" \
