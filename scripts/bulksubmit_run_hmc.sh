#!/bin/bash

# Arguments (update me)
train_sizes=( 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 ) # and 'fulldataset' for the full dataset
model_versions=( 'v3' 'v4' ) # v2.0 is MSE NN, v3.0 is MAE NN. If doing fractions of train_size, model_version corresponds to which random sample of train data model was trained on (only v1 and v2)
file_version='2023' # 2024 is the yearly data, 2023 is the old data
integrate='false' # If False, Chi2 is interpolated. If True, Chi2 is integrated.
par_equals_perr='false' # If True, only 3 parameters will be sampled by the HMC and pwr1par==pwr1perr and pwr2par==pwr2perr
constant_vspoles='false' # If True, vspoles is fixed to 400.0. If False, vspoles is specified in the data file.

# Run 
for train_size in "${train_sizes[@]}"; do
  for model_version in "${model_versions[@]}"; do
    # Set the HMC version based on the model version and train size
    hmc_version="v27/${model_version}_${train_size}"

    # Submit the job to the SLURM scheduler
    sbatch --export=ALL,MODEL_VERSION=$model_version,HMC_VERSION=$hmc_version,FILE_VERSION=$file_version,INTEGRATE=$integrate,PAR_EQUALS_PERR=$par_equals_perr,CONSTANT_VSPOLES=$constant_vspoles,TRAIN_SIZE=$train_size run_hmc.slurm;
  done
done