#!/bin/bash

# Arguments (update me)
train_size='0.9'
file_version='2023' # 2024 is the yearly data, 2023 is the old data
integrate='false' # If False, Chi2 is interpolated. If True, Chi2 is integrated.
par_equals_perr='false' # If True, only 3 parameters will be sampled by the HMC and pwr1par==pwr1perr and pwr2par==pwr2perr
constant_vspoles='false' # If True, vspoles is fixed to 400.0. If False, vspoles is specified in the data file.
hmc_version_number='debug'
model_save_dir='../models/model_size_investigation_optuna_09122025_lowestlinearrelu' # Remember to update!

# Submit a new job
data_version='d1' # 'd1', 'd2'
bootstrap='b1' # 'b0' or 'b1'
model_version='init1' # 'init1', 'init2'
hmc_run='hmc1' # 'hmc1', 'hmc2'
hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

export MODEL_VERSION=$model_version
export HMC_VERSION=$hmc_version
export FILE_VERSION=$file_version
export INTEGRATE=$integrate
export PAR_EQUALS_PERR=$par_equals_perr
export CONSTANT_VSPOLES=$constant_vspoles
export TRAIN_SIZE=$train_size
export DATA_VERSION=$data_version
export BOOTSTRAP=$bootstrap
export MODEL_SAVE_DIR=$model_save_dir
sbatch --export=ALL run_hmc.slurm