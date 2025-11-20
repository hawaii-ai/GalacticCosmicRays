#!/bin/bash

# Arguments (update me)
train_sizes=( 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
file_version='test_data_pamela_not_sampled' # 2023 is what we are using for this paper # 2024 is the yearly data, 2023 is the old data, test_data is NN test dataset
integrate='false' # If False, Chi2 is interpolated. If True, Chi2 is integrated.
par_equals_perr='false' # If True, only 3 parameters will be sampled by the HMC and pwr1par==pwr1perr and pwr2par==pwr2perr
constant_vspoles='false' # If True, vspoles is fixed to 400.0. If False, vspoles is specified in the data file.
hmc_version_number='v34_trial5_test_100000_better_error_'$file_version # Remember to update!
model_save_dir='../models/model_size_investigation_optuna_10082025_trial5_full' # Remember to update!
mcmc_or_hmc='hmc' # 'mcmc' or 'hmc'

# Run 
# for train_size in "${train_sizes[@]}"; do
#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc2' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc3' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc4' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc5' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init2' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init3' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init4' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d1' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init5' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d2' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d3' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d4' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

#   # Submit a new job
#   data_version='d5' # 'd1', 'd2'
#   bootstrap='b1' # 'b0' or 'b1'
#   model_version='init1' # 'init1', 'init2'
#   hmc_run='hmc1' # 'hmc1', 'hmc2'
#   hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

#   export MODEL_VERSION=$model_version
#   export HMC_VERSION=$hmc_version
#   export FILE_VERSION=$file_version
#   export INTEGRATE=$integrate
#   export PAR_EQUALS_PERR=$par_equals_perr
#   export CONSTANT_VSPOLES=$constant_vspoles
#   export TRAIN_SIZE=$train_size
#   export DATA_VERSION=$data_version
#   export BOOTSTRAP=$bootstrap
#   export MODEL_SAVE_DIR=$model_save_dir
#   export MCMC_OR_HMC=$mcmc_or_hmc
#   sbatch --export=ALL run_hmc.slurm

# done

# Submit a new job
data_version='d1' # 'd1', 'd2'
bootstrap='b0' # 'b0' or 'b1'
model_version='init1' # 'init1', 'init2'
train_size='1.0' # '1.0' for the full dataset
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
export MCMC_OR_HMC=$mcmc_or_hmc
sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init2' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc1' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init3' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc1' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init4' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc1' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init5' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc1' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init1' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc2' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init1' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc3' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init1' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc4' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init1' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='hmc5' # 'hmc1', 'hmc2'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm

# # Submit mcmc run
# # Submit a new job
# data_version='d1' # 'd1', 'd2'
# bootstrap='b0' # 'b0' or 'b1'
# model_version='init1' # 'init1', 'init2'
# train_size='1.0' # '1.0' for the full dataset
# hmc_run='mcmc1' # 'hmc1', 'hmc2'
# mcmc_or_hmc='mcmc' # 'mcmc' or 'hmc'
# hmc_version="${hmc_version_number}/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

# export MODEL_VERSION=$model_version
# export HMC_VERSION=$hmc_version
# export FILE_VERSION=$file_version
# export INTEGRATE=$integrate
# export PAR_EQUALS_PERR=$par_equals_perr
# export CONSTANT_VSPOLES=$constant_vspoles
# export TRAIN_SIZE=$train_size
# export DATA_VERSION=$data_version
# export BOOTSTRAP=$bootstrap
# export MODEL_SAVE_DIR=$model_save_dir
# export MCMC_OR_HMC=$mcmc_or_hmc
# sbatch --export=ALL run_hmc.slurm
