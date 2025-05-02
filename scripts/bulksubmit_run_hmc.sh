#!/bin/bash

# Arguments (update me)
train_sizes=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.0001 0.001 0.01 )
file_version='2023' # 2024 is the yearly data, 2023 is the old data
integrate='false' # If False, Chi2 is interpolated. If True, Chi2 is integrated.
par_equals_perr='false' # If True, only 3 parameters will be sampled by the HMC and pwr1par==pwr1perr and pwr2par==pwr2perr
constant_vspoles='false' # If True, vspoles is fixed to 400.0. If False, vspoles is specified in the data file.

# Run 
for train_size in "${train_sizes[@]}"; do
  # Submit a new job
  data_version='d1' # 'd1', 'd2'
  bootstrap='b1' # 'b0' or 'b1'
  model_version='init1' # 'init1', 'init2'
  hmc_run='hmc1' # 'hmc1', 'hmc2'
  hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

  sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

# Submit a new job
  data_version='d1' # 'd1', 'd2'
  bootstrap='b1' # 'b0' or 'b1'
  model_version='init1' # 'init1', 'init2'
  hmc_run='hmc2' # 'hmc1', 'hmc2'
  hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

  sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

  # Submit a new job
  data_version='d1' # 'd1', 'd2'
  bootstrap='b1' # 'b0' or 'b1'
  model_version='init2' # 'init1', 'init2'
  hmc_run='hmc1' # 'hmc1', 'hmc2'
  hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

  sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

  # Submit a new job
  data_version='d2' # 'd1', 'd2'
  bootstrap='b1' # 'b0' or 'b1'
  model_version='init1' # 'init1', 'init2'
  hmc_run='hmc1' # 'hmc1', 'hmc2'
  hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

  sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

done

# Submit a new job
data_version='d1' # 'd1', 'd2'
bootstrap='b0' # 'b0' or 'b1'
model_version='init1' # 'init1', 'init2'
train_size='1.0' # '1.0' for the full dataset
hmc_run='hmc1' # 'hmc1', 'hmc2'
hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

# Submit a new job
data_version='d1' # 'd1', 'd2'
bootstrap='b0' # 'b0' or 'b1'
model_version='init2' # 'init1', 'init2'
train_size='1.0' # '1.0' for the full dataset
hmc_run='hmc1' # 'hmc1', 'hmc2'
hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm

# Submit a new job
data_version='d1' # 'd1', 'd2'
bootstrap='b0' # 'b0' or 'b1'
model_version='init1' # 'init1', 'init2'
train_size='1.0' # '1.0' for the full dataset
hmc_run='hmc2' # 'hmc1', 'hmc2'
hmc_version="v29/${data_version}_${bootstrap}_${model_version}_${hmc_run}_${train_size}"

sbatch --export=ALL,\
MODEL_VERSION=$model_version,\
HMC_VERSION=$hmc_version,\
FILE_VERSION=$file_version,\
INTEGRATE=$integrate,\
PAR_EQUALS_PERR=$par_equals_perr,\
CONSTANT_VSPOLES=$constant_vspoles,\
TRAIN_SIZE=$train_size,\
DATA_VERSION=$data_version,\
BOOTSTRAP=$bootstrap \
run_hmc.slurm