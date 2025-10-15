"""
Distributed HMC
Author: Peter July 2023
Updated: Linnea March 2025 (and many times since)
"""

import os
import sys
os.environ["KERAS_BACKEND"] = "jax"  # Must be specified before loading keras_core
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # CPU is faster for batchsize=1 inference.

import utils as utils
from preprocess import transform_input, untransform_input

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import tensorflow_probability
xmcmc = tensorflow_probability.experimental.mcmc
import keras_core as keras

# Load tabular embedding layers
sys.path.append('./nn_train_size_analysis/')
from rtdl_num_embeddings_keras import (
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
)

# Run in DEBUG mode if there is no slurm task id.
try:
    SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    SLURM_ARRAY_JOB_ID = int(os.environ['SLURM_ARRAY_JOB_ID'])
    DEBUG = False
except:
    print(f'DEBUG MODE: Could not load SLURM_ARRAY_TASK_ID. Assuming we are debugging.')
    SLURM_ARRAY_TASK_ID = 0
    SLURM_ARRAY_JOB_ID = 0
    DEBUG = True

# Parse arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise UserWarning(f'Boolean value expected for parameter {v}.')


model_version = os.getenv('MODEL_VERSION', default='v3.0')
data_version = os.getenv('DATA_VERSION', default='d1')
hmc_version = os.getenv('HMC_VERSION', default='v25.0')
file_version = os.getenv('FILE_VERSION', default='2024')
integrate = str2bool(os.getenv('INTEGRATE', default=False))
par_equals_perr = str2bool(os.getenv('PAR_EQUALS_PERR', default=False))
constant_vspoles = str2bool(os.getenv('CONSTANT_VSPOLES', default=False))
train_size_env = os.getenv('TRAIN_SIZE', default=None)
train_size = float(train_size_env) if train_size_env is not None else None
bootstrap = os.getenv('BOOTSTRAP', default='b0')
model_save_dir = os.getenv('MODEL_SAVE_DIR', default=None)

# Select experiment parameters
df = utils.index_mcmc_runs(file_version=file_version)  # List of all experiments (0-209) for '2023', 0-14 for '2024'
print(f'Found {df.shape[0]} combinations to run MCMC on. Performing MCMC on index {SLURM_ARRAY_TASK_ID}.')
df = df.iloc[SLURM_ARRAY_TASK_ID]

# Setup  output directory.
results_dir = f'../../results/{hmc_version}/'
Path(results_dir).mkdir(parents=True, exist_ok=True)
print(f'Running HMC version {hmc_version} on model version {model_version} and train_size {train_size}. The results will be saved in {results_dir}.')

# Load observation data and define logprob. 
if file_version == '2023': 
    data_path = f'../data/oct2022/{df.experiment_name}/{df.experiment_name}_{df.interval}.dat'  # This data is the same.
elif file_version == '2024': 
    year = 2000 + SLURM_ARRAY_TASK_ID # assumes only negative intervals. If otherwise, fix this
    data_path = f'../data/2024/yearly/{year}.dat'
else:
    raise ValueError(f"Invalid file_version {file_version}. Must be '2023' or '2024'.")

# Load NN model
if train_size is not None:
    save_name = f'data_{data_version}_bootstrap_{bootstrap}_model_{model_version}_train_size_{train_size}_{df.polarity}'
    model_path = f'{model_save_dir}/{save_name}.keras'  # Must end with keras.
    print(f'Loading model from {model_path}.')
else:
    model_path = f'../models/model_{model_version}_{df.polarity}.keras'

# Define parameters for HMC
seed = SLURM_ARRAY_TASK_ID + SLURM_ARRAY_JOB_ID
penalty = 1e6
specified_parameters = utils.get_parameters(df.filename_heliosphere, df.interval, constant_vspoles=constant_vspoles)

# Number of parameters for HMC to sample. 5 normally, 3 if par_equals_perr=True
if par_equals_perr:
    num_params = 3
else:
    num_params = 5

target_log_prob = utils.define_log_prob(model_path, data_path, specified_parameters, penalty=penalty, integrate=integrate, par_equals_perr=par_equals_perr)

# Hyperparameters for MCMC
if DEBUG:
    # For running interactive tests.
    mcmc_or_hmc = 'hmc' # 'mcmc' or 'hmc'
    num_results = 500 
    num_steps_between_results = 0 # Thinning
    num_burnin_steps = 100 # Number of steps before beginning sampling
    num_adaptation_steps = np.floor(.8*num_burnin_steps) # Default is .8*num_burnin_steps. Somewhat smaller than number of burnin
    step_size = 1e-1 # Smaller values raise acceptance, but mean the space is not explored as well. Automatically shrinks step size to achieve target_accept_prob
    target_accept_prob = 0.75 # Default is 0.75, normally want between 0.6-0.9
    max_tree_depth = 10 # Default 10. Smaller results in shorter steps. Larger takes memory.
    max_energy_diff = 1000 # Default 1000.0. Divergent samples are those that exceed this.
    unrolled_leapfrog_steps = 1 # Default 1. The number of leapfrogs to unroll per tree expansion step

    scale = 1e-2 # for mcmc

else:
    mcmc_or_hmc = 'hmc' # 'mcmc' or 'hmc'
    num_results = 100_000 #110_000 for hmc, 400_000 for mcmc
    num_steps_between_results = 10 # Thinning
    num_burnin_steps = 1_000 # Number of steps before beginning sampling

    # Note: this is just for mcmc
    scale = 1e-2

    # Note: below parameters are only for hmc
    num_adaptation_steps = np.floor(.8*num_burnin_steps) #Somewhat smaller than number of burnin
    step_size = 1e-4 # 1e-4 is good for hmc
    max_tree_depth = 10 # Default=10. Smaller results in shorter steps. Larger takes memory.
    max_energy_diff = 1000 # Default 1000.0. Divergent samples are those that exceed this.
    unrolled_leapfrog_steps = 1 # Default 1. The number of leapfrogs to unroll per tree expansion step
    target_accept_prob = 0.75 # the automatic value

print(f'Running {mcmc_or_hmc} with {num_results} samples, {num_burnin_steps} burn-in steps, and {num_steps_between_results} steps between results.')
print(f'Hyperparameters: step_size={step_size}, max_tree_depth={max_tree_depth}, max_energy_diff={max_energy_diff}, unrolled_leapfrog_steps={unrolled_leapfrog_steps}, target_accept_prob={target_accept_prob}, num_adaptation_steps={num_adaptation_steps}, and scale={scale}.')

@jit
def run_chain(key, state):
    if mcmc_or_hmc == 'mcmc':
        # Kernel for MCMC is RandomWalkMetropolis
        base_kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale),
        )

        def traced_fields(_, pkr):
            return {
                'log_accept_ratio': pkr.inner_results.log_accept_ratio,
            }
        
    else:
        # Kernel for HMC is NoUTurnSampler
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob, 
            step_size=step_size, 
            max_tree_depth=max_tree_depth, 
            max_energy_diff=max_energy_diff, 
            unrolled_leapfrog_steps=unrolled_leapfrog_steps
        )
    
        # Adjust step size of mcmc kernel to have noisy steps
        base_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            target_accept_prob=target_accept_prob
        )

        def traced_fields(_, pkr):
            # Trace all possible items. Possibilities:
            # 'count', 'energy', 'grads_target_log_prob', 'has_divergence', 'index', 'is_accepted', 'leapfrogs_taken', 'log_accept_ratio', 'reach_max_depth', 'seed', 'step_size', 'target_log_prob'
            # count, index, seed, and grads_target_log_prob give errors if traced
            return {
                'log_accept_ratio': pkr.inner_results.log_accept_ratio,
                'target_log_prob': pkr.inner_results.target_log_prob,
                'step_size': pkr.inner_results.step_size,
                'is_accepted': pkr.inner_results.is_accepted,
                'energy': pkr.inner_results.energy,
                'has_divergence': pkr.inner_results.has_divergence,
                'leapfrogs_taken': pkr.inner_results.leapfrogs_taken,
                'reach_max_depth': pkr.inner_results.reach_max_depth,
            }

    # With reductions means we have to unwrap once more
    def trace_fn(_, pkr):
        pkr = pkr.inner_results
        return traced_fields(_, pkr)
    
    # add progress bar to the kernel
    progress = xmcmc.ProgressBarReducer(num_results=num_results)  # prints one line that updates 
    kernel = xmcmc.WithReductions(base_kernel, progress)

    # Run the mcmc chain
    samples, pkr = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        kernel=kernel,
        trace_fn=trace_fn,
        current_state=state,
        seed=key
        )
    
    return samples, pkr

# Run the chain with a random seed 
start_time = time.time()
np.random.seed(seed)
state = np.random.random((num_params,)) # used for 29091984
key = random.PRNGKey(seed)
samples_transformed_all, pkr_all = run_chain(key, state)

# Remove duplicates.
samples_transformed, pkr = utils.remove_consecutive_duplicates(samples_transformed_all, pkr_all, atol=0.0)

print('Finished in %d minutes.' % int((time.time() - start_time)//60))
print(f'Acceptance rate: {len(samples_transformed)/len(samples_transformed_all)}. Decrease step_size to increase rate.')

# If par==perr, then only predicting ['cpa', 'pwr1par', 'pwr2par']. Need to create array of ['cpa', 'pwr1par', 'pwr1par', 'pwr2par', 'pwr2par']
# Need to adjust every row in samples to be [cpa, pwr1par, pwr1par, pwr2par, pwr2par] where pwr1par==pwr1perr and pwr2par==pwr2perr
if par_equals_perr:
    expanded_samples = np.zeros((samples_transformed.shape[0], 5))
    expanded_samples = np.column_stack((samples_transformed[:, 0], 
                                    samples_transformed[:, 1], 
                                    samples_transformed[:, 1], 
                                    samples_transformed[:, 2], 
                                    samples_transformed[:, 2]))
    samples_transformed = expanded_samples

# Inverse transform samples.
samples = utils.untransform_input(samples_transformed)

# Different values are traced and returned depending on mcmc or hmc
if mcmc_or_hmc == 'mcmc': 
    log_accept_ratio = pkr['log_accept_ratio']
else: 
    if DEBUG:
        for key, a in pkr.items():
            print(f'{key}: {a}, mean: {jnp.mean(a)}, std: {jnp.std(a)}, min: {jnp.min(a)}, max: {jnp.max(a)}\n\n')
        print(f"Samples: {samples}")

    log_accept_ratio = pkr['log_accept_ratio']
    log_probs = pkr['target_log_prob']
    step_sizes = pkr['step_size']
    has_divergences = pkr['has_divergence']
    is_accepteds = pkr['is_accepted']
    leapfrogs_takens = pkr['leapfrogs_taken']
    reach_max_depths = pkr['reach_max_depth']
    energies = pkr['energy']

# Generate samples and save results if not in DEBUG mode
# Save results: samples and plots
np.savetxt(fname=f'{results_dir}/samples_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=samples, delimiter=',')
np.savetxt(fname=f'{results_dir}/logacceptratio_{SLURM_ARRAY_TASK_ID}.csv', X=log_accept_ratio, delimiter=',')
if mcmc_or_hmc == 'hmc':
    np.savetxt(fname=f'{results_dir}/stepsizes_{SLURM_ARRAY_TASK_ID}.csv', X=step_sizes, delimiter=',')
    np.savetxt(fname=f'{results_dir}/logprobs_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=log_probs, delimiter=',')
    np.savetxt(fname=f'{results_dir}/has_divergences_{SLURM_ARRAY_TASK_ID}.csv', X=has_divergences, delimiter=',')
    np.savetxt(fname=f'{results_dir}/is_accepteds_{SLURM_ARRAY_TASK_ID}.csv', X=is_accepteds, delimiter=',')
    np.savetxt(fname=f'{results_dir}/leapfrogs_takens_{SLURM_ARRAY_TASK_ID}.csv', X=leapfrogs_takens, delimiter=',')
    np.savetxt(fname=f'{results_dir}/reach_max_depths_{SLURM_ARRAY_TASK_ID}.csv', X=reach_max_depths, delimiter=',')
    np.savetxt(fname=f'{results_dir}/energies_{SLURM_ARRAY_TASK_ID}.csv', X=energies, delimiter=',')

# Get NN predictions on these samples.
specified_parameters_transformed = transform_input(np.array(specified_parameters).reshape((1,-1)))
xs = utils._form_batch(samples_transformed, specified_parameters_transformed)
model = keras.models.load_model(model_path)
predictions_transformed = model.predict(xs, verbose=2)
predictions = utils.untransform_output(predictions_transformed)
np.savetxt(fname=f'{results_dir}/predictions_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=predictions, delimiter=',')


