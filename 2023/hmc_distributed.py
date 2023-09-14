"""
Distributed HMC
Author: Peter July 2023
Edited by Linnea August 2023

Originally inspired by this example:
https://colab.research.google.com/github/tensorflow/probability/blob/master/spinoffs/oryx/examples/notebooks/probabilistic_programming.ipynb#scrollTo=nmjmxzGhN855

# New Requirements:
# conda install python=3.9 numpy scipy pandas matplotlib
# conda install -c anaconda cudatoolkit
# pip install --upgrade "jax[cuda12_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install jax pip tensorflow-probability
# pip install matplotlib

"""
import os
os.environ["KERAS_BACKEND"] = "jax"  # Must be specified before loading keras_core
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # CPU is faster for batchsize=1 inference.

import keras_core as kerasjk
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
#assert jax.default_backend() == 'gpu'
import utils
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
#import elegy # pip install elegy. # Trying to do this with keras core instead.
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
# %load_ext autoreload
# %autoreload 2

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
    

def index_mcmc_runs():
    """Make a list of combinations for which we want to run MCMC."""
    experiments = ['AMS02_H-PRL2021', 'PAMELA_H-ApJ2013', 'PAMELA_H-ApJL2018']
    dfs = []
    for experiment_name in experiments:
        filename = f'../data/2023/{experiment_name}_heliosphere.dat'
        df = utils.index_experiment_files(filename) 
        df['experiment_name'] = experiment_name
        df['filename_heliosphere'] = filename
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=0)
    return df

# Select experiment parameters
df = index_mcmc_runs()  # List of all ~200 experiments.
print(f'Found {df.shape[0]} combinations to run MCMC on. Performing MCMC on index {SLURM_ARRAY_TASK_ID}.')
df = df.iloc[SLURM_ARRAY_TASK_ID]

# Model specification
model_version = 'v2.0'
hmc_version = 'v10.0'

# Setup  output directory.
results_dir = f'../../results/{hmc_version}/'
Path(results_dir).mkdir(parents=True, exist_ok=True)

# Load observation data and define logprob. 
specified_parameters = utils.get_parameters(df.filename_heliosphere, df.interval)
data_path = f'../data/oct2022/{df.experiment_name}/{df.experiment_name}_{df.interval}.dat'  # This data is the same.
model_path = f'../models/model_{model_version}_{df.polarity}.keras'
seed = SLURM_ARRAY_TASK_ID + SLURM_ARRAY_JOB_ID
penalty = 1e6
target_log_prob = utils.define_log_prob(model_path, data_path, specified_parameters, penalty=penalty)

# Hyperparameters for MCMC
if DEBUG:
    # For running interactive tests.
    num_results = 500 #150000 #500000 # 10k takes 11min. About 1/5 of these accepted? now .97
    num_burnin_steps = 100 #500
    num_adaptation_steps = np.floor(.8*num_burnin_steps) #Somewhat smaller than number of burnin
    step_size = 1e-3 # 1e-3 (experiment?) # 1e-5 has 0.95 acc rate and moves. 1e-4 0.0 acc.
    max_tree_depth = 10 # Default=10. Smaller results in shorter steps. Larger takes memory.
else:
    num_results = 110_000 #1_000_000 #150000 #500000 # 10k takes 11min. About 1/5 of these accepted? now .97
    num_steps_between_results = 10 # Thinning
    num_burnin_steps = 100_000 #2500 #500
    num_adaptation_steps = np.floor(.8*num_burnin_steps) #Somewhat smaller than number of burnin
    step_size = 1e-1 # 1e-3 (experiment?) # 1e-5 has 0.95 acc rate and moves. 1e-4 0.0 acc.
    max_tree_depth = 10 # Default=10. Smaller results in shorter steps. Larger takes memory.
    max_energy_diff = 1000 #1e32 #1e21 # Default 1000.0. Divergent samples are those that exceed this.
    unrolled_leapfrog_steps = 1 # Default 1. The number of leapfrogs to unroll per tree expansion step

@jit
def run_chain(key, state):
    # Define kernel for mcmc to be the No U-Turn Sampler
    kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, step_size=step_size, max_tree_depth=max_tree_depth, 
                                     max_energy_diff=max_energy_diff, unrolled_leapfrog_steps=unrolled_leapfrog_steps)
    def trace_fn(_, pkr):
        return [pkr.log_accept_ratio,
                pkr.target_log_prob,
                pkr.step_size]
    
    # Adjust step size of mcmc kernel to have noisy steps
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8),
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    )
    def trace_fn(_, pkr):
        return [pkr.inner_results.log_accept_ratio,
                pkr.inner_results.target_log_prob,
                pkr.inner_results.step_size]
    
    # Run the mcmc chain
    samples, pkr = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        kernel=kernel,
        trace_fn=trace_fn,
        current_state=state,
        seed=key)
    
    return samples, pkr


start_time = time.time()
np.random.seed(seed)
state = np.random.random((5,)) # used for 29091984
#state = jnp.array(utils.minmax_scale_input(np.array([90, .5, 1.7, 1.4, 1.1 ]))) # This is ~MLE
#state = 0.5 * jnp.ones((5,), dtype='float32')
key = random.PRNGKey(seed)
samples_transformed_all, pkr_all = run_chain(key, state)
# Remove duplicates.
samples_transformed, pkr = utils.remove_consecutive_duplicates(samples_transformed_all, pkr_all, atol=0.0)
log_accept_ratio, log_probs, step_sizes  = pkr
#all_log_accept_ratio, all_log_probs, all_step_sizes = pkr_all
print('Finished in %d minutes.' % int((time.time() - start_time)//60))
print(f'Acceptance rate: {len(samples_transformed)/len(samples_transformed_all)}. Decrease step_size to increase rate.')

# Inverse transform samples.
samples = utils.untransform_input(samples_transformed)

# Save results: samples and plots
np.savetxt(fname=f'{results_dir}/samples_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=samples, delimiter=',')
np.savetxt(fname=f'{results_dir}/logprobs_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=log_probs, delimiter=',')
np.savetxt(fname=f'{results_dir}/logacceptratio_{SLURM_ARRAY_TASK_ID}.csv', X=log_accept_ratio, delimiter=',')
np.savetxt(fname=f'{results_dir}/stepsizes_{SLURM_ARRAY_TASK_ID}.csv', X=step_sizes, delimiter=',')

# Get NN predictions on these samples.
from preprocess.preprocess import transform_input, untransform_input
specified_parameters_transformed = transform_input(jnp.array(specified_parameters).reshape((1,-1)))
xs = utils._form_batch(samples_transformed, specified_parameters_transformed)
model = kerasjk.models.load_model(model_path)
predictions_transformed = model.predict(xs, verbose=2)
predictions = utils.untransform_output(predictions_transformed)
np.savetxt(fname=f'{results_dir}/predictions_{SLURM_ARRAY_TASK_ID}_{df.experiment_name}_{df.interval}_{df.polarity}.csv', X=predictions, delimiter=',')


