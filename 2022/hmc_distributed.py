# Distributed HMC with Oryx on Mana
# Author: Peter Nov 5 2022

# New Requirements:
# conda install python=3.9 numpy scipy pandas matplotlib
### Version 0.3.24 didn't work: pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install -Iv jax==0.3.16 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install elegy tfp-nightly

# Old Requirements: 
# !module load system/CUDA/11.0.2 
# !pip install --upgrade jax jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html 
# !pip install tensorflow-io oryx elegy

import os
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
assert jax.default_backend() == 'gpu'
import utils
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import elegy # pip install elegy. 
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# Arguments
SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
SLURM_ARRAY_JOB_ID = int(os.environ['SLURM_ARRAY_JOB_ID'])
EXPERIMENT_NAME = os.environ['EXPERIMENT_NAME'] #'AMS02_H-PRL2021' # 'AMS02_H-PRL2018', 'AMS02_H-PRL2021', 'PAMELA_H-ApJ2013', 'PAMELA_H-ApJL2018'

# Setup  output directory.
results_dir = f'../../results/{EXPERIMENT_NAME}/'
Path(results_dir).mkdir(parents=True, exist_ok=True)

# Load observation data and define logprob. 
filename_heliosphere = f'../data/oct2022/{EXPERIMENT_NAME}_heliosphere.dat'
interval = utils.get_interval(filename_heliosphere, SLURM_ARRAY_TASK_ID) # e.g. '20110520-20110610'
alpha, cmf = utils.get_alpha_cmf(filename_heliosphere, interval)
data_path = f'../data/oct2022/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_{interval}.dat'
model_path = '../models/model_2_256_selu_l21e-6' #'model_2_256_selu_l21e-6_do' # 'model_2_256_selu_l21e-6' #'model_2_256_selu'
seed = SLURM_ARRAY_TASK_ID + SLURM_ARRAY_JOB_ID
target_log_prob = utils.define_log_prob(model_path, data_path, alpha, cmf, penalty=0)

# Hyperparameters
num_results = 150000 #500000 # 10k takes 11min. About 1/5 of these accepted? now .97
num_burnin_steps = 1000 #500
num_adaptation_steps = np.floor(.8*num_burnin_steps) #Somewhat smaller than number of burnin
target_accept_prob = 0.3
step_size = 1e-3 # 1e-5 has 0.95 acc rate and moves. 1e-4 0.0 acc.
num_leapfrog_steps = 100
max_tree_depth = 10 # Default=10. Smaller results in shorter steps. Larger takes memory.

@jit
def run_chain(key, state):
    # Example from https://colab.research.google.com/github/tensorflow/probability/blob/master/spinoffs/oryx/examples/notebooks/probabilistic_programming.ipynb#scrollTo=nmjmxzGhN855
    
    # kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #             #tfp.mcmc.HamiltonianMonteCarlo(flat_log_prob, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps),
    #             tfp.mcmc.HamiltonianMonteCarlo(target_log_prob, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps),
    #             num_adaptation_steps=num_adaptation_steps,
    #             target_accept_prob=target_accept_prob)
    # def trace_fn(_, pkr):
    #     return [pkr.inner_results.log_accept_ratio,
    #             pkr.inner_results.accepted_results.target_log_prob,
    #             pkr.inner_results.accepted_results.step_size]
    
    #kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)
    max_energy_diff = 1000 #1e32 #1e21 # Default 1000.0. Divergent samples are those that exceed this.
    kernel = tfp.mcmc.NoUTurnSampler(target_log_prob, step_size=step_size, max_tree_depth=max_tree_depth, 
                                     max_energy_diff=max_energy_diff, unrolled_leapfrog_steps=1,)
    def trace_fn(_, pkr):
        return [pkr.log_accept_ratio,
                pkr.target_log_prob,
                pkr.step_size]
    
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
    
    samples, pkr = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        kernel=kernel,
        trace_fn=trace_fn,
        current_state=state,
        seed=key)
    
    return samples, pkr


start_time = time.time()
np.random.seed(seed)
state = np.random.random((5,)) #jnp.random.random((5,), dtype='float32') # used for 29091984
#state = jnp.array(utils.minmax_scale_input(np.array([90, .5, 1.7, 1.4, 1.1 ]))) # This is ~MLE
#state = 0.5 * jnp.ones((5,), dtype='float32')
key = random.PRNGKey(seed)
unnormalized_samples, pkr = run_chain(key, state)
print('Finished in %d minutes.' % int((time.time() - start_time)//60))

# De-normalize samples and remove duplicates.
unnormalized_samples = unnormalized_samples.to_py()
all_samples = utils.deminmax_scale_input(unnormalized_samples)
samples, pkr_select = utils.remove_consecutive_duplicates(all_samples, pkr, atol=0.0)
all_log_accept_ratio, all_log_probs, all_step_sizes = pkr
log_accept_ratio, log_probs, step_sizes  = pkr_select
#samples, (log_accept_ratio, log_probs, step_sizes) = utils.remove_consecutive_duplicates(all_samples, (log_accept_ratio, log_probs, step_sizes))
print(f'Acceptance rate: {len(samples)/len(all_samples)}. Decrease step_size to increase rate.')

# Save results: samples and plots
np.savetxt(fname=f'{results_dir}/samples_{SLURM_ARRAY_TASK_ID}.csv', X=samples, delimiter=',')
np.savetxt(fname=f'{results_dir}/logacceptratio_{SLURM_ARRAY_TASK_ID}.csv', X=log_accept_ratio, delimiter=',')
np.savetxt(fname=f'{results_dir}/log_probs_{SLURM_ARRAY_TASK_ID}.csv', X=log_probs, delimiter=',')
np.savetxt(fname=f'{results_dir}/stepsizes_{SLURM_ARRAY_TASK_ID}.csv', X=step_sizes, delimiter=',')

