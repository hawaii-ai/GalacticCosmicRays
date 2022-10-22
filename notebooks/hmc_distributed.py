# Distributed HMC with Oryx on Mana
# Author: Peter Oct 28 2021

# Requirements: 
# !module load system/CUDA/11.0.2 
# !pip install --upgrade jax jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html 
# !pip install tensorflow-io oryx elegy

import os
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
assert jax.default_backend() == 'gpu'
import gcr_utils
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import elegy # pip install elegy. 
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# Hyperparameters
num_results = 500000 # 10k takes 11min. About 1/5 of these accepted.
num_burnin_steps = 1000 #500
num_adaptation_steps = np.floor(.8*num_burnin_steps) #Somewhat smaller than number of burnin
target_accept_prob = 0.3
step_size = 1e-3 # 1e-5 has 0.95 acc rate and moves. 1e-4 0.0 acc.
num_leapfrog_steps = 100
max_tree_depth = 10 # Default=10. Smaller results in shorter steps. Larger takes memory.
model_path = 'model_2_256_selu_l21e-6' #'model_2_256_selu_l21e-6_do' # 'model_2_256_selu_l21e-6' #'model_2_256_selu'

seed = int(os.environ['SLURM_ARRAY_TASK_ID']) + int(os.environ['SLURM_ARRAY_JOB_ID'])
results_dir = f'../../results/job_{os.environ["SLURM_ARRAY_JOB_ID"]}/'

# Load observation data and define logprob.
interval = '20110520-20110610'
data_path = f'../data/oct2022/AMS02_H-PRL2018/AMS02_H-PRL2018_{interval}.dat'
alpha, cmf = gcr_utils.get_alpha_cmf('../data/oct2022/AMS02_H-PRL2018_heliosphere.dat', interval)
target_log_prob = gcr_utils.define_log_prob(model_path, data_path, alpha, cmf)

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
#state = jnp.array(gcr_utils.minmax_scale_input(np.array([90, .5, 1.7, 1.4, 1.1 ]))) # This is ~MLE
#state = 0.5 * jnp.ones((5,), dtype='float32')
key = random.PRNGKey(seed)
unnormalized_samples, pkr = run_chain(key, state)
print('Finished in %d minutes.' % int((time.time() - start_time)//60))

# De-normalize samples and remove duplicates.
unnormalized_samples = unnormalized_samples.to_py()
all_samples = gcr_utils.deminmax_scale_input(unnormalized_samples)
samples, pkr_select = gcr_utils.remove_consecutive_duplicates(all_samples, pkr, atol=0.0)
all_log_accept_ratio, all_log_probs, all_step_sizes = pkr
log_accept_ratio, log_probs, step_sizes  = pkr_select
#samples, (log_accept_ratio, log_probs, step_sizes) = gcr_utils.remove_consecutive_duplicates(all_samples, (log_accept_ratio, log_probs, step_sizes))
print(f'Acceptance rate: {len(samples)/len(all_samples)}. Decrease step_size to increase rate.')

# Save results: samples and plots
np.savetxt(fname=f'{results_dir}/samples_{seed}.csv', X=samples, delimiter=',')
np.savetxt(fname=f'{results_dir}/logacceptratio_{seed}.csv', X=log_accept_ratio, delimiter=',')
np.savetxt(fname=f'{results_dir}/log_probs_{seed}.csv', X=log_probs, delimiter=',')
np.savetxt(fname=f'{results_dir}/stepsizes_{seed}.csv', X=step_sizes, delimiter=',')

