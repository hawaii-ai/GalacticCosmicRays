# GalacticCosmicRays
This repository contains code and data for the paper "Neural Surrogate HMC: On Using Neural Likelihoods for Hamiltonian Monte Carlo in Simulation-Based Inference" by Wolniewicz et al. (https://arxiv.org/abs/2407.20432).

Corresponding data for the project and paper can be found on Zenodo in community https://zenodo.org/communities/neural_surrogate_hmc/about. The DOIs for the four parts of the dataset are:
    Part 1/4: DOI 10.5281/zenodo.17729139
    Part 2/4: DOI 10.5281/zenodo.17764339
    Part 3/4: DOI 10.5281/zenodo.17764343
    Part 4/4: DOI 10.5281/zenodo.17764345

The code in this repository is used to generate samples from the posterior of galactic cosmic ray transport parameters using Hamiltonian Monte Carlo (HMC) with a neural network surrogate likelihood. The neural network is trained to emulate the output of a physics-based cosmic ray transport simulation, allowing for efficient sampling of the posterior distribution of model parameters given observational data.

## Project structure
### /data
Datasets used in this project.
- BR2461.dat initial dataset.
- new_models.dat Dataset received Feb 2022.The order of the parameters in each line is the usual one: alpha, cmf, cpa, pwr1par, pwr2par, pwr1perr, pwr2perr. For alpha and cmf, I added new values (alpha = 0, 10, 85; cmf = 2.5, 3.5, 9.5), which reflect more extreme conditions in the heliosphere: lower values of alpha and cmf correspond to a more quiet heliosphere, which result in less modulation of cosmic rays (less difference in the rigidity spectrum with respect to the local interstellar spectrum, the boundary condition at the external edge of the heliosphere), while higher values of alpha and cmf correspond to a more turbulent heliosphere, which result in more modulation. 
- 2023/ PAMELA and AMS-02 observation files used in the paper.
- 2024/ Yearly data files.
- oct2022/ Original data files received in Oct 2022.

### /notebooks
iPython notebooks used for quick analysis, plotting, training, and generating new data.
- /plot - Notebooks for plotting results.
- /dev - Notebooks for development and testing.
- generate_ppc_dat_files.ipynb - Notebook to generate posterior predictive test data files from held-out simulation data.

### /scripts
Python scripts for data processing and model training.
- /nn - Neural network related scripts.
- /analysis - Analysis and plotting scripts.

### /models
Trained NN models. 
- The NN models used for generating the results in the paper is 'model_size_investigation_optuna_10082025_trial5_full'

HMC python file was originally inspired by this example:
https://colab.research.google.com/github/tensorflow/probability/blob/master/spinoffs/oryx/examples/notebooks/probabilistic_programming.ipynb#scrollTo=nmjmxzGhN855

## Environment set-up
If you only want to work with the samples and NN files, the following should be sufficient:

```
export PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
pip install -U 'tensorflow[and-cuda]==2.15' tensorflow-io==0.36.0 keras==2.15.0 keras-core==0.1.7 tensorflow-probability==0.23
```

Otherwise, creating the environment is tricky, because it requires certain package dependencies to let JAX and TensorFlow access GPU. 
The environment.yml should work, but just in case the below instructions should cover any problems.

Note: Tensorflow probability 0.23 is compatible with Tensorflow 2.15 & jax 0.4.20. Tensorflow 2.15 is compatible with 12.2 and cuDNN 8.9. Tensorflow-io 0.36.0 is compatible with Tensorflow 2.15. Jax 0.4.20 requires scipy 1.12.* (scipy.linalg.tril removed in scipy 1.13.0), and NumPy < 1.27.

```
conda install python=3.9 -y
conda install -c nvidia/label/cuda-12.2.0 -c conda-forge cuda-toolkit=12.2 cudnn=8.9 cuda-nvcc -y
conda install numpy scipy pandas matplotlib ipykernel h5py optuna statsmodels pillow -y
pip install --upgrade "jax[cuda12_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "tensorflow[and-cuda]==2.15.*"
pip install "tensorflow-probability==0.23.*"
pip install "tensorflow-io==0.36.*"
pip install keras-core

pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install rtdl_num_embeddings

pip install --upgrade "scipy==1.12.*" "numpy==1.26.*"
```

Then, in $CONDA_PREFIX/etc/conda/activate.d/activate.sh, add "export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/bin", and in deactivate.d/deactivate.sh add "unset XLA_FLAGS". These are needed to set up jax with the gpu configuration. NOTE: this will fail if $CUDA_DIR or $XLA_FLAGS is set in your .bash_profile or .bashrc.

Note: torch==2.3.1 and rtdl_num_embeddings are only necessary to do the tabular embedding method of Gorishniy et al. 2022 (https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md). However, as it is coded in PyTorch, I have implemented a new version in Tensorflow in scripts/rtdl_num_embeddings_tf.py, which does not require these packages.
