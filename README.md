# GalacticCosmicRays
GCR forecasting project with Claudio Corti

/data
Datasets used in this project.
- BR2461.dat initial dataset.
- new_models.dat Dataset received Feb 2022.The order of the parameters in each line is the usual one: alpha, cmf, cpa, pwr1par, pwr2par, pwr1perr, pwr2perr. For alpha and cmf, I added new values (alpha = 0, 10, 85; cmf = 2.5, 3.5, 9.5), which reflect more extreme conditions in the heliosphere: lower values of alpha and cmf correspond to a more quiet heliosphere, which result in less modulation of cosmic rays (less difference in the rigidity spectrum with respect to the local interstellar spectrum, the boundary condition at the external edge of the heliosphere), while higher values of alpha and cmf correspond to a more turbulent heliosphere, which result in more modulation.

/notebooks
Analysis notebooks.

/scripts
Python scripts for data processing and model training.

/models
Trained models.

HMC python file was originally inspired by this example:
https://colab.research.google.com/github/tensorflow/probability/blob/master/spinoffs/oryx/examples/notebooks/probabilistic_programming.ipynb#scrollTo=nmjmxzGhN855

### Environment set-up
Creating the environment is tricky, because it requires certain package dependencies to let JAX and TensorFlow access GPU. The environment.yml should work, but just in case the below instructions should cover any problems.

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
