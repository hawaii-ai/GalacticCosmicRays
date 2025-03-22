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

HMC python file was originally inspired by this example:
https://colab.research.google.com/github/tensorflow/probability/blob/master/spinoffs/oryx/examples/notebooks/probabilistic_programming.ipynb#scrollTo=nmjmxzGhN855

### Environment set-up
Creating the environment is tricky, because it cannot be done directly from environment.yml. This is partly because of the need for ```"jax[cuda12_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```, which cannot be installed from conda.
Requirements.txt has been edited to not have the jax line, and so after setting up some basic conda packages can be installed directly (with the need for installing jax separately). Installing in a different way may fail because jax, tensorflow, tensorflow-io, keras, keras-core, and tensorflow-probability all need to be compatible with each other.

Instead, do the following:
```
conda install python=3.9 numpy scipy pandas matplotlib ipykernel
conda install -c anaconda cudatoolkit
pip install --upgrade "jax[cuda12_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
