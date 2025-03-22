"""
Profile NN inference for one model
"""

import os
from collections import defaultdict
import numpy as np
import h5py
import matplotlib.pyplot as plt

import keras_core as keras

import tensorflow_io as tfio
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

import sys
sys.path.append('../')
import preprocess.preprocess
import utils

# Create dataset object using IODataset
polarity = 'pos'
path = '/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/data/2023_07_01'
f_highquality = f'{path}/{polarity}/model_collection_1AU_90deg_0deg_fixed_training.h5'
f_lowquality = f'{path}/{polarity}/model_collection_1AU_90deg_0deg_fixed_training_lowquality.h5'
f_full = f'{path}/{polarity}/model_collection_1AU_90deg_0deg_fixed_training_full.h5'

assert os.path.exists(f_highquality), f'File {f_highquality} does not exist'
assert os.path.exists(f_lowquality), f'File {f_lowquality} does not exist'
assert os.path.exists(f_full), f'File {f_full} does not exist'

# Load highquality data used for training/testing and split into train/test splits

# 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
# features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
# N = 1000  # Just get a subset for testing
with h5py.File(f_highquality, 'r') as h5:
    print(h5.keys())
    
    num_samples, num_inputs,  = h5['X_minmax'].shape
    _, num_flux,  = h5['Y_log_scaled'].shape
    imodel  = h5['imodel'][:]
    ipar  = h5['ipar'][:]
    vseq  = h5['vseq'][:]
    quality  = h5['quality'][:]
    # num_samples = N
    # x = h5['/X_minmax'][:N, ...]
    # y = h5['Y_log_scaled'][:N,...]
x = tfio.IODataset.from_hdf5(f_highquality, dataset='/X_minmax')
y = tfio.IODataset.from_hdf5(f_highquality, dataset='/Y_log_scaled')

full = Dataset.zip((x, y))

BATCH_SIZE = 1
train = full.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

single_sample = train.take(1)
print(single_sample)

# Load model
model_version = 'v3.0'
model_path = f'../../models/model_{model_version}_{polarity}.keras'  # Must end with keras.
model = keras.models.load_model(model_path)

print(f"Model: {model_path}. Predicting...")

import time
execution_time = []
num_reps = 100

for i in range(num_reps):
    start_time = time.time()
    model.predict(single_sample, steps=1)
    end_time = time.time()

    execution_time.append(end_time - start_time)

print("Minimum inference time: ", min(execution_time))
print("Maximum inference time: ", max(execution_time))
print("Average inference time: ", np.mean(execution_time))

exit()