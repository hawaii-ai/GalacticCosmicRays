"""
Train both the positive and negative NN with varying amounts of the training data to determine the effect of training data size on the model performance.
"""

# Imports
import os
from collections import defaultdict
import numpy as np
import h5py
import matplotlib.pyplot as plt
import datetime

import keras_core as keras

import tensorflow_io as tfio
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

def main():
    polarity = os.getenv('POLARITY')
    train_size_percent = os.getenv('TRAIN_SIZE_PERCENT')

    data_path = '/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/data/2023_07_01'
    data_file = f'{data_path}/{polarity}/model_collection_1AU_90deg_0deg_fixed_training.h5'
    # 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
    # features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
    with h5py.File(data_file, 'r') as h5:
        num_samples, num_inputs,  = h5['X_minmax'].shape
        _, num_flux,  = h5['Y_log_scaled'].shape
    x = tfio.IODataset.from_hdf5(data_file, dataset='/X_minmax')
    y = tfio.IODataset.from_hdf5(data_file, dataset='/Y_log_scaled')

    # Split
    full = Dataset.zip((x, y))
    train = full.take(np.floor(num_samples *.9))#.repeat()
    test = full.skip(np.floor(num_samples *.9))#.repeat()

    # Reduce train size based on the train_size_percent
    train_size = len(train)
    train = train.take(np.floor(train_size * train_size_percent))

    # Batch
    batch_size = 128
    train = train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    test = test.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    # Some calcs
    steps_per_epoch = int(num_samples * .9 / batch_size )
    validation_steps = int(num_samples * .1 / batch_size)
    print(f'Steps per epoch: {steps_per_epoch}')

    # Define model. 
    l2 = keras.regularizers.L2(l2=1e-6)
    model = keras.Sequential(layers=[
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(256, activation='selu', kernel_regularizer=l2),
        keras.layers.Dense(256, activation='selu', kernel_regularizer=l2),
        keras.layers.Dense(32, activation='linear', kernel_regularizer=l2),
    ])

    # Create save and log directories
    save_dir = '../models/model_size_investigation'
    model_path = f'{save_dir}/model_train_size_{train_size_percent}_{polarity}.keras'  # Must end with keras.
    log_dir = f'../../tensorboard_logs/fit/model_train_size_{train_size_percent}_{polarity}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    print("\nTensorboard log dir: ", log_dir)
    if not os.path.exists(save_dir):
        print(f'Creating directory: {save_dir}')
        os.makedirs(save_dir)

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
        keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Compile and fit the model
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mae', optimizer=optimizer)

    history = model.fit(
        train,
        epochs=100,
        validation_data=test,
        shuffle=False,
        verbose=2,
        callbacks=callbacks,
    )

    # Save the performance on the train and test set
    train_mae = model.evaluate(train)
    test_mae = model.evaluate(test)

    save_file = f'{save_dir}/{polarity}.csv'
    with open(save_file, 'a') as f:
        f.write(f'{train_size_percent},{train_mae},{test_mae}\n')

    return

if main == '__main__':
    main()