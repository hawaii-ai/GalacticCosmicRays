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
import argparse

import keras_core as keras

import tensorflow_io as tfio
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--polarity', type=str, help='The polarity of the data to train on. Either pos or neg.')
    parser.add_argument('--train_size_fraction', type=float, help='The fraction of the training data to use. Must be between 0 and 1.')
    parser.add_argument('--bootstrap', action='store_true', help='Whether to use bootstrap sampling. If passed, bootstrapping, else no bootstrapping.')
    parser.add_argument('--model_version', type=str, default='v0', help='The version of the model to use. Default is v0, so no shuffling of train set (take first x% of the train set)')
    args = parser.parse_args()

    data_path = '/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/data/2023_07_01'
    data_file = f'{data_path}/{args.polarity}/model_collection_1AU_90deg_0deg_fixed_training.h5'
    # 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
    # features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
    with h5py.File(data_file, 'r') as h5:
        num_samples, num_inputs,  = h5['X_minmax'].shape
        _, num_flux,  = h5['Y_log_scaled'].shape
    x = tfio.IODataset.from_hdf5(data_file, dataset='/X_minmax')
    y = tfio.IODataset.from_hdf5(data_file, dataset='/Y_log_scaled')

    # Split
    full = Dataset.zip((x, y))
    train_cardinality = np.floor(num_samples * .9)
    full_train = full.take(train_cardinality) # Keep train set we sample from consistent as 90% of the data
    test = full.skip(train_cardinality) # Keep test set consistent as 10% of the data

    # Get number of training samples (from the dataset)
    train_size = int(np.floor(train_cardinality * args.train_size_fraction))
    print(f'Train size: {train_size} = {args.train_size_fraction} * {train_cardinality}')

    # Choose seed based on model version
    model_seeds = {
        'v1': 42,
        'v1.1': 42, # to double check training a new model on same data is what we expect
        'v2': 87,
        'v3': 5,
        'v4': 98,
    }
    seed = model_seeds.get(args.model_version, None)

    if args.bootstrap:
        print("Using bootstrap sampling (with replacement)")

        # Reproducible bootstrap indices
        rng = np.random.default_rng(seed)
        sampled_indices = rng.integers(low=0, high=train_cardinality, size=train_size)

        # Load dataset into memory
        train_list = list(full_train.as_numpy_iterator())

        # Sample with replacement
        bootstrapped_data = [train_list[i] for i in sampled_indices]

        # Separate into inputs and outputs
        x_bootstrap, y_bootstrap = zip(*bootstrapped_data)

        # Convert back to tf.data.Dataset
        train = Dataset.from_tensor_slices((list(x_bootstrap), list(y_bootstrap)))

    else:
        print("Using traditional sampling (without replacement)")

        # Shuffle deterministically
        if args.model_version in model_seeds:
            train_shuffled = full_train.shuffle(
                buffer_size=train_cardinality, seed=seed, reshuffle_each_iteration=False
            )
        else:
            train_shuffled = full_train

        # Take subset without replacement
        train = train_shuffled.take(train_size)

    # Set batch_size to 128 unless the train size is smaller than 128, then set it to the train size.
    if train_size < 128:
        batch_size = train_size
    else:
        batch_size = 128
    print(f'Setting batch size: {batch_size}')

    train = train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    test = test.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    # Some calcs
    steps_per_epoch = int(train_size / batch_size )
    validation_steps = int(num_samples * .1 / batch_size)
    print(f'Steps per epoch: {steps_per_epoch}, validation steps: {validation_steps}')

    # Define model. 
    l2 = keras.regularizers.L2(l2=1e-6)
    model = keras.Sequential(layers=[
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(256, activation='selu', kernel_regularizer=l2),
        keras.layers.Dense(256, activation='selu', kernel_regularizer=l2),
        keras.layers.Dense(32, activation='linear', kernel_regularizer=l2),
    ])

    # Create save and log directories
    if args.bootstrap:
        save_dir = '../../models/model_size_investigation_bootstrap'
    else:
        save_dir = '../../models/model_size_investigation'

    model_path = f'{save_dir}/model_{args.model_version}_train_size_{args.train_size_fraction}_{args.polarity}.keras'  # Must end with keras.
    log_dir = f'../../../tensorboard_logs/fit/model_{args.model_version}_bootstrap_{args.bootstrap}_train_size_{args.train_size_fraction}_{args.polarity}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    print("\nTensorboard log dir: ", log_dir)
    if not os.path.exists(save_dir):
        print(f'Creating directory: {save_dir}')
        os.makedirs(save_dir)

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Compile and fit the model
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mae', optimizer=optimizer)

    history = model.fit(
        train,
        epochs=1_000,
        steps_per_epoch=steps_per_epoch,
        validation_data=test,
        shuffle=False,
        verbose=2,
        callbacks=callbacks,
    )

    # Save the performance on the train and test set
    train_mae = model.evaluate(train)
    test_mae = model.evaluate(test)

    save_file = f'{save_dir}/{args.model_version}_{args.polarity}.csv'
    with open(save_file, 'a') as f:
        f.write(f'{args.train_size_fraction},{train_mae},{test_mae}\n')
        print(f'Saved to {save_file}\n')

if __name__ == '__main__':
    main()
