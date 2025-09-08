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

def load_dataset(polarity, data_version, train_size_fraction, bootstrap):
    # 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
    # features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
    data_path = '/home/linneamw/sadow_koastore/personal/linneamw/research/gcr/data/shuffled_may2025'
    train_file = f'{data_path}/{polarity}/train.h5'
    test_file = f'{data_path}/{polarity}/test.h5'

    # Load train data
    with h5py.File(train_file, 'r') as h5:
        num_train_samples, num_inputs,  = h5['X_minmax'].shape
        _, num_flux,  = h5['Y_log_scaled'].shape
    x_train = tfio.IODataset.from_hdf5(train_file, dataset='/X_minmax')
    y_train = tfio.IODataset.from_hdf5(train_file, dataset='/Y_log_scaled')
    full_train = Dataset.zip((x_train, y_train))

    # Load test data
    with h5py.File(test_file, 'r') as h5:
        num_test_samples, num_inputs,  = h5['X_minmax'].shape
        _, num_flux,  = h5['Y_log_scaled'].shape
    x_test = tfio.IODataset.from_hdf5(test_file, dataset='/X_minmax')
    y_test = tfio.IODataset.from_hdf5(test_file, dataset='/Y_log_scaled')
    test = Dataset.zip((x_test, y_test))

    # Get number of training samples (from the dataset)
    train_size = int(np.floor(num_train_samples * train_size_fraction))

    # Choose seed based on model version
    data_seeds = {
        'd1': 42,
        'd2': 87,
        'd3': 5,
        'd4': 98,
        'd5': 123,
    }
    data_seed = data_seeds.get(data_version, None)

    if bootstrap == 'b1':
        # Reproducible bootstrap indices
        rng = np.random.default_rng(data_seed)
        sampled_indices = rng.integers(low=0, high=num_train_samples, size=train_size)

        # Load dataset into memory
        train_list = list(full_train.as_numpy_iterator())

        # Sample with replacement
        bootstrapped_data = [train_list[i] for i in sampled_indices]

        # Separate into inputs and outputs
        x_bootstrap, y_bootstrap = zip(*bootstrapped_data)

        # Convert back to tf.data.Dataset
        train = Dataset.from_tensor_slices((list(x_bootstrap), list(y_bootstrap)))

    else:
        # Shuffle deterministically
        if data_version in data_seeds:
            train_shuffled = full_train.shuffle(
                buffer_size=num_train_samples, seed=data_seed, reshuffle_each_iteration=False
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

    train = train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    test = test.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    return train, test, train_size, num_test_samples, batch_size

def build_model(n_layers, n_units):
    model = keras.Sequential()
    model.add(keras.Input(shape=(8,)))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(n_units, activation="selu"))
    model.add(keras.layers.Dense(32, activation="linear"))

    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--polarity', type=str, help='The polarity of the data to train on. Either pos or neg.')
    parser.add_argument('--train_size_fraction', type=float, help='The fraction of the training data to use. Must be between 0 and 1.')
    parser.add_argument('--bootstrap', type=str, default='b0', help='Whether to use bootstrap sampling. If b0, then no bootstrap sampling. If b1, then bootstrap sampling.')
    parser.add_argument('--model_version', type=str, default='init0', help='The version of the model to use. Normally init0, but can be init1, init2, etc. to test different initializations.')
    parser.add_argument('--data_version', type=str, default='d1', help='The version of the data seed to use. Default is d1, so just seed = 42.')
    parser.add_argument('--regularizer', type=float, default='1e-6', help='The ls regularization to apply to each layer. Default is 1e-6.')
    parser.add_argument('--save_dir', type=str, default='../../models/model_size_investigation', help='The directory to save the model and training history to.')
    args = parser.parse_args()

    print(f'Polarity: {args.polarity}, Train size fraction: {args.train_size_fraction}, Bootstrap: {args.bootstrap}, Model version: {args.model_version}, Data version: {args.data_version}')

    train, test, train_size, num_test_samples, batch_size = load_dataset(
        polarity=args.polarity,
        bootstrap=args.bootstrap,
        data_version=args.data_version, 
        train_size_fraction=args.train_size_fraction
    )

    # Some calcs
    steps_per_epoch = train_size // batch_size
    validation_steps = num_test_samples // batch_size

    # Create save and log directories
    save_name = f'data_{args.data_version}_bootstrap_{args.bootstrap}_model_{args.model_version}_train_size_{args.train_size_fraction}_{args.polarity}'

    model_path = f'{args.save_dir}/{save_name}.keras'  # Must end with keras.
    log_dir = f'../../../tensorboard_logs/best_optuna/{save_name}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    print("\nTensorboard log dir: ", log_dir)

    if not os.path.exists(args.save_dir):
        print(f'Creating directory: {args.save_dir}')
        os.makedirs(args.save_dir)

    if not os.path.exists(log_dir):
        print(f'Creating directory: {log_dir}')
        os.makedirs(log_dir)

    # Model hyperparameters
    learning_rate = 1.918416336823577e-05
    weight_decay = 3.251785236175247e-06
    n_layers = 10
    n_units = 1024

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Define model and compile
    model = build_model(n_layers=n_layers, n_units=n_units)
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    # Train
    history = model.fit(
        train,
        epochs=1_000,
        validation_data=test,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        shuffle=False,
        verbose=2,
        callbacks=callbacks
    )

    # Evaluate on train
    train_results = model.evaluate(train, verbose=0)
    train_mae = train_results[0]  # This is the loss (MAE)
    train_mse = train_results[1]  # This is the metric (MSE)

    # Evaluate on test
    test_results = model.evaluate(test, verbose=0)
    test_mae = test_results[0]
    test_mse = test_results[1]

    print(f"Train MAE: {train_mae}, Train MSE: {train_mse}")
    print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")

    # Save the performance on the train and test set
    save_file = f'{args.save_dir}/data_{args.data_version}_bootstrap_{args.bootstrap}_model_{args.model_version}_{args.polarity}_mae_mse.csv'
    with open(save_file, 'a') as f:
        f.write(f'{args.train_size_fraction},{train_mae},{train_mse},{test_mae},{test_mse}\n')
        print(f'Saved to {save_file}\n')


if __name__ == '__main__':
    main()
