import os
import datetime
import h5py
import numpy as np
import optuna
import tensorflow as tf
import tensorflow_io as tfio
import keras_core as keras
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from optuna.storages import JournalStorage, JournalFileStorage

from rtdl_num_embeddings_keras import (
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
)

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
    print(f"Number of training samples: {train_size} out of {num_train_samples} total")
    print(f"Number of test samples: {num_test_samples}")

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

    return train, test, train_size, num_test_samples, batch_size, num_inputs

def build_model(input_dim, n_layers, units, embedding_method, embed_dim=12, n_bins=48, n_frequencies=8, value_range=None):
    print(f"Building model with embedding {embedding_method}, {n_layers} layers, {units} units per layer, embed_dim {embed_dim}, n_bins {n_bins}, n_frequencies {n_frequencies}, value_range {value_range}")

    model = keras.Sequential([keras.Input(shape=(input_dim,), dtype="float32")])

    # Tabular embedding layer
    if embedding_method == "periodic":
        # Defaults: k=64, sigma=0.02, activation=True (you can change)
        model.add(PeriodicEmbeddings(
            n_features=input_dim,
            n_frequencies=n_frequencies,
            learnable_frequencies=True,
            use_phase=True,
            learnable_phases=True,
        ))
        model.add(keras.layers.Flatten())
    elif embedding_method == "piecewise_encoding":
        if value_range is None:
            model.add(PiecewiseLinearEncoding(
                n_features=input_dim,
                n_bins=n_bins,
                use_adaptive_range=True,
                clip=True,
            ))
        else:
            model.add(PiecewiseLinearEncoding(
                n_features=input_dim,
                n_bins=n_bins,
                use_adaptive_range=False,
                value_range=value_range,
                clip=True,
            ))
    elif embedding_method == "piecewise_embedding":
        if value_range is None:
            model.add(PiecewiseLinearEmbeddings(
                n_features=input_dim,
                n_bins=n_bins,
                d_embedding=embed_dim,
                activation=True,          # ReLU(Linear(PLE))
                use_adaptive_range=True,
                clip=True,
            ))
        else:
            model.add(PiecewiseLinearEmbeddings(
                n_features=input_dim,
                n_bins=n_bins,
                d_embedding=embed_dim,
                activation=True,          # ReLU(Linear(PLE))
                use_adaptive_range=False,
                value_range=value_range,
                clip=True,
            ))
        model.add(keras.layers.Flatten())
    else:
        # No embedding, use raw inputs
        pass

    # If you’re using SELU, pair with lecun_normal + AlphaDropout (recommended for SELU)
    for _ in range(n_layers):
        model.add(keras.layers.Dense(units, activation="selu", kernel_initializer="lecun_normal"))
        # optional:
        # model.add(keras.layers.AlphaDropout(0.05))

    model.add(keras.layers.Dense(32, activation="linear"))
    return model

def objective(trial):
    print(f"Starting trial {trial.number} at {datetime.datetime.now()} -----------------------")

    # Fixed args – customize if needed
    args = {
        "polarity": "neg",
        "train_size_fraction": 1.0,
        "bootstrap": "b0",
        "data_version": "d1"
    }

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)
    print(f"Trial {trial.number}: Learning rate: {learning_rate}, Weight decay: {weight_decay}")

    train, test, train_size, num_test_samples, batch_size, num_inputs = load_dataset(
        args["polarity"], args["data_version"], args["train_size_fraction"], args["bootstrap"]
    )

    # For piecewise linear embeddings, we need to provide the value range (min, max) for each feature.
    # All input data is min-max scaled already
    n_features = 8
    mins = np.zeros(n_features, dtype="float32")
    maxs = np.ones(n_features, dtype="float32")
    value_range = (mins, maxs)

    # Get trial hyperparameters
    n_layers = trial.suggest_int("n_layers", 3, 8)
    units = trial.suggest_categorical("units", [512, 1024, 2048, 4096])
    embedding_method = trial.suggest_categorical("embedding_method", [
        "none",
        "periodic",
        "piecewise_encoding",
        "piecewise_embedding",
    ])
    n_frequencies = trial.suggest_int("n_frequencies", 4, 16)
    n_bins = trial.suggest_int("n_bins", 4, 16)
    embed_dim = trial.suggest_int("embed_dim", 4, 32)

    # Define model
    model = build_model(
        num_inputs, 
        n_layers, 
        units, 
        embedding_method,
        embed_dim=embed_dim, 
        n_frequencies=n_frequencies, 
        n_bins=n_bins, 
        value_range=value_range
    )
    print(model.summary())

    # Compile model
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    # Train
    history = model.fit(
        train,
        epochs=150,
        validation_data=test,
        shuffle=False,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        ]
    )

    # Evaluate on train
    train_results = model.evaluate(train, verbose=0)
    train_mae = train_results[0]  # This is the loss (MAE)
    train_mse = train_results[1]  # This is the metric (MSE)

    # Evaluate on test
    test_results = model.evaluate(test, verbose=0)
    test_mae = test_results[0]
    test_mse = test_results[1]

    print(f"Trial {trial.number}: Train MAE: {train_mae}, Train MSE: {train_mse}")
    print(f"Trial {trial.number}: Test MAE: {test_mae}, Test MSE: {test_mse}")

    return test_mae

if __name__ == "__main__":
    save_dir = "./../../../optuna"
    storage = JournalStorage(
        JournalFileStorage(
            f"{save_dir}/opt_journal.log"
        )
    )

    study = optuna.create_study(direction="minimize", storage=storage, load_if_exists=True,)
    study.optimize(objective, n_trials=500, n_jobs=1)

    print("Best trial:")
    print(study.best_trial)