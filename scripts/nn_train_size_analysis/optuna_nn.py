import os
import datetime
import h5py
import numpy as np
import optuna
# import keras_core as keras
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from optuna.storages import JournalStorage, JournalFileStorage

from rtdl_num_embeddings_tf import (
    LinearEmbeddings,
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEmbeddings,
    compute_bins,
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

def build_model(input_dim, n_layers, units, embedding_method, embed_dim=12, n_bins=48):
    print(f"Building model with embedding {embedding_method}, {n_layers} layers, and {units} units per layer")

    model = keras.Sequential([keras.Input(shape=(input_dim,), dtype="float32")])

    # Tabular embedding layer
    if embedding_method == "linear":
        model.add(LinearEmbeddings(input_dim, embed_dim))
        model.add(keras.layers.Flatten())
    elif embedding_method == "linear_relu":
        model.add(LinearReLUEmbeddings(input_dim, embed_dim))
        model.add(keras.layers.Flatten())
    elif embedding_method == "periodic":
        # Defaults: k=64, sigma=0.02, activation=True (you can change)
        model.add(PeriodicEmbeddings(input_dim, embed_dim))
        model.add(keras.layers.Flatten())
    # TODO: fix piecewise_linear embedding (currently returning NaN loss)
    # elif embedding_method in {"piecewise_linear", "piecewise_linear_relu"}:
    #     # Compute bins **once** outside the training loop; pass numpy or a dense tensor
    #     bins = compute_bins(x_train, n_bins)
    #     model.add(PiecewiseLinearEmbeddings(
    #         bins, embed_dim,
    #         activation=(embedding_method == "piecewise_linear_relu"),
    #         version="B"  # residual linear, as in your code
    #     ))
    #     model.add(keras.layers.Flatten())
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

    # Commented out due to issue with piecewise linear embeddings
    # # Get x_final_train from the zipped and shuffled and batched train
    # # Collect all batches into memory
    # x_batches = []
    # for x_batch, _ in train:   # iterate through the dataset
    #     x_batches.append(x_batch.numpy())  # convert to numpy

    # # Concatenate into a single array
    # x_final_train = np.concatenate(x_batches, axis=0)
    # x_final_train_tensor = tf.convert_to_tensor(x_final_train, dtype=tf.float32)

    # Get trial hyperparameters
    n_layers = trial.suggest_int("n_layers", 3, 10)
    units = trial.suggest_categorical("units", [512, 1024, 2048, 4096])
    embedding_method = trial.suggest_categorical("embedding_method", [
        "none",
        "linear_relu",
        "periodic",
        # "piecewise_linear_relu"
    ])

    # Define model
    model = build_model(num_inputs, n_layers, units, embedding_method)
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