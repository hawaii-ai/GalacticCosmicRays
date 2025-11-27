import numpy as np
import h5py
import keras_core as keras
import tensorflow_io as tfio
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

# Set parameters
polarity = 'neg'
train_size_fraction = 1.0 # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
model_version = 'init1' # 'init1', 'init2'
data_version = 'd1' # 'd1', 'd2'
bootstrap = 'b0' # 'b0', 'b1'
model_dir = 'your_model_path_here' # Replace with your model directory
data_dir = 'your_data_path_here' # Replace with your data directory 

##################################################
# Load train and test data
train_file = f'{data_dir}/shuffled_train_set.h5'
test_file = f'{data_dir}/shuffled_test_set.h5'

# 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
# features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
with h5py.File(train_file, 'r') as h5:
    print('Train set keys:', h5.keys())
    num_train_samples, num_train_inputs,  = h5['X_minmax'].shape
    _, num_flux,  = h5['Y_log_scaled'].shape
    quality  = h5['quality'][:]

train_x = tfio.IODataset.from_hdf5(train_file, dataset='/X_minmax')
train_y = tfio.IODataset.from_hdf5(train_file, dataset='/Y_log_scaled')
train_quality = tfio.IODataset.from_hdf5(train_file, dataset='/quality')
full_train = Dataset.zip((train_x, train_y))

# 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
# features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
with h5py.File(test_file, 'r') as h5:
    print('Train set keys:', h5.keys())
    num_test_samples, num_test_inputs,  = h5['X_minmax'].shape
    _, num_flux,  = h5['Y_log_scaled'].shape
    quality  = h5['quality'][:]

test_x = tfio.IODataset.from_hdf5(train_file, dataset='/X_minmax')
test_y = tfio.IODataset.from_hdf5(train_file, dataset='/Y_log_scaled')
test_quality = tfio.IODataset.from_hdf5(train_file, dataset='/quality')
test = Dataset.zip((test_x, test_y))

# Split according to d1 and d2 and smaller train size fractions
train_size = int(np.floor(num_train_samples * train_size_fraction))
print(f'Train size: {train_size} = {train_size_fraction} * {num_train_samples}')

# Choose seed based on model version
data_seeds = {
    'd1': 42,
    'd2': 87,
}
data_seed = data_seeds.get(data_version, None)

if bootstrap == 'b1':
    print(f"Using bootstrap sampling (with replacement) for data version {data_version} and seed {data_seed}")

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
    print("Using traditional sampling (without replacement)")

    # Shuffle deterministically
    if data_version in data_seeds:
        train_shuffled = full_train.shuffle(
            buffer_size=num_train_samples, seed=data_seed, reshuffle_each_iteration=False
        )
    else:
        train_shuffled = full_train

    # Take subset without replacement
    train = train_shuffled.take(train_size)

# Batch test/validation set
batch_size = 128
train = train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
train_steps = int(train_size / batch_size)
test = test.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
validation_steps = int(num_test_samples / batch_size)

print(f'Train steps: {train_steps}')
print(f'Validation steps: {validation_steps}')

##################################################
# Load model
save_name = f'data_{data_version}_bootstrap_{bootstrap}_model_{model_version}_train_size_{train_size_fraction}_{polarity}'
model_path = f'{model_dir}/{save_name}.keras'  # Must end with keras.
print(f"Model: {model_path}. Predicting...")
model = keras.models.load_model(model_path)

# Predict on the test sets
test_pred = model.predict(test, steps=validation_steps+1)

# Calculate MAE on test set
test_mae = model.evaluate(test, steps=validation_steps+1)
print(f'Overall test MAE: {test_mae}')

# Extract all validation set features and labels
test_x = []
test_y = []

for batch in test:
    x, y = batch
    test_x.append(x.numpy())  # Convert tensors to numpy arrays
    test_y.append(y.numpy())

# Convert lists to single numpy arrays
test_x = np.concatenate(test_x, axis=0)  # Shape: (num_samples, 8)
test_y = np.concatenate(test_y, axis=0)  # Shape: (num_samples, 32)

# Calculate relative MAE for each flux
relative_mae = np.mean(np.abs((test_y - test_pred) / test_y), axis=0)  # Shape: (32,)