
import h5py
import numpy as np

# path = '/mnt/lts/nfs_fs02/sadow_lab/shared/gcr/data/2023_07_01/'
# original_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed.h5'
# fixed_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed.h5'
# train_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed_processed.h5'

# This needs to be hardcoded for transforms. Used in both parities.
Y_LOG_MAX = 8.815241

def prepreprocess(infile, outfile):
    # Takes one minute. Reduces size by 20%?
    # https://docs.h5py.org/en/stable/high/group.html

    with h5py.File(outfile,'w') as f_dest:
        with h5py.File(infile,'r') as f_src:
                #f_src.copy(f_src["/"],f_dest["/"], name='.')
                f_src.copy(f_src["/info"], f_dest)
                # Copy model data.
                fmodel = f_dest.create_group('model')
                dnames = ['alpha', 'cmf', 'cpa', 'flux', 'imodel', 'ipar', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'quality', 'vseq', 'vspoles']
                for dname in dnames:
                    if dname == 'flux':
                        X = f_src['model/flux'][:]
                        X = X.T # Transpose
                        f_dest.create_dataset(name='model/flux', data=X)
                        #dset = f.create_dataset('flux_fixed', data=X)
                    else:
                        f_src.copy(f_src[f"/model/{dname}"], fmodel)


def preprocess_for_training(filename, select_quality=True, shuffle_flag=True):
    """
    Preprocess simulation data from Claudio so we can train NN on it.
    Transformations:
    - Select subset with high quality. Quality flag =0 means good.
    - Permute data for training. 
    """
    # Load and process data.
    with h5py.File(filename, 'r') as f:
        print([k for k in f])
        #print(f['info'])
        print([k for k in f['info']])
        #print(f['model'])
        print([k for k in f['model']])
        if select_quality:
            selected = f['model/quality'][:] == 0
        else: 
            selected = slice(None)  # Return everything.

        #cols = ['alpha', 'cmf', 'cpa', 'flux', 'imodel', 'ipar', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'quality', 'vseq', 'vspoles']
        # 8 input parameters for the NN: alpha, cmf, vspoles, cpa, pwr1par, pwr2par, pwr1perr, and pwr2perr.
        features = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
        metas = ['imodel', 'ipar', 'quality', 'vseq']
        
        # Select subset of variables to be used for training.
        data = f['model']
        dsets = [data[k][selected].reshape(-1,1) for k in features]
        X = np.hstack(dsets)  # Seconds
        Y = np.array(data['flux'])  # Seconds
        Y = Y[selected, :]
        metadata = {k:data[k][selected] for k in metas}

    num_samples = X.shape[0]
    num_flux = Y.shape[1]
    print(f'Found samples: {num_samples}')
    print(f'Num flux targets: {num_flux}')

    if shuffle_flag:
        # Shuffle data because simulations were not done in random order.
        np.random.seed(0)
        num_samples = Y.shape[0]
        shuffle_indices = np.random.permutation(num_samples)
        X = X[shuffle_indices, :]
        Y = Y[shuffle_indices]
        for k,v in metadata.items():
            metadata[k] = v[shuffle_indices]

    return X, Y, metadata


def make_preprocessed_file(infile, outfile):
    """
    Preprocess data, transform, then write to file.
    """
    X,Y,metadata = preprocess_for_training(filename=infile, select_quality=True, shuffle_flag=True)

    # Transform data and write to new file.
    with h5py.File(outfile, 'w') as dest:
        # Write metadata vars as own dsets.
        for k,v in metadata.items():
            dset = dest.create_dataset(k, data=v)
        
        dset = dest.create_dataset("X", data=X)
        X_MIN = np.min(X, axis=0) #e.g. np.array([20., 4.5, 50., 0.2, 0.2, 0.2, 0.2])
        X_MAX = np.max(X, axis=0) #e.g. np.array([75., 8.5, 250., 2., 2.3, 2., 2.3])
        X_RANGE = X_MAX - X_MIN
        # print(X_MIN)
        # print(X_MAX)
        # print(X_RANGE)
        dset = dest.create_dataset("X_minmax", data=(X-X_MIN)/X_RANGE)
        dset.attrs['X_MIN'] = X_MIN
        dset.attrs['X_MAX'] = X_MAX
        dset.attrs['X_RANGE'] = X_RANGE
        
        # We don't need to do logp1 since we have reasonable values. 
        assert np.min(Y, axis=0).min() > 1e-6, np.min(Y, axis=0)
        #Y_logp1 = np.log(Y+1.)
        #Y_logp1_MAX = np.max(Y_logp1) # Y_MAX = 8.268953
        # Y_logp1_scaled = Y_logp1 / Y_logp1_MAX # Min should already be 0, and we want to emphasize the larger vals.
        # dset = dest.create_dataset("Y", data=Y)
        # dset = dest.create_dataset("Y_logp1", data=Y_logp1)
        # dset = dest.create_dataset("Y_logp1_scaled", data=Y_logp1_scaled)
        # dset.attrs['Y_logp1_MAX'] = Y_logp1_MAX
        Y_log = np.log(Y)
        Y_log_scaled = Y_log / Y_LOG_MAX
        dset = dest.create_dataset("Y_log_scaled", data=Y_log_scaled)
        dset.attrs['Y_log_MAX'] = Y_LOG_MAX
    
