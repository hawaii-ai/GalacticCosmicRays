
import h5py
import numpy as np

# path = '/mnt/lts/nfs_fs02/sadow_lab/shared/gcr/data/2023_07_01/'
# original_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed.h5'
# fixed_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed.h5'
# train_file = f'{path}/pos/model_collection_1AU_90deg_0deg_fixed_processed.h5'

# Hardcoded model choices.
INPUTS = ['alpha', 'cmf', 'cpa', 'pwr1par', 'pwr1perr', 'pwr2par', 'pwr2perr', 'vspoles']
# These are hardcoded for transforms. Used in both parities.
X_MIN = np.array([0.,  2.5, 100., 0.4, 0.4, 0.4, 0.4, 400.]) 
X_MAX = np.array([85., 9.5, 870., 1.7, 1.7, 2.3, 2.3, 700.])
X_RANGE = np.array([ 85., 7., 770., 1.3000001, 1.3000001, 1.9, 1.9, 300.])
# X_MIN,X_MAX,X_RANGE = get_minmax_params(get_attributes(infile))
# These are selected from above and hardcoded.
PARAMETERS = ['cpa', 'pwr1par', 'pwr2par', 'pwr1perr', 'pwr2perr'] 
PARAMETERS_MIN = np.array([100., 0.4, 0.4, 0.4, 0.4]) 
PARAMETERS_MAX = np.array([870., 1.7, 1.7, 2.3, 2.3]) 
# These parameter don't include (alpha, cmf, vspoles) which we specify separately.
PARAMETERS_SPECIFIED = ['alpha', 'cmf', 'vspoles']
PARAMETERS_SPECIFIED_MIN = np.array([0.,  2.5, 400.])
PARAMETERS_SPECIFIED_MAX = np.array([85., 9.5, 700.])
# Calculated from positive file, used for both.
Y_LOG_MAX = 8.815241

NN_SPLIT_SEED = 36 # Random seed for splitting data into train/test sets for reproducibiity. Buffer size must be the same as num_samples.

# Now there are only 32 rigidity values.
# filename = f'{path}/{polarity}/model_collection_1AU_90deg_0deg_fixed.h5'
# with h5py.File(filename,'r') as f:
#     RIGIDITY_VALS = f['/info/rigidity'][:]
RIGIDITY_VALS = np.array(
      [  0.2       ,   0.20217378,   0.20659248,   0.21340226,
         0.22525435,   0.24034894,   0.25924241,   0.28573246,
         0.31835225,   0.35855115,   0.41265203,   0.48007787,
         0.56459135,   0.67849465,   0.82423959,   1.0121744 ,
         1.27012627,   1.61114001,   2.06592477,   2.70698217,
         3.54695997,   4.80077869,   6.49781115,   8.98694648,
        12.70126265,  17.95071033,  25.92424111,  37.84646324,
        56.45913512,  85.14084907, 129.78860616, 200.])


def get_attributes(filename):
    """
    Get attributes from h5 file.
    """
    with h5py.File(filename,'r') as f:
        d = dict(f['/info'].attrs)
    return d


def get_minmax_params(d):
    """Compute min, max, range used for normalizing inputs"""
    X_MIN = np.array([d[k].min() for k in INPUTS])
    X_MAX = np.array([d[k].max() for k in INPUTS])
    X_RANGE = X_MAX - X_MIN
    return X_MIN, X_MAX, X_RANGE


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
        metas = ['imodel', 'ipar', 'quality', 'vseq']
        
        # Select subset of variables to be used for training.
        data = f['model']
        dsets = [data[k][selected].reshape(-1,1) for k in INPUTS]
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

        # These values can and should be extracted from attributes. Serves as sanity check.
        x_min,x_max,x_range = get_minmax_params(get_attributes(infile))
        assert (X_MIN == x_min).all()
        assert (X_MAX == x_max).all()
        assert (X_RANGE == x_range).all()

        # Comment out for select_quality=False, Y has nan values
        assert np.min(Y, axis=0).min() > 1e-6, np.min(Y, axis=0)

        # We don't need to do logp1 since we have reasonable values. 
        #Y_logp1 = np.log(Y+1.)
        #Y_logp1_MAX = np.max(Y_logp1) # Y_MAX = 8.268953
        # Y_logp1_scaled = Y_logp1 / Y_logp1_MAX # Min should already be 0, and we want to emphasize the larger vals.
        # dset = dest.create_dataset("Y", data=Y)
        # dset = dest.create_dataset("Y_logp1", data=Y_logp1)
        # dset = dest.create_dataset("Y_logp1_scaled", data=Y_logp1_scaled)
        # dset.attrs['Y_logp1_MAX'] = Y_logp1_MAX

        # Comment below out for select_quality=False, Y has nan values
        Y_log = np.log(Y)
        Y_log_scaled = Y_log / Y_LOG_MAX
        dset = dest.create_dataset("Y_log_scaled", data=Y_log_scaled)
        dset.attrs['Y_log_MAX'] = Y_LOG_MAX
    

def _get_transform_params(X):
    """
    Helper function for calculating min max.
    """
    assert len(PARAMETERS) != len(PARAMETERS_SPECIFIED)
    input_dim = X.ndim
    if (X.ndim == 1 and len(X) == len(INPUTS)) or (X.ndim == 2 and X.shape[1] == len(INPUTS)):
        # Full set of inputs. 
        MIN, MAX = X_MIN, X_MAX
    elif ((X.ndim == 1 and len(X) == len(PARAMETERS)) or (X.ndim == 2 and X.shape[1] == len(PARAMETERS))):
        # Assume specified parameters have already been specified separately.
        MIN, MAX = PARAMETERS_MIN, PARAMETERS_MAX
    elif ((X.ndim == 1 and len(X) == len(PARAMETERS_SPECIFIED)) or (X.ndim == 2 and X.shape[1] == len(PARAMETERS_SPECIFIED))):
        # Assume other parameters have already been specified separately.
        MIN, MAX = PARAMETERS_SPECIFIED_MIN, PARAMETERS_SPECIFIED_MAX
    else:
        raise Exception
    return (MIN, MAX)
    

def transform_input(X):
    '''
    Parameters from HMC are all in min-max scaled space.
    This function tries to smartly handle case where some of the inputs are specified separately.
    '''
    MIN, MAX = _get_transform_params(X)
    RANGE = MAX - MIN
    rval = (X - MIN) / RANGE
    return rval


def untransform_input(X):
    '''
    Parameters from HMC are all in min-max scaled space.
    This function tries to smartly handle case where some of the inputs are specified separately.
    '''
    MIN, MAX = _get_transform_params(X)
    RANGE = MAX - MIN
    rval = X * RANGE + MIN
    return rval

