import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
#assert jax.default_backend() == 'gpu'

import elegy # pip install elegy. 

import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# These parameter don't include, alpha and cmf, which have been specified.
PARAMETERS = ['cpa', 'pwr1par', 'pwr2par', 'pwr1perr', 'pwr2perr']
MIN_PARAMETERS = np.array([50., 0.2, 0.2, 0.2, 0.2]) 
MAX_PARAMETERS = np.array([250., 2., 2.3, 2., 2.3]) 
RANGE_PARAMETERS = MAX_PARAMETERS - MIN_PARAMETERS


# These are the 245 rigidity vals for the NN training data.
RIGIDITY_VALS = np.array([  1.0001013,   1.022055 ,   1.0444907,   1.0674188,   1.0908501,
         1.1147959,   1.1392673,   1.1642759,   1.1898335,   1.215952 ,
         1.242644 ,   1.2699219,   1.2977985,   1.326287 ,   1.355401 ,
         1.385154 ,   1.4155602,   1.4466338,   1.4783896,   1.5108424,
         1.5440075,   1.5779008,   1.6125381,   1.6479356,   1.6841102,
         1.7210789,   1.7588592,   1.7974687,   1.8369257,   1.8772489,
         1.9184573,   1.9605702,   2.0036075,   2.0475898,   2.0925374,
         2.1384716,   2.185414 ,   2.2333872,   2.2824132,   2.3325157,
         2.3837178,   2.436044 ,   2.4895186,   2.5441673,   2.6000154,
         2.6570895,   2.7154164,   2.775024 ,   2.8359396,   2.898193 ,
         2.9618123,   3.0268285,   3.0932717,   3.1611736,   3.230566 ,
         3.3014817,   3.373954 ,   3.4480174,   3.5237062,   3.6010568,
         3.6801052,   3.760889 ,   3.843446 ,   3.9278152,   4.0140367,
         4.1021504,   4.1921988,   4.2842236,   4.3782687,   4.474378 ,
         4.572597 ,   4.6729727,   4.775551 ,   4.8803816,   4.987513 ,
         5.0969963,   5.208883 ,   5.323225 ,   5.440078 ,   5.5594954,
         5.681535 ,   5.8062525,   5.933708 ,   6.063962 ,   6.197075 ,
         6.33311  ,   6.472131 ,   6.6142035,   6.7593946,   6.9077735,
         7.059409 ,   7.2143736,   7.37274  ,   7.534582 ,   7.6999774,
         7.869003 ,   8.041739 ,   8.218267 ,   8.39867  ,   8.583034 ,
         8.771443 ,   8.963989 ,   9.160762 ,   9.361855 ,   9.567361 ,
         9.777378 ,   9.992006 ,  10.211345 ,  10.435499 ,  10.664574 ,
        10.898677 ,  11.137918 ,  11.382412 ,  11.632273 ,  11.887618 ,
        12.148569 ,  12.415248 ,  12.68778  ,  12.966296 ,  13.250925 ,
        13.541802 ,  13.839065 ,  14.142853 ,  14.453309 ,  14.77058  ,
        15.094816 ,  15.426169 ,  15.764796 ,  16.110857 ,  16.464514 ,
        16.825933 ,  17.195288 ,  17.57275  ,  17.958496 ,  18.352713 ,
        18.75558  ,  19.167294 ,  19.588043 ,  20.01803  ,  20.457455 ,
        20.906527 ,  21.365456 ,  21.834457 ,  22.313757 ,  22.803576 ,
        23.304148 ,  23.815708 ,  24.338497 ,  24.872763 ,  25.418756 ,
        25.976736 ,  26.546963 ,  27.129707 ,  27.725245 ,  28.333855 ,
        28.955824 ,  29.591448 ,  30.241022 ,  30.904858 ,  31.583263 ,
        32.27656  ,  32.98508  ,  33.709152 ,  34.449116 ,  35.205326 ,
        35.978134 ,  36.767906 ,  37.575016 ,  38.39984  ,  39.242775 ,
        40.104214 ,  40.98456  ,  41.884228 ,  42.80365  ,  43.74325  ,
        44.70348  ,  45.684788 ,  46.687637 ,  47.712498 ,  48.759857 ,
        49.830208 ,  50.924053 ,  52.041912 ,  53.184307 ,  54.35178  ,
        55.54488  ,  56.76417  ,  58.01023  ,  59.283638 ,  60.585003 ,
        61.914932 ,  63.274055 ,  64.66301  ,  66.08246  ,  67.533066 ,
        69.01552  ,  70.53051  ,  72.07876  ,  73.66099  ,  75.277954 ,
        76.93042  ,  78.619156 ,  80.34496  ,  82.10865  ,  83.91106  ,
        85.75303  ,  87.63543  ,  89.55916  ,  91.52511  ,  93.53422  ,
        95.58743  ,  97.685715 ,  99.83006  , 102.02148  , 104.261    ,
       106.54968  , 108.8886   , 111.27887  , 113.7216   , 116.21796  ,
       118.76911  , 121.37627  , 124.04066  , 126.76353  , 129.54617  ,
       132.38991  , 135.29605  , 138.266    , 141.30115  , 144.40291  ,
       147.57277  , 150.8122   , 154.12276  , 157.50597  , 160.96346  ,
       164.49684  , 168.10779  , 171.798    , 175.56921  , 179.42322  ,
       183.36182  , 187.38689  , 191.5003   , 195.70401  , 200.       ])


def get_interval(filename, index: int) -> str:
    """Read filename and return interval at index (zero indexed, so first index is 0)."""
    assert index >= 0, index
    df = pd.read_csv(filename, sep=' ', skiprows=1, names=['interval', 'alpha', 'cmf', 'alpha_std', 'cmf_std'])
    if index < df.shape[0]:
        return df['interval'].values[index]
    else:
        raise ValueError(f'Index {index} exceeds zero-indexed list of intervals in {filename} of length {df.shape[0]}.')


def get_alpha_cmf(filename, interval: str = None):
    """ Load alpha and cmf from file supplied by Claudio.
        Use alpha and cmf mean over interval. 
        Ignores std for now.
    """
    if 'BR2461.dat' in filename:
        assert interval == None
        alpha, cmf = 69.19, 5.17 # These are fixed for the current experiment.

    df = pd.read_csv(filename, sep=' ', skiprows=1, names=['interval', 'alpha', 'cmf', 'alpha_std', 'cmf_std'])
    row = df.loc[df['interval'] == interval]
    alpha = row['alpha'].values[0]
    cmf = row['cmf'].values[0]
    return alpha, cmf
        
    
def load_data_ams(filename):
    """ Load AMS data from Claudio.
    Args:
        filename = Filename of observations.
                   Original dataset was '../data/BR2461.dat'
    """
    dataset_ams = np.loadtxt(filename) # Rigidity1, Rigidity2, Flux, Error
    r1, r2 = dataset_ams[:,0], dataset_ams[:,1]
    bins = np.concatenate([r1[:], r2[-1:]])
    observed = dataset_ams[:,2]   # Observed Flux
    uncertainty = dataset_ams[:,3]
    assert len(bins) == len(observed)+1
    return bins, observed, uncertainty


def load_preprocessed_data_ams(filename):
    """ Load AMS data along with hardcoded auxiliary vectors for ppmodel.
    """
    bins, observed, uncertainty = load_data_ams(filename)
    # iloc = np.searchsorted(RIGIDITY_VALS, bins)
    # xloc = np.sort(np.concatenate([RIGIDITY_VALS, bins]))
    # assert np.all(xloc[iloc] == bins)
    xloc = np.concatenate([RIGIDITY_VALS, bins])
    sorted_indices = np.argsort(xloc)
    xloc = xloc[sorted_indices] # xloc is sorted list of rigidity locations.
    iloc = np.where(sorted_indices>=len(RIGIDITY_VALS))[0]
    assert np.all(xloc[iloc] == bins)
    
    return xloc, iloc, observed, uncertainty


def remove_consecutive_duplicates(samples, aux=[], atol=0.0):
    ''' 
    Remove consecutive duplicate rows from array. This means sample from MCMC was rejected.
    Args:
        samples = 2d array, where each row is a sample
        aux = list of 
        atol = absolute tolerance for being equal
    Returns:
        rval = 2d array with consecutive duplicate rows removed
    '''
    consecutive_repeat_rows = np.all(np.isclose(samples[1:,:], samples[:-1,:], atol=atol, rtol=0.0), axis=1)
    
    if len(aux) == 0:
        return samples[1:,...][~consecutive_repeat_rows, :]
    
    aux_return = []
    for a in aux:
        a = a[1:,...][~consecutive_repeat_rows, ...]
        aux_return.append(a)
    return samples[1:,...][~consecutive_repeat_rows, :], aux_return


def define_nn_pred(model_path, normalize_input_flag=False, denormalize_output_flag=True, rebin_output_flag=False):
    # Load trained NN model that maps 7 parameters to predicted flux at rigidity vals range(245).
    model = elegy.load(model_path)
    model.run_eagerly = True # Settable attribute. Required to be true for ppmodel.

    def f(x):
        # Make NN predictions on xs.
        if normalize_input_flag:
            x = minmax_scale_input(x)
        yhat = model.predict(x)
        if denormalize_output_flag:
            yhat = denormalize_output(yhat)
        if rebin_output_flag:
            yhat = rebin_output(yhat)
        return yhat
    return f


def minmax_scale_input(X):
    '''Parameters from HMC are all in min-max scaled space.'''
    input_dim = X.ndim
    if (X.ndim == 1 and len(X) == 7) or (X.ndim == 2 and X.shape[1] == 7):
        MIN = np.concatenate([np.array([20., 4.5]), MIN_PARAMETERS])
        MAX = np.concatenate([np.array([75., 8.5]), MAX_PARAMETERS])
    elif (X.ndim == 1 and len(X) == 5) or (X.shape[1] == 5):
        # Assume alpha and cmf have already been specified separately.
        MIN = MIN_PARAMETERS
        MAX = MAX_PARAMETERS 
    else:
        raise Exception
    RANGE = MAX - MIN
    rval = (X - MIN) / RANGE
    return rval


def deminmax_scale_input(X):
    '''Parameters from HMC are all in min-max scaled space. Undo minmax scaling. '''
    input_dim = X.ndim
    if (X.ndim == 1 and len(X) == 7) or (X.ndim == 2 and X.shape[1] == 7):
        MIN = np.concatenate([np.array([20., 4.5]), MIN_PARAMETERS])
        MAX = np.concatenate([np.array([75., 8.5]), MAX_PARAMETERS])
    elif (X.ndim == 1 and len(X) == 5) or (X.shape[1] == 5):
        # Assume alpha and cmf have already been specified separately.
        MIN = MIN_PARAMETERS
        MAX = MAX_PARAMETERS
    else:
        raise Exception
    RANGE = MAX - MIN
    rval = X * RANGE + MIN
    return rval


def denormalize_output(yhat):
    # Normalization: xnorm = log(x+1) / 8.268953
    # Inversion: x = exp(xnorm*8.268953) - 1
    yhat = yhat * 8.268953 # Undo max scaling.
    yhat = jnp.exp(yhat) - 1. # Undo logp1 transform of target output.
    return yhat


def rebin_output(yhat, data_path):
    # Interpolate to get predicted flux at both lattice and bin points.
    xloc, iloc, observed, uncertainty = load_preprocessed_data_ams(data_path)
    yloc = jnp.interp(xloc, RIGIDITY_VALS, yhat) 
    # Integrate over bin regions, and compare to observed to get likelihood.
    nbins = len(iloc)-1
    rebinned = np.zeros(nbins)
    for i in range(nbins):
        # Integrate over bin by trapezoid method.
        istart, istop = iloc[i], iloc[i+1]
        area = jnp.trapz(y=yloc[istart:(istop+1)], x=xloc[istart:(istop+1)])
        length = (xloc[istop] - xloc[istart])
        rebinned[i] = area / length
    return rebinned


def calc_loglikelihood(yhat, data_path):
    # Interpolate to get predicted flux at both lattice and bin points.
    xloc, iloc, observed, uncertainty = load_preprocessed_data_ams(data_path)
    yloc = jnp.interp(xloc, RIGIDITY_VALS, yhat) 
    chi2 = 0.0
    nbins = len(iloc)-1
    for i in range(nbins):
        # Integrate over bin by trapezoid method.
        istart, istop = iloc[i], iloc[i+1]
        area = jnp.trapz(y=yloc[istart:(istop+1)], x=xloc[istart:(istop+1)])
        length = (xloc[istop] - xloc[istart])
        predicted = area / length
        # Use equation provided by Claudio for likelihood of bin.
        chi2 += ((predicted - observed[i])/uncertainty[i])**2
    return -chi2/2


def remove_outliers(samples, buffer=0):
    """Remove samples that exceed min, max range. 
    """
    assert samples.shape[1] == 5
    outlier = np.bitwise_or(samples < MIN_PARAMETERS - buffer, samples > MAX_PARAMETERS + buffer)
    outlier = np.any(outlier, axis=1)
    return samples[~outlier, :]


def define_log_prob(model_path, data_path, alpha, cmf):
    # Load trained NN model that maps 7 parameters to predicted flux at RIGIDITY_VALS.
    model = elegy.load(model_path)
    model.run_eagerly = True # Settable attribute. Required to be true for ppmodel.

    # Load observation data from Claudio
    xloc, iloc, observed, uncertainty = load_preprocessed_data_ams(data_path)
    alpha_norm = (alpha - 20.) / 55. # Min max scaling
    cmf_norm = (cmf - 4.5) / 4. # Min max scaling
    
    def nn_predict(x):
        yhat = model.predict(x)    
        yhat = denormalize_output(yhat) # Undo scaling and minmax.
        return yhat

    def target_log_prob(xs):
        # Include logprior in loglikelihood. This keeps HMC from going off into no-mans land.
        penalty = 1e9 
        nlogprior = 0.
        for i in range(5):
            nlogprior += penalty * jnp.abs((jnp.minimum(0., xs[i]))) # Penalty for being <0
            nlogprior += penalty * jnp.abs((jnp.maximum(1., xs[i]) - 1.))  # Penalty for being >1

        xs = jnp.concatenate([jnp.array([alpha_norm, cmf_norm]), xs]) # Create 7d input to NN.
        yhat = nn_predict(xs)
        
        # Interpolate to get predicted flux at both lattice and bin points.
        yloc = jnp.interp(xloc, RIGIDITY_VALS, yhat) 
        nbins = len(iloc)-1
        # Integrate over bin regions, and compare to observed to get likelihood.
        chi2 = 0.0
        for i in range(nbins):
            # Integrate over bin by trapezoid method.
            istart, istop = iloc[i], iloc[i+1]
            area = jnp.trapz(y=yloc[istart:(istop+1)], x=xloc[istart:(istop+1)])
            length = (xloc[istop] - xloc[istart])
            predicted = area / length
            # Use equation provided by Claudio for likelihood of bin.
            chi2 += ((predicted - observed[i])/uncertainty[i])**2
            
        log_prob = -chi2/2.  - nlogprior
        return log_prob

    return target_log_prob



    
# # Deprecated
# def define_nn(alpha, cmf, model_path='model'):
#     # Load trained NN model that maps 7 parameters to predicted flux at rigidity vals range(245).
#     model = elegy.load(model_path)
#     model.run_eagerly = True # Settable attribute. Required to be true for ppmodel.

#     alpha_norm = (alpha - 20.) / 55. # Min max scaling
#     cmf_norm = (cmf - 4.5) / 4. # Min max scaling
    
#     def nn_predict(xs):
#         # Make NN predictions on xs.
#         xs = jnp.concatenate([jnp.array([alpha_norm, cmf_norm]), xs]) # Create 7d input to NN.
#         yhat = model.predict(xs)
#         yhat = yhat * 8.268953 # Undo max scaling.
#         yhat = jnp.exp(yhat) - 1. # Undo logp1 transform of target output.
#         return yhat
#     return nn_predict

# def define_target_log_prob(nn_predict, xloc, iloc, observed, uncertainty, key):
#     def target_log_prob(xs):
#         # Include logprior in loglikelihood. This keeps HMC from going off into no-mans land.
#         penalty = 1e11 # 1e20 #1e11 # Penalty for leaving uniform prior region. Should be at least 1e11 or so. 
#         nlogprior = 0.
#         for i in range(5):
#             nlogprior += penalty * jnp.abs((jnp.minimum(0., xs[i]))) # Penalty for being <0
#             nlogprior += penalty * jnp.abs((jnp.maximum(1., xs[i]) - 1.))  # Penalty for being >1
        
#         # Define random variables. Set these using oryx.core.intervene.
#         # alpha_key, cmf_key, obs_key, unc_key = random.split(key, 4)
#         # alpha_norm = random_variable(tfd.Uniform(0., 1.), name='alpha_norm')(alpha_key)
#         # cmf_norm = random_variable(tfd.Uniform(0., 1.), name='cmf_norm')(cmf_key)
#         # low, high = [0.]*45, [1e8]*45
#         # observed = random_variable(tfd.Independent(tfd.Uniform(low,high), reinterpreted_batch_ndims=1), name='observed')(obs_key)
#         # uncertainty = random_variable(tfd.Independent(tfd.Uniform(low,high), reinterpreted_batch_ndims=1), name='uncertainty')(unc_key)

#         yhat = nn_predict(xs)
        
#         # Interpolate to get predicted flux at both lattice and bin points.
#         yloc = jnp.interp(xloc, jnp.arange(245), yhat) 
#         # Integrate over bin regions, and compare to observed to get likelihood.
#         nloglikelihood = 0.0
#         for i in range(45):
#             # Integrate over bin by trapezoid method.
#             predicted : float = jnp.trapz(yloc[iloc[i]:(iloc[i+1]+1)], x=xloc[iloc[i]:(iloc[i+1]+1)])
#             # Use equation provided by Claudio for likelihood of bin.
#             nloglikelihood += ((predicted - observed[i])/uncertainty[i])**2
            
#         log_prob = -(nloglikelihood + nlogprior)
#         return log_prob #jnp.exp(loglikelihood) # Oryx expects likelihood.

#     #target_log_prob = intervene(target_log_prob, alpha_norm=alpha_norm, cmf_norm=cmf_norm, observed=observed, uncertainty=uncertainty) 
#     return target_log_prob

