"""
Utilities for running the MCMC. 
Most of the training data details are imported from preprocess.py
These utils focus on the observation data. 
"""

import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
#assert jax.default_backend() == 'gpu'
import keras_core as keras

import preprocess
from preprocess import INPUTS, PARAMETERS, PARAMETERS_SPECIFIED, RIGIDITY_VALS
from preprocess import transform_input, untransform_input
from preprocess import PARAMETERS_MIN, PARAMETERS_MAX


def read_experiment_summary(filename) -> pd.DataFrame:
    """
    Read .dat filename that describes experimental conditions during time intervals.
    Update 2023: Added vspoles and vspoles std
    """
    # Header reads "time interval; alpha avg; cmf avg; vspoles avg; alpha std; cmf std; vspoles std"
    df = pd.read_csv(filename, sep=' ', skiprows=1, names=['interval', 'alpha', 'cmf', 'vspoles', 'alpha_std', 'cmf_std', 'vspoles_std'])

    # Parse interval
    df['beginning'] = df.interval.apply(lambda x: pd.to_datetime(x.split('-')[0], format='%Y%m%d'))
    df['ending'] = df.interval.apply(lambda x: pd.to_datetime(x.split('-')[0], format='%Y%m%d'))
    return df


def index_experiment_files(filename)->pd.DataFrame:
    """Create list of experiments that need to be done.
    filename = f'../data/2023/{EXPERIMENT_NAME}_heliosphere.dat'
    """
    df = read_experiment_summary(filename)
    # The datasets to be fitted are: PAMELA_H-ApJ2013, PAMELA_H-ApJL2018, and AMS02_H-PRL2021.
    # You should use the neg models for data files up to February 2015, and the pos models for data files from October 2013.
    # So, between October 2013 and February 2015, the data files should be fitted independently with both neg and pos models.
    # All PAMELA files are before February 2015, so only neg models for them.
    # For AMS02 files, 20130925-20131021.dat is the first file to be fitted with pos models, while 20150124-20150219.dat is the last file to be fitted with neg models.
    # 
    # For PAMELA_H-ApJL2018, the files 20130928-20131025.dat, 20131121-20131219.dat, and 20140115-20140211.dat should be fit independently with both neg and pos models.

    dfneg = df[df.beginning < pd.to_datetime('March 1 2015')].copy(deep=True)
    dfneg['polarity'] = 'neg'

    dfpos = df[df.ending >= pd.to_datetime('October 1 2013')].copy(deep=True)
    dfpos['polarity'] = 'pos'

    rval = pd.concat([dfneg, dfpos], axis=0, ignore_index=True)
    return rval


def get_interval(filename, index: int) -> str:
    """
    Return time interval for the given index (zero indexed, so first index is 0).

    """
    assert index >= 0, index
    df = read_experiment_summary(filename)
    if index < df.shape[0]:
        return df['interval'].values[index]
    else:
        raise ValueError(f'Index {index} exceeds zero-indexed list of intervals in {filename} of length {df.shape[0]}.')


def get_parameters(filename, interval: str = None, return_std=False):
    """ Load  alpha and cmf from file supplied by Claudio.
        Use alpha and cmf mean over interval. 
        Ignores std for now.
    """
    # if 'BR2461.dat' in filename:
    #     assert interval == None
    #     alpha, cmf = 69.19, 5.17 # These are fixed for the current experiment.

    df = df = read_experiment_summary(filename)
    row = df.loc[df['interval'] == interval]
    alpha = row['alpha'].values[0]
    cmf = row['cmf'].values[0]
    if not return_std:
        return row['alpha'].values[0], row['cmf'].values[0], row['vspoles'].values[0]
    else:
        return (row['alpha'].values[0], 
                row['cmf'].values[0], 
                row['vspoles'].values[0],
                row['alpha_std'].values[0], 
                row['cmf_std'].values[0], 
                row['vspoles_std'].values[0])
    
    
def load_data_ams(filename):
    """ Load AMS data from Claudio. Each file contains measurements over a certain time interval. 
    Args:
        filename = Filename of observations.
                   Original dataset was '../data/BR2461.dat'
                   New datasets are in ../data/oct2022/
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


def remove_consecutive_duplicates_simple(samples:np.array, atol=0.0)->np.array:
    ''' 
    Remove consecutive duplicate rows from array. This means sample from MCMC was rejected.
    Args:
        samples = 2d array, where each row is a sample
        atol = absolute tolerance for being equal
    Returns:
        rval = 2d array with consecutive duplicate rows removed
    '''
    consecutive_repeat_rows = np.all(np.isclose(samples[1:,:], samples[:-1,:], atol=atol, rtol=0.0), axis=1)
    # The first sample is not a repeat by definition.
    consecutive_repeat_rows = np.concatenate([np.array([False]), consecutive_repeat_rows])
    return samples[~consecutive_repeat_rows, :]


def remove_consecutive_duplicates(samples, aux=[], atol=0.0):
    ''' 
    Remove consecutive duplicate rows from array. This means sample from MCMC was rejected.
    Args:
        samples = 2d array, where each row is a sample
        aux = list of additional 2d arrays containing information that we want to subset.
        atol = absolute tolerance for being equal
    Returns:
        rval = 2d array with consecutive duplicate rows removed
    '''
    consecutive_repeat_rows = np.all(np.isclose(samples[1:,:], samples[:-1,:], atol=atol, rtol=0.0), axis=1)
    # The first sample is not a repeat by definition.
    consecutive_repeat_rows = np.concatenate([np.array([False]), consecutive_repeat_rows])
    if len(aux) == 0:
        return samples[~consecutive_repeat_rows, :]
    else:
        aux_return = []
        for a in aux:
            a = a[~consecutive_repeat_rows, ...]
            aux_return.append(a)
        return samples[~consecutive_repeat_rows, :], aux_return


# def define_nn_pred(model_path, normalize_input_flag=False, untransform_output_flag=True, rebin_output_flag=False):
#     # Load trained NN model that maps 7 parameters to predicted flux at rigidity vals range(245).
#     model = keras.models.load_model(model_path)
#     model.run_eagerly = True # Settable attribute. Required to be true for ppmodel.

#     def f(x):
#         # Make NN predictions on xs.
#         if normalize_input_flag:
#             x = minmax_scale_input(x)
#         yhat = model.predict(x)
#         if untransform_output_flag:
#             yhat = untransform_output(yhat)
#         if rebin_output_flag:
#             yhat = rebin_output(yhat)
#         return yhat
#     return f


def untransform_output(yhat):
    """
    NN is trained on transformed data. 
    Rigidity measurments were transformed at log(x)/Y_LOG_MAX.
    See preprocess.py
    """
    yhat = yhat * preprocess.Y_LOG_MAX # Undo max scaling.
    yhat = jnp.exp(yhat) # Undo log transform of target output.
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


def remove_outliers(samples,  min_bound=PARAMETERS_MIN, max_bound=PARAMETERS_MAX, buffer=0):
    """Remove samples that exceed min, max range. 
    """
    assert samples.shape[1] == 5
    outlier = np.bitwise_or(samples < min_bound - buffer, samples > max_bound + buffer)
    outlier = np.any(outlier, axis=1)
    return samples[~outlier, :]



def _form_batch(params_trans, params_spec_trans):
    """
    Create jnp batch from sets of ordered features.
    These should already be transformed.
    Output is fed to NN. Might want to clean this up since we are using it for two purposes.
    Args:
        params_trans: jnp array with dimension 1
        params_spec_trans: tuple 
    Output:
        xs = njp array with dimensions (batch=1, -1)
    """
    if type(params_spec_trans) == tuple:
        # this is the case during HMC
        alpha, cmf, vspoles = params_spec_trans  # Should only have three.
        assert params_trans.ndim==1, params_trans.ndim
        xs = jnp.concatenate([jnp.array([alpha, cmf]), params_trans, jnp.array([vspoles])])
        # Assumes we are doing on example at a time, as in MCMC
        xs = jnp.reshape(xs, (1,8)) # Added because Jax was throwing shape error. Need batch dimension.
    else: # elif isinstance(params_spec_trans, type(jnp.array([])))
        # This is used after the MCMC to get nn predictions on the samples. Can handle batches.
        if params_trans.ndim==1:
            params_trans = params_trans.reshape((1,-1))
        if params_spec_trans.ndim==1:
            # Need to reshape into example by feature matrix.
            params_spec_trans = params_spec_trans.reshape((1,-1))
        broadcasted_spec = params_spec_trans.repeat(params_trans.shape[0], axis=0)
        xs = jnp.concatenate([broadcasted_spec[:,0:2], params_trans, broadcasted_spec[:,2:3]], axis=1)
    # else:
    #     raise Exception()
    return xs


def define_log_prob(model_path, data_path, parameters_specified, penalty=1e9):
    """
    Args:
        model_path: filename of model
        data_path: filename of experimental data for interval, e.g. path/AMS02_H-PRL2018_20110520-20110610.dat
        parameters_specified: tuple of alpha, cmf, vspoles.
        penalty = scalar to punish drifting outside zone of interest. 
    """
    # Load trained NN model that maps 7 parameters to predicted flux at RIGIDITY_VALS.
    model = keras.models.load_model(model_path)
    model.run_eagerly = True # Settable attribute (in elegy). Required to be true for ppmodel.

    # Load observation data from Claudio
    #xloc, iloc, observed, uncertainty = load_preprocessed_data_ams(data_path) # Replaced with bin midpoints
    bins, observed, uncertainty = load_data_ams(data_path)
    #bin_midpoints = (bins[:-1] + bins[1:])/2  # Arithmetic 
    bin_midpoints = (bins[:-1] * bins[1:]) ** 0.5  # Geometric mean seemed to work better in exp.
    parameters_specified_transformed = transform_input(jnp.array(parameters_specified))
    
    # def nn_predict(x):
    #     """
    #     Predict fluxes (untransformed) from transformed input. 
    #     """
    #     yhat = model(x)    
    #     yhat = untransform_output(yhat) # Undo scaling and minmax.
    #     return yhat

    def target_log_prob(xs):
        """
        Compute log likilihood of parameters given some data.
        Args:
            xs: 1d array containing parameters (transformed to be in range 0--1).
        Returns:
            log_prob: Scalar valued log probability
        """
        # Include logprior in loglikelihood. This keeps HMC from going off into no-mans land.
        nlogprior = 0.
        for i in range(5):
            nlogprior += penalty * jnp.abs((jnp.minimum(0., xs[i]))) # Penalty for being <0
            nlogprior += penalty * jnp.abs((jnp.maximum(1., xs[i]) - 1.))  # Penalty for being >1

        batch = _form_batch(xs, parameters_specified_transformed)
        yhat = model(batch)    
        yhat = yhat[0,:]  # Remove batch dimension.
        
        # Interpolate to get predicted flux at midpoint of bin points.
        yhat = jnp.interp(bin_midpoints, RIGIDITY_VALS, yhat)
        yhat = untransform_output(yhat.reshape((1,-1))).reshape(-1) # Undo scaling and minmax.
        
        # Compute log prob
        chi2 = (((yhat - observed)/uncertainty)**2).sum()
        log_prob = -chi2/2.  - nlogprior
        return log_prob


        # batch = _form_batch(xs, parameters_specified_transformed)
        # yhat = nn_predict(batch)
        # yhat = yhat[0,:]  # Remove batch dimension.
        
        # # Interpolate to get predicted flux at both lattice and bin points.
        # predicted = jnp.interp(bin_midpoints, RIGIDITY_VALS, yhat)
        # chi2 = (((predicted - observed)/uncertainty)**2).sum()
        # log_prob = -chi2/2.  - nlogprior
        # return log_prob
    
        # # Interpolate to get predicted flux at both lattice and bin points.
        # yloc = jnp.interp(xloc, RIGIDITY_VALS, yhat) 
        # nbins = len(iloc)-1
        # # Integrate over bin regions, and compare to observed to get likelihood.
        # chi2 = 0.0
        # for i in range(nbins):
        #     # Integrate over bin by trapezoid method.
        #     istart, istop = iloc[i], iloc[i+1]
        #     area = jnp.trapz(y=yloc[istart:(istop+1)], x=xloc[istart:(istop+1)])
        #     length = (xloc[istop] - xloc[istart])
        #     predicted = area / length
        #     # Use equation provided by Claudio for likelihood of bin.
        #     chi2 += ((predicted - observed[i])/uncertainty[i])**2
            
        # log_prob = -chi2/2.  - nlogprior
        # return log_prob

    return target_log_prob



    