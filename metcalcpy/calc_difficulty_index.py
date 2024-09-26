# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** Naval Research Lab
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
# -*- coding: utf-8 -*-
"""
Program Name: calc_difficulty_index.py

Forecast decision difficulty indices.

Implement a set of decision difficulty indices for forecasts of postive
definite quantities such as wind speed and wave height.
"""

import numpy as np
from metcalcpy.piecewise_linear import PiecewiseLinear as plin
from metcalcpy.util.safe_log import safe_log

__author__ = 'Bill Campbell (NRL) and Lindsay Blank (NCAR)'
__version__ = '0.1.0'


# Enforce positive definiteness of quantities such as standard deviations
EPS = np.finfo(np.float32).eps
# Only allow 2D fields for now
FIELD_DIM = 2


def _input_check(sigmaij, muij, threshold, fieldijn, sigma_over_mu_ref, under_factor, logger=None):
    """
    Check for valid input to _difficulty_index.

    Parameters
    ----------
    sigmaij : 2D numpy array
        Positive definite array of standard deviations of a 2D field.
    muij : Float scalar or 2D numpy array
        The mean values corresponding to sigmaij.
    threshold : Float (or int) scalar
        A significant value to be compared with values of the forecast field
        for each ensemble member.
    sigma_over_mu_ref : Scalar
    fieldijn : 3D numpy array
        Values of the forecast field. Third dimension is ensemble member.
    under_factor : Float scalar
        Must be between 0 and 1.

    Returns
    -------
    None.

    """
    safe_log(logger, "debug",f"Checking input parameters: sigmaij shape: {np.shape(sigmaij)}, "
                 f"muij type: {type(muij)}, threshold: {threshold}, "
                 f"fieldijn shape: {np.shape(fieldijn)}, sigma_over_mu_ref: {sigma_over_mu_ref}, "
                 f"under_factor: {under_factor}")
    assert isinstance(threshold, (int, float, np.int32, np.float32))
    safe_log(logger, "debug", f"Threshold type is valid: {type(threshold)}")
    assert np.ndim(sigmaij) == FIELD_DIM
    safe_log(logger, "debug", f"sigmaij is {FIELD_DIM}D as expected.")
    assert np.all(sigmaij) >= EPS
    fieldshape = np.shape(fieldijn)
    # muij is a scalar or 2D array
    if isinstance(muij, np.ndarray):
        # If muij is an array, it must have the same shape as sigmaij
        assert np.shape(muij) == np.shape(sigmaij)
        safe_log(logger, "debug", "muij is a valid array and matches the shape of sigmaij.")
    else:
        safe_log(logger, "debug", "muij is a scalar.")
    assert sigma_over_mu_ref >= EPS
    safe_log(logger, "debug", "sigma_over_mu_ref is valid.")
    assert np.shape(sigmaij) == tuple(np.squeeze(fieldshape[0:-1]))
    safe_log(logger, "debug", "sigmaij and fieldijn shapes are compatible.")
    assert isinstance(under_factor, (int, float,
                                     np.int32, np.float32))
    assert 0.0 <= under_factor <= 1.0
    safe_log(logger, "debug", "under_factor is valid and within range.")

def _difficulty_index(sigmaij, muij, threshold, fieldijn, Aplin, sigma_over_mu_ref=EPS, under_factor=0.5, logger=None):
    """
    Calculates version 6.1 of forecast difficulty index.
    The threshold terms all penalize equal (or slightly unequal) spread.

    Parameters
    ----------
    sigmaij : 2D numpy array
        Positive definite array of standard deviations of a 2D field.
    muij : Float scalar or 2D numpy array
        The mean values corresponding to sigmaij.
    threshold : Float (or int) scalar
        A significant value to be compared with values of the forecast field
        for each ensemble member.
    fieldijn : 3D numpy array
        Values of the forecast field. Third dimension is ensemble member.
    Aplin: PiecewiseLinear object (envelope function)
        Essentially a piecewise linear localization function based on muij
    sigma_over_mu_ref : Scalar, optional
        Highest value of sigmaij/muij for past 5 days (nominally).
    under_factor: Scalar, optional
        Defaults to 0.5 except should be 0.4 for v4 difficulty index

    Returns
    -------
    dij : 2D numpy array
        Normalized v6.1 difficulty index ([0,1.5]). Larger (> 0.5)
        means more difficult.

    """
    # Check for valid input
    _input_check(sigmaij, muij, threshold, fieldijn, sigma_over_mu_ref, under_factor, logger=logger)
    safe_log(logger, "debug", "Input check passed successfully.")
    # Variance term in range 0 to 1
    safe_log(logger, "debug", "Calculating variance term.")
    sigma_over_mu = sigmaij / muij
    sigma_over_mu_max = np.nanmax(sigma_over_mu)
    safe_log(logger, "debug", f"sigma_over_mu_max: {sigma_over_mu_max}")
    
    # Force reference value to be greater than current max of sigmaij / muij
    sigma_over_mu_ref = np.nanmax([sigma_over_mu_ref,
                                   sigma_over_mu_max])
    variance_term = sigma_over_mu / sigma_over_mu_ref
    safe_log(logger, "debug", f"variance_term calculated, max variance_term: {np.nanmax(variance_term)}")

    # Depends on under_factor.
    under_threshold_count =\
        np.ma.masked_greater_equal(fieldijn, threshold).count(axis=-1)
    nmembers = np.shape(fieldijn)[-1]
    under_prob = under_threshold_count / nmembers
    safe_log(logger, "debug", f"under_threshold_count: {under_threshold_count}, under_prob: {under_prob}")
    # Threshold term in range 0 to 1
    threshold_term = 1.0 - np.abs(1.0 - under_factor - under_prob)
    safe_log(logger, "debug", f"threshold_term: {threshold_term}")

    # Linear taper factor
    taper_term = Aplin.values(muij)
    safe_log(logger, "debug", f"taper_term: {taper_term}")
    # Difficulty index is the average of the two terms
    # multiplied by a linear taper factor
    dij = 0.5 * taper_term * (variance_term + threshold_term)
    safe_log(logger, "info", f"Difficulty index calculation complete. Max dij: {np.nanmax(dij)}, Min dij: {np.nanmin(dij)}")
    return dij


def forecast_difficulty(sigmaij, muij, threshold, fieldijn,
                        Aplin, sigma_over_mu_ref=EPS, logger=None):
    """
    Calls private function _difficulty_index, 
    to calculate version (v6.1) of forecast difficulty index.

    Parameters
    ----------
    sigmaij : 2D numpy array
        Positive definite array of standard deviations of a 2D field.
    muij : Float scalar or 2D numpy array
        The mean values corresponding to sigmaij.
    threshold : Float (or int) scalar
        A significant value to be compared with values of the forecast field
        for each ensemble member.
    fieldijn : 3D numpy array
        Values of the forecast field. Third dimension is ensemble member.
    Aplin : PiecewiseLinear object (envelope function), optional
        Essentially a piecewise linear localization function based on muij
    sigma_over_mu_ref : Scalar, optional
        Highest value of sigmaij/muij for past 5 days (nominally).

    Returns
    -------
    dij : 2D numpy array
        Normalized difficulty index ([0,1.5]).
        Larger (> 0.5) means more difficult.

    """
    safe_log(logger, "debug", f"sigmaij shape: {sigmaij.shape}, muij: {'scalar' if isinstance(muij, float) else muij.shape}, "
                 f"threshold: {threshold}, fieldijn shape: {fieldijn.shape}, sigma_over_mu_ref: {sigma_over_mu_ref}")
    if Aplin is None:
        safe_log(logger, "info", "No Aplin provided. Creating default Aplin object (version 6.1).")
        #  Default to envelope version 6.1
        xunits="feet"                                                                           
        A6_1_name = "A6_1"                                                          
        A6_1_left = 0.0                                                                         
        A6_1_right = 0.0                                                                        
        A6_1_xlist = [3.0, 9.0, 12.0, 21.0]                                                         
        A6_1_ylist = [0.0, 1.5, 1.5, 0.0]                                                   
        Aplin =\
                plin(None, A6_1_xlist, A6_1_ylist, xunits=xunits,
                        right=A6_1_right, left=A6_1_left,                                       
                        name=A6_1_name)
        safe_log(logger, "debug", "Default Aplin object created.")
    safe_log(logger, "debug", "Calling _difficulty_index function.")                                                                  
    dij = _difficulty_index(sigmaij, muij, threshold, fieldijn,
                       Aplin, sigma_over_mu_ref, logger=logger)
    safe_log(logger, "info", "Forecast difficulty index calculation completed.")
    return dij

if __name__ == "__main__":
    pass
