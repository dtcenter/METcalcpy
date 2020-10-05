# -*- coding: utf-8 -*-
"""
Program Name: calc_difficulty_index.py

Forecast decision difficulty indices.

Implement a set of decision difficulty indices for forecasts of postive
definite quantities such as wind speed and wave height.

Created on Thu Feb 20 16:23:06 2020
Last modified on Mon Mar 30 11:25:30 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from piecewise_linear import PiecewiseLinear as plin

__author__ = 'Bill Campbell (NRL) and Lindsay Blank (NCAR)'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

# Enforce positive definiteness of quantities such as standard deviations
EPS = np.finfo(np.float32).eps
# Only allow 2D fields for now
FIELD_DIM = 2

def _input_check(sigmaij, muij, threshold, fieldijn, sigma_over_mu_ref, under_factor):
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

    assert isinstance(threshold, (int, float, np.int32, np.float32))
    assert np.ndim(sigmaij) == FIELD_DIM
    assert np.all(sigmaij) >= EPS
    fieldshape = np.shape(fieldijn)
    # muij is a scalar or 2D array
    if isinstance(muij, np.ndarray):
        # If muij is an array, it must have the same shape as sigmaij
        assert np.shape(muij) == np.shape(sigmaij)
    assert sigma_over_mu_ref >= EPS
    assert np.shape(sigmaij) == tuple(np.squeeze(fieldshape[0:-1]))
    assert isinstance(under_factor, (int, float,
                                     np.int32, np.float32))
    assert 0.0 <= under_factor <= 1.0


def _difficuly_index(sigmaij, muij, threshold, fieldijn, Aplin, sigma_over_mu_ref=EPS, under_factor=0.5):
    """
    Calculates public version (v7) of forecast difficulty index.
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
        Normalized v4567 difficulty index ([0,1.5]). Larger (> 0.5)
        means more difficult.

    """
    # Check for valid input
    _input_check(sigmaij, muij, threshold, fieldijn, sigma_over_mu_ref, under_factor)

    # Variance term in range 0 to 1
    sigma_over_mu = sigmaij / muij
    sigma_over_mu_max = np.nanmax(sigma_over_mu)
    
    # Force reference value to be greater than current max of sigmaij / muij
    sigma_over_mu_ref = np.nanmax([sigma_over_mu_ref,
                                   sigma_over_mu_max])
    variance_term = sigma_over_mu / sigma_over_mu_ref

    # Depends on under_factor.
    under_threshold_count =\
        np.ma.masked_greater_equal(fieldijn, threshold).count(axis=-1)
    nmembers = np.shape(fieldijn)[-1]
    under_prob = under_threshold_count / nmembers

    # Threshold term in range 0 to 1
    threshold_term = 1.0 - np.abs(1.0 - under_factor - under_prob)

    # Linear taper factor
    taper_term = Aplin.values(muij)

    # Difficulty index is the average of the two terms
    # multiplied by a linear taper factor
    dij = 0.5 * taper_term * (variance_term + threshold_term)

    return dij


def forecast_difficulty(sigmaij, muij, threshold, fieldijn,
                        Aplin=None, sigma_over_mu_ref=EPS):
    """
    Calls private function _index, 
    to calculate the public version (v7) of forecast difficulty index.

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
    if Aplin is None:
        # Envelope for public version (v7) the default
        xunits = 'feet'
        A7_name = "A7"
        A7_left = 0.0
        A7_right = 0.0
        A7_xlist = [3.0, 9.0, 12.0, 21.0]
        A7_ylist = [0.0, 1.0, 1.0, 0.0] 
        Aplin = plin(A7_xlist, A7_ylist, xunits=xunits,
                right=A7_right, left=A7_left, name=A7_name)

    dij = _difficuly_index(sigmaij, muij, threshold, fieldijn,
                       Aplin, sigma_over_mu_ref)

    return dij

# Get rid of main
def main():
    
    
    """
    Test harness for difficulty indices.

    Returns
    -------
    None.

    """
    nlon = 18
    nlat = 9
    nmembers = 10
    np.random.seed(12345)
    # Wave heights and wind speeds generally follow the Rayleigh distribution,
    # which is a Weibull distribution with a scale factor of 2,
    # and any shape parameter.
    # If the shape parameter is 1, then it is the same as a chi-square
    # distribution with 2 dofs.
    # Expected value E[x] = sigma * sqrt(pi/2)
    # mean_windspeed = 6.64  # meters/second
    xunits = 'feet'
    mean_height = 11.0  # mean wave height in feet
    # var_height = mean_height * mean_height * ((4.0 - np.pi) / 2.0)
    mode_height = np.sqrt(2.0 / np.pi) * mean_height
    fieldijn = np.random.rayleigh(scale=mode_height,
                                  size=(nlat, nlon, nmembers))
    muij = np.mean(fieldijn, axis=-1)
    pertijn = fieldijn - np.dstack([muij] * nmembers)
    sigmaij = np.sqrt(np.mean(pertijn * pertijn, axis=-1))
    threshold = 9.0
    regularizer = 0.01
    smax = 9.0
    sigma_max = smax + np.zeros_like(sigmaij)
    thresh_eps = 2.0
    kwargs = {'thresh_eps': thresh_eps, 'threshold_type': 'proximity'}

    # Envelope for version 7
    A7_name = "A7"
    A7_left = 0.0
    A7_right = 0.0
    A7_xlist = [3.0, 9.0, 12.0, 21.0]
    A7_ylist = [0.0, 1.0, 1.0, 0.0]
    A7 = plin(A7_xlist, A7_ylist, xunits=xunits,
                              right=A7_right, left=A7_left, name=A7_name)
    dijv7 = _difficuly_index(sigmaij, muij, threshold, fieldijn,
                         A7, sigma_over_mu_ref=EPS)
    
    # Test public forecast_difficulty, which should be the same as version 7
    dijfd = forecast_difficulty(sigmaij, muij, threshold, fieldijn, Aplin=A7,
                                sigma_over_mu_ref=EPS)

    print(dijfd)


if __name__ == "__main__":
    main()
