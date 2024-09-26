"""Tests the operation of calc_difficulty_index.py"""

import numpy as np
import pytest
from metcalcpy.calc_difficulty_index import forecast_difficulty

__author__ = "Lindsay Blank (NCAR)"

def test_forecast_difficulty():
    """
    Test that the output of forecast_difficulty function is correct.
    
    Returns
    -------
    None.
    """
    
    #Settings
    EPS = np.finfo(np.float32).eps
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
    mean_height = 11.0 #mean wave height in feet
    mode_height = np.sqrt(2.0 / np.pi) * mean_height

    fieldijn = np.random.rayleigh(scale=mode_height,
            size=(nlat, nlon, nmembers))
    muij = np.mean(fieldijn, axis=-1)
    pertijn = fieldijn - np.dstack([muij] * nmembers)
    sigmaij = np.sqrt(np.mean(pertijn * pertijn, axis=-1))

    threshold = 9.0
    regularize = 0.01
    smax = 9.0
    sigma_max = smax + np.zeros_like(sigmaij)
    thresh_eps = 2.0
    kwargs = {'thresh_eps': thresh_eps, 'threshold_type': 'proximity'}

    assert 0.9095608641027515 == forecast_difficulty(sigmaij, muij, threshold, fieldijn,
            Aplin=None, sigma_over_mu_ref=EPS)[0][0]
    assert 0.8191620255148825 == forecast_difficulty(sigmaij, muij, threshold, fieldijn,
            Aplin=None, sigma_over_mu_ref=EPS)[8][17]
    assert 1.227707670365556 == forecast_difficulty(sigmaij, muij, threshold, fieldijn,
            Aplin=None, sigma_over_mu_ref=EPS)[4][9]


if __name__ == "__main__":
    test_forecast_difficulty()

