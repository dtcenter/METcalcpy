import sys, os, re
import argparse
import xarray as xr
import numpy as np
import pytest

"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
sys.path.append("../metcalcpy/contributed/spacetime")
from spacetime import mjo_cross
from spacetime import get_symmasymm
from spacetime_utils import save_Spectra


def cross_spectral(datapath, pathout):
    '''
       Taking the logic from the original code, cross_spectra.py to generate
       a cross spectral analysis.

       Args:
           datapath: directory path to the input data.
           pathout:  directory path to where output data will reside.
    '''

    filename = os.path.join(datapath, 'precip.erai.sfc.1p0.2x.2014-2016.nc')
    ds = xr.open_dataset(filename )

    """
    Set parameters for the spectra calculation.
    spd:  Number of observations per day.
    nperseg:  Number of data points per segment.
    segOverlap:  How many data points of overlap between segments. If negative there is overlap
                 If positive skip that number of values between segments.
    Symmetry:    Can be "symm", "asymm" for symmetric or anti-symmetric across the equator. "latband" 
                 if the spectra averaged across the entire latitude band are requested.
    latMin:      Minimum latitude to include.
    latMax:      Maximum latitude to include.
    """
    spd = 2
    nperseg = 46 * spd
    segOverLap = -20 * spd
    Symmetry = "symm"
    latMin = -15.
    latMax = 15.
    datestrt = '2015-12-01'  # plot start date, format: yyyy-mm-dd
    datelast = '2016-03-31'  # plot end date, format: yyyy-mm-dd

    z = ds.precip
    z = z.sel(lat=slice(latMin, latMax))
    z = z.sel(time=slice(datestrt, datelast))
    z = z.squeeze()
    latC = ds.lat.sel(lat=slice(latMin, latMax))

    # ds = xr.open_dataset(datapath+'precip.erai.sfc.1p0.'+str(spd)+'x.2014-2016.nc')
    filename = os.path.join(datapath, 'precip.erai.sfc.1p0.2x.2014-2016.nc' )
    ds = xr.open_dataset(filename)

    x = ds.precip
    x = x.sel(lat=slice(latMin, latMax))
    x = x.sel(time=slice(datestrt, datelast))
    x = x.squeeze()
    latA = ds.lat.sel(lat=slice(latMin, latMax))
    ds.close()

    erai_850 = 'div.erai.850.1p0.' + str(spd) + 'x.2014-2016.nc'
    erai_file = os.path.join(datapath, erai_850)
    ds = xr.open_dataset(erai_file)
    y = ds.div
    y = y.sel(lat=slice(latMin, latMax))
    y = y.sel(time=slice(datestrt, datelast))
    y = y.squeeze()
    latB = ds.lat.sel(lat=slice(latMin, latMax))

    erai_200 = 'div.erai.200.1p0.' + str(spd) + 'x.2014-2016.nc'
    erai_200_file = os.path.join(datapath, erai_200)
    ds = xr.open_dataset(erai_200_file)
    w = ds.div
    w = y.sel(lat=slice(latMin, latMax))
    w = y.sel(time=slice(datestrt, datelast))
    w = y.squeeze()
    latB = ds.lat.sel(lat=slice(latMin, latMax))

    if any(latA - latB) != 0:
        print("Latitudes must be the same for both variables! Check latitude ordering.")

    print("get symmetric/anti-symmetric components:")
    if Symmetry == "symm" or Symmetry == "asymm":
        X = get_symmasymm(x, latA, Symmetry)
        Y = get_symmasymm(y, latB, Symmetry)
        Z = get_symmasymm(z, latC, Symmetry)
        W = get_symmasymm(w, latB, Symmetry)
    else:
        X = x
        Y = y
        Z = z
        W = w

    print("compute cross-spectrum:")
    """
    The output from mjo_cross includes:
    STC = [8,nfreq,nwave]
    STC[0,:,:] : Power spectrum of x
    STC[1,:,:] : Power spectrum of y
    STC[2,:,:] : Co-spectrum of x and y
    STC[3,:,:] : Quadrature-spectrum of x and y
    STC[4,:,:] : Coherence-squared spectrum of x and y
    STC[5,:,:] : Phase spectrum of x and y
    STC[6,:,:] : Phase angle v1
    STC[7,:,:] : Phase angle v2
    freq 
    wave
    number_of_segments
    dof
    prob
    prob_coh2
    """
    result = mjo_cross(X, Y, nperseg, segOverLap)
    STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
    freq = result['freq']
    freq = freq * spd
    wnum = result['wave']
    # save spectra in netcdf file
    fileout = 'SpaceTimeSpectra_ERAI_P_D850_' + Symmetry + '_' + str(spd) + 'spd'
    print('saving spectra to file: ' + pathout + fileout + '.nc')
    save_Spectra(STC, freq, wnum, fileout, pathout)

    result = mjo_cross(X, W, nperseg, segOverLap)
    STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
    freq = result['freq']
    freq = freq * spd
    wnum = result['wave']
    # save spectra in netcdf file
    fileout = 'SpaceTimeSpectra_ERAI_P_D200_' + Symmetry + '_' + str(spd) + 'spd'
    print('saving spectra to file: ' + pathout + fileout + '.nc')
    save_Spectra(STC, freq, wnum, fileout, pathout)

    result = mjo_cross(X, Z, nperseg, segOverLap)
    STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
    freq = result['freq']
    freq = freq * spd
    wnum = result['wave']
    # save spectra in netcdf file
    fileout = 'SpaceTimeSpectra_ERAI_TRMM_P_' + Symmetry + '_' + str(spd) + 'spd'
    print('saving spectra to file: ' + pathout + fileout + '.nc')
    save_Spectra(STC, freq, wnum, fileout, pathout)


@pytest.mark.skip("test data is too large to save in repository")
def test_nc_files_created():
    '''
       Check that the three expected
       netCDF files for a cross-spectral analysis are created.

    '''

    # !!! Modify these two paths to reflect where your input and output data reside !!!
    datapath = '/d1/projects/METcalcpy/METcalcpy_Data/TropicalDiagnostics'
    pathout = '/home/minnawin/METcalcpy/output'

    cross_spectral(datapath, pathout)
    expected_filenames = ['SpaceTimeSpectra_ERAI_P_D850_symm_2spd.nc','SpaceTimeSpectra_ERAI_P_D200_symm_2spd.nc',
                      'SpaceTimeSpectra_ERAI_TRMM_P_symm_2spd.nc']
    expected_full_files = []
    for fn in expected_filenames:
        expected = os.path.join(pathout, fn)
        expected_full_files.append(expected)

    actual_files = []
    for root, dir, files in os.walk(pathout):
        for file in files:
            match = re.match(r'.*.nc$', file)
            if match:
                actual_full_file = os.path.join(pathout, file)
                actual_files.append(actual_full_file)
                if actual_full_file in expected_full_files:
                    assert True
                else:
                    # The created file isn't one of the expected files, Fail
                    assert False

    #clean up the files you just created
    for actual in actual_files:
        os.remove(actual)



