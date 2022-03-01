# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
This is a collection of utility functions.

List of functions:

save_Spectra:
    Save space-time spectra for plottting.

lonFlip:
    Flip longitudes from -180:180 to 0:360 or vice versa.

"""

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def save_Spectra(STCin, freq_in, wnum_in, filename, filepath, opt=False):
    nc = Dataset(filepath + filename + '.nc', 'w', format='NETCDF4')

    nvar, nfrq, nwave = STCin.shape
    # dimensions
    nc.createDimension('freq', nfrq)
    nc.createDimension('wnum', nwave)
    nc.createDimension('var', nvar)

    # variables
    freq = nc.createVariable('freq', 'double', ('freq',))
    wnum = nc.createVariable('wnum', 'int', ('wnum',))
    var = nc.createVariable('var', 'int', ('var',))
    STC = nc.createVariable('STC', 'double', ('var', 'freq', 'wnum',))

    # attributes
    STC.varnames = ['PX', 'PY', 'CXY', 'QXY', 'COH2', 'PHA', 'V1', 'V2']
    STC.long_name = "Space time spectra"
    freq.units = "cpd"
    freq.long_name = "frequency"
    wnum.units = ""
    wnum.long_name = "zonal wavenumber"
    var.long_name = "variable number"

    # data
    var[:] = np.linspace(0, nvar - 1, nvar)
    freq[:] = freq_in
    wnum[:] = wnum_in
    STC[:, :, :] = STCin

    nc.close()


def lonFlip(data,lon):
    """
    Change the longitude coordinates from -180:180 to 0:360 or vice versa.
    :param data: Input xarray data array (time x lat x lon).
    :param lon: Longitude array of the input data.
    :return: dataflip
    """

    lonnew = lon.values

    if lonnew.min() < 0:
        # change longitude to 0:360
        ilonneg = np.where(lon<0)
        nlonneg = len(ilonneg[0])
        ilonpos = np.where(lon>=0)
        nlonpos = len(ilonpos[0])

        lonnew[0:nlonpos] = lon[ilonpos[0]].values
        lonnew[nlonpos:] = lon[ilonneg[0]].values + 360

        dataflip = xr.DataArray(np.roll(data, nlonneg, axis=2), dims=data.dims,
                          coords={'time': data['time'], 'lat': data['lat'], 'lon': lonnew})

    else:
        # change longitude to -180:180
        ilonneg = np.where(lon >= 180)
        nlonneg = len(ilonneg[0])
        ilonpos = np.where(lon < 180)
        nlonpos = len(ilonpos[0])

        lonnew[0:nlonneg] = lon[ilonneg[0]].values - 360
        lonnew[nlonneg:] = lon[ilonpos[0]].values

        dataflip = xr.DataArray(np.roll(data, nlonpos, axis=2), dims=data.dims,
                          coords={'time': data['time'], 'lat': data['lat'], 'lon': lonnew})

    return dataflip
