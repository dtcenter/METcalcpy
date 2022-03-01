# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** CIRES, Regents of the University of Colorado
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
import numpy as np
import xarray as xr


def zonal_mean(dat,dimvar='longitude'):
    """Compute the zonal mean.
    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a dimension that you want to compute a zonal mean on
    dimvar: Name of the dimension to compute the zonal mean.  Longitude is the 
        default if it's not specified
    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        the mean across the zonal dimension
    """
    return dat.mean(dimvar)


def meridional_mean(dat, lat1, lat2, dimvar='latitude'):
    """Compute the cos(lat) weighted mean of a quantity between two latitudes.
    Parameters
    ----------
    dat : `xarray.DataArray` or `xarray.Dataset`
        data containing a dimension that you want ot compute a meridional mean
        on that spans lat1 and lat2
    lat1 : float
        The beginning latitude limit of the band average.  This should always be less 
        than lat2
    lat2 : float
        The ending latitude limit of the band average.  This should always be greater
        than lat1
    dimvar: Name of the dimension to compute the meridional mean.  Latitude is the
        default if it's not specified
    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        the weighted mean across the latitude dimension limited
        by lat1 and lat2
    """

    # Check inputs
    if lat1 > lat2:
        raise ValueError('lat1 is greater than lat2, but it must be less than lat2')
    elif lat1 == lat2:
        raise ValueError('lat1 is equal to lat2, but it must be less than lat2')

    wgts = np.cos(np.deg2rad(dat[dimvar].where((dat[dimvar] >= 60) & (dat[dimvar] <= 90),drop=True)))

    return dat.where((dat[dimvar] >= 60) & (dat[dimvar] <= 90),drop=True).weighted(wgts).mean(dimvar)
