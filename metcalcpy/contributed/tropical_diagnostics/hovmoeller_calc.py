import numpy as np

"""
Routines used to compute Hovmoeller diagrams and pattern correlation.

Included:

lat_avg:

pattern_corr:
"""


def lat_avg(data, latmin, latmax):
    """
    Compute latitudinal average for hovmoeller diagram.
    :param data: input data (time, lat, lon)
    :type data: xarray.Dataarray
    :param latmin: southern latitude for averaging
    :type latmin: float
    :param latmax: northern latitude for averaging
    :type latmax: float
    :return: data (time, lon)
    :rtype: xarray.Dataarray
    """
    data = data.sel(lat=slice(latmin, latmax))
    units = data.attrs['units']
    data = data.mean(dim='lat')
    data.attrs['units'] = units
    data = data.squeeze()

    return data


def pattern_corr(a, b):
    """
    Compute the pattern correlation between two 2D (time, lon) fields
    :param a: (time, lon) data array
    :type a: float
    :param b: (time, lon) data array
    :type b: float
    :return: correlation
    :rtype: float
    """
    a1d = a.stack(lt=('time', 'lon'))
    b1d = b.stack(lt=('time', 'lon'))

    corr = np.corrcoef(a1d, b1d)
    corr = corr[0, 1]

    return corr
