"""Diagnostics relevant to Land/Surface applications"""

from xarray.core.dataarray import DataArray
from pandas.core.series import Series

def calc_tci(soil_data,sfc_flux_data):
  """ Function for computing the Terrestrial Coupling Index 

  :param soil_data: The land data for the function
  :type soil_data: Xarray DataArray, Pandas DataFrame
  
  :param sfc_flux_data: The surface flux data for the function
  :type sfc_flux_data: Xarray DataArray, Pandas DataFrame
 
  :raises: TypeError
  
  :return: Terrestrial Coupling Index
  :rtype: Xarray DataArray, Pandas DataFrame

  :param test:
  """

  # For Xarray objects, compute the mean 
  if isinstance(soil_data,DataArray) and isinstance(sfc_flux_data,DataArray):
    soil_mean = soil_data.mean(dim='time')
    soil_count = soil_data.count(dim='time')
    sfc_flux_mean = sfc_flux_data.mean(dim='time')
    soil_std = soil_data.std(dim='time')
    numer = ((soil_data-soil_mean) * (sfc_flux_data-sfc_flux_mean)).sum(dim='time')

  # For Pandas objects, compute the mean
  elif isinstance(soil_data,Series) and isinstance(sfc_flux_data,Series):
    soil_mean = soil_data.mean()
    soil_count = soil_data.count()
    sfc_flux_mean = sfc_flux_data.mean()
    soil_std = soil_data.std()
    numer = ((soil_data-soil_mean) * (sfc_flux_data-sfc_flux_mean)).sum()

  # No other object types are supported
  else:
    raise TypeError("Only Xarray DataArray or Pandas DataFrame Objects are supported. Input objects must be of the same type. Got "+str(type(soil_data))+" for soil_data and "+str(type(sfc_flux_data))+" for sfc_flux_data")

  # Compute the covariance term
  covarTerm = numer / soil_count 

  # Return the Terrestrial Coupling Index (TCI)
  return covarTerm/soil_std

