"""Diagnostics relevant to Land/Surface applications"""

from xarray.core.dataarray import DataArray
from pandas.core.series import Series

def calc_tci(soil_data,sfc_flux_data,skipna=True):
  """ Function for computing the Terrestrial Coupling Index

  Args:
      soil_data (Xarray DataArray or Pandas Series): The moisture variable to use for computing TCI.
      sfc_flux_data (Xarray DataArray or Pandas Series): The latent heat flux variable to use for computing TCI.
      skipna (bool): Skip NA values. Passed to Pandas or Xarray.
 
  Returns:
      Xarray DataArray or float32: If Xarray DataArray's are passed, then an Xarray DataArray 
      containing the gridded TCI is returned. If a Pandas Series is passed, then a single TCI 
      value is returned.

  Raises:
      TypeError: If an unrecognized object type is passed, or the object types do not match.

  Reference:
      Dirmeyer, P. A., 2011: The terrestrial segment of soil moisture-climate coupling. *Geophys. Res. Lett.*, **38**, L16702, doi: 10.1029/2011GL048268.
  
  """

  # For Xarray objects, compute the mean 
  if isinstance(soil_data,DataArray) and isinstance(sfc_flux_data,DataArray):
    soil_mean = soil_data.mean(dim='time',skipna=skipna)
    soil_count = soil_data.count(dim='time')
    sfc_flux_mean = sfc_flux_data.mean(dim='time',skipna=skipna)
    soil_std = soil_data.std(dim='time',skipna=skipna)
    numer = ((soil_data-soil_mean) * (sfc_flux_data-sfc_flux_mean)).sum(dim='time',skipna=skipna)

  # For Pandas objects, compute the mean
  elif isinstance(soil_data,Series) and isinstance(sfc_flux_data,Series):
    soil_mean = soil_data.mean(skipna=skipna)
    soil_count = soil_data.count()
    sfc_flux_mean = sfc_flux_data.mean(skipna=skipna)
    soil_std = soil_data.std(skipna=skipna)
    numer = ((soil_data-soil_mean) * (sfc_flux_data-sfc_flux_mean)).sum(skipna=skipna)

  # No other object types are supported
  else:
    raise TypeError("Only Xarray DataArray or Pandas DataFrame Objects are supported. Input objects must be of the same type. Got "+str(type(soil_data))+" for soil_data and "+str(type(sfc_flux_data))+" for sfc_flux_data")

  # Compute the covariance term
  covarTerm = numer / soil_count 

  # Return the Terrestrial Coupling Index (TCI)
  return covarTerm/soil_std

