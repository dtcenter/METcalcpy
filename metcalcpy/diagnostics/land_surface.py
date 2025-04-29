"""Diagnostics relevant to Land/Surface applications"""

import metpy.constants as mpconsts
import numpy as np
from metpy import calc as mpcalc
from metpy.interpolate import log_interpolate_1d as log_interp_1d
from metpy.units import units
import os
from pandas.core.series import Series
from xarray.core.dataarray import DataArray

def calc_ctp(pressure,temperature,station_index,start_pressure_hpa=-1,bot_pressure_hpa=100.0,top_pressure_hpa=300.0,interp=True,db=True,plotskewt=False,plotdir="",station_name=""):

  """ Function for computing the Convective Triggering Potential (CTP), defined as
      :math:`\\mathrm{CTP} = R_d \\int_{P_s - 100\\,\\text{hPa}}^{P_s - 300\\,\\text{hPa}} \\left( T_{\\text{env}} - T_{\\text{MALR}} \\right) \\, d\\ln p`

  Args:
      pressure (pint.Quantity or xarray.DataArray): the pressure variable with units of Hectopascals (hPa).
      temperature (pint.Quantity or xarray.DataArray): the temperature variable with units of Kelvin (K).
      station_index (int): the integer index of the station currently being processed. Use -1 if a single station is being passed.
      start_pressure_hpa (float, optional): the starting pressure to use. Default: -1 (bottom level in profile). This is :math:`\\, P_s` in the CTP equation.
      bot_pressure_hpa (float, optional): bottom pressure value of the layer, subtracted from start_pressure_hpa. Default: 100 hPa.
      top_pressure_hpa (float, optional): top pressure value of the layer, subtracted from start_pressure_hpa. Default: 300 hPa.
      interp (bool): Whether to interpolate data to exact pressures or use the closest. Default: True.
      db (bool): Print debugging statements. Default: False
      plotskewt (bool): Plot a Skew-T Log-P graphic of the CTP calculation. Default: False.
      plotdir (string, optional): Directory where Skew-T plots should be written. Default: "".
      station_name (string, optional): Location ID string used for labeling the Skew-T plot and image file name. Default: "".

  Returns:
      float32
 
  Reference:
      Findell, K. L., and E. A. B. Eltahir, 2003: Atmospheric Controls on Soil Moisture–Boundary Layer Interactions. Part I: Framework Development. J. Hydrometeor., 4, 552–569, https://doi.org/10.1175/1525-7541(2003)004<0552:ACOSML>2.0.CO;2.
      
      Also see: https://www.pauldirmeyer.com/coupling-metrics for a summary of this and other land-atmosphere coupling metrics.

  Notes:
      Pressure and temperature can either be a 1D profile or a 3D array of data. If a 3D array, station_index needs to be a non-negative value
      indicating the single column to extract from the 3D data. If they are 1D profiles, use -1 for the station_index as that will be the only
      profile to use.
 
  """
 
  # Import utility functions for CTP/HI 
  import metcalcpy.diagnostics.land_surface_ctphi_util as land_surface_ctphi_util

  # If the station index is a non-negative value, then extract the profile at the station index
  if station_index>=0:
    temperature=temperature.isel(sid=station_index).values*units('degK')
    pressure=pressure.isel(sid=station_index).values*units('hPa')

  # Subset the profile to only levels where pressure is > min_prs_profile
  min_prs_profile = 100.0*units('hPa')
  temperature = temperature[pressure>=min_prs_profile]
  pressure = pressure[pressure>=min_prs_profile]
  
  # A pressure difference for warning and a maximum value where CTP
  # won't be calculated. If warn_prs_diff is exceeded, a warning is printed but 
  # calculation proceeds. If max_prs_diff is exeeded, an error is printed
  # and missing data is returned. These are only used for the non-interpolation case.
  warn_prs_diff = 50.0*units('hPa')
  max_prs_diff = 100.0*units('hPa')

  # Find the starting pressure to use
  start_prs = land_surface_ctphi_util.find_start_pressure(start_pressure_hpa*units('hPa'),pressure,interp,max_prs_diff,db,'CTP')
  if start_prs < 0:
    return(-9999.*units('J/kg'))
  start_prs = start_prs*units('hPa')

  # Based on the starting pressure, set the initial layer bottom and top pressures
  layer_bot_prs = start_prs-(bot_pressure_hpa*units('hPa'))
  layer_top_prs = start_prs-(top_pressure_hpa*units('hPa'))

  if db:
    print("")
    print(f"TARGET LAYER BOTTOM PRESSURE: {layer_bot_prs.m}")
    print(f"TARGET LAYER TOP PRESSURE: {layer_top_prs.m}\n")
  
  # Obtain information at the top and bottom of the layer
  if interp:
    
    layer_bottom_temperature = log_interp_1d(layer_bot_prs.m,pressure.m,temperature.m)
    layer_top_temperature = log_interp_1d(layer_top_prs.m,pressure.m,temperature.m)
    layer_bottom_pressure = np.array([layer_bot_prs.m])
    layer_top_pressure = np.array([layer_top_prs.m])
    
    if db:
      print("")
      print(f"USING INTERPOLATED LAYER BOTTOM PRESSURE: {layer_bottom_pressure}")
      print(f"USING INTERPOLATED LAYER TOP PRESSURE: {layer_top_pressure}\n")
  
    # Find the top and bottom of the layer, where the interpolated values should be inserted
    if any(np.diff(pressure.m)[np.diff(pressure.m)>=0]):
      print("ERROR! PRESSURES DO NOT MONOTONICALLY DECREASE!")
      print("UNABLE TO COMPUTE CTP.")
      return(-9999.*units('J/kg'))
    layer_bot_idx = len(pressure.m)-np.searchsorted(pressure.m[::-1],layer_bottom_pressure,side="left")[0]
    layer_top_idx = len(pressure.m)-np.searchsorted(pressure.m[::-1],layer_top_pressure,side="left")[0]
    if db:
      print("")
      print(f"INSERTING INTERPOLATED BOT DATA AT INDEX: {layer_bot_idx}")
      print(f"INSERTING INTERPOLATED TOP DATA AT INDEX: {layer_top_idx}")
    
    # Create a new sounding to use, which has the interpolated T/P at bottom/top inserted
    prs = np.append(np.append(np.append(np.append(pressure.m[0:layer_bot_idx],layer_bottom_pressure),\
                    pressure.m[layer_bot_idx:layer_top_idx]),layer_top_pressure),pressure.m[layer_top_idx:])
    tmp = np.append(np.append(np.append(np.append(temperature.m[0:layer_bot_idx],layer_bottom_temperature),\
                    temperature.m[layer_bot_idx:layer_top_idx]),layer_top_temperature),temperature.m[layer_top_idx:])
    # Assign units to the new sounding variables
    prs = prs*units('hPa')
    tmp = tmp*units('degK')

    # Reset the variables as if this was the true sounding
    pressure = prs
    temperature = tmp

    # Find the new layer top and bottom indices, which should be the indices of the interpolated values that were inserted above
    layer_bot_idx = np.where(pressure.m==layer_bottom_pressure)[0][0]
    layer_top_idx = np.where(pressure.m==layer_top_pressure)[0][0]
    if db:
      print("")
      print(f"INDEX OF LAYER BOT: {layer_bot_idx}")
      print(f"INDEX OF LAYER TOP: {layer_top_idx}")

    # Compute the moist adiabatic lapse rate
    try:
      MALR = mpcalc.moist_lapse(pressure[layer_bot_idx:],layer_bottom_temperature*units('degK'),reference_pressure=layer_bottom_pressure*units('hPa'))
    except ValueError:
      print("UNABLE TO COMPUTE MALR IN calc_ctp()")
      return(-9999.*units('J/kg'))

  else:

    # Find the index of the closest value. 
    # We'll just take the difference between the top/bottom pressure and find the index of the minimum
    layer_bot_idx = np.argmin(np.abs(pressure-layer_bot_prs))
    layer_top_idx = np.argmin(np.abs(pressure-layer_top_prs))

    # Warn if the distance between the closest value to the bottom and top values exceeds warn_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=warn_prs_diff:
      print(f"WARNING! ACTUAL BOTTOM PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
      print(f"requested: layer_bot_prs = {layer_bot_prs.m} hPa")
      print(f"closest: layer_bot_prs = {pressure[layer_bot_idx].m} hPa")
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=warn_prs_diff:
      print(f"WARNING! ACTUAL TOP PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
      print(f"requested: layer_top_prs = {layer_top_prs.m} hPa")
      print(f"closest: layer_top_prs = {pressure[layer_top_idx].m} hPa")
    # Return missing data if the distance between the closest value to the bottom and top values exceeds max_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=max_prs_diff:
      print(f"ERROR! ACTUAL BOTTOM PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
      print(f"requested: layer_bot_prs = {layer_bot_prs.m} hPa")
      print(f"closest: layer_bot_prs = {pressure[layer_bot_idx].m} hPa")
      print("UNABLE TO COMPUTE CTP.")
      return(-9999.*units('J/kg'))      
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=max_prs_diff:
      print(f"ERROR! ACTUAL TOP PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
      print(f"requested: layer_top_prs = {layer_top_prs.m} hPa")
      print(f"closest: layer_top_prs = {pressure[layer_top_idx].m} hPa")
      print("UNABLE TO COMPUTE CTP.")
      return(-9999.*units('J/kg'))
    if db:
      print("")
      print(f"INDEX OF LAYER BOT: {layer_bot_idx}")
      print(f"INDEX OF LAYER TOP: {layer_top_idx}")

    layer_bottom_temperature = temperature.m[layer_bot_idx]
    layer_top_temperature = temperature.m[layer_top_idx]
    layer_bottom_pressure = pressure.m[layer_bot_idx]
    layer_top_pressure = pressure.m[layer_top_idx]

    if db:
      print("")
      print("USING NEAREST LAYER BOTTOM PRESSURE: %f\n" % (layer_bottom_pressure))
      print("USING NEAREST LAYER TOP PRESSURE: %f\n" % (layer_top_pressure))

    # Compute the moist adiabatic lapse rate
    try:
      MALR = mpcalc.moist_lapse(pressure[layer_bot_idx:],layer_bottom_temperature*units('degK'),reference_pressure=layer_bottom_pressure*units('hPa'))
    except ValueError:
      print("UNABLE TO COMPUTE MALR IN calc_ctp()")
      return(-9999.*units('J/kg'))

  # The MALR was only computed from the pressure at the bottom of the layer to the top of the sounding,
  # so subset the data to align with the levels where the MALR was computed
  ctp_prs = pressure[layer_bot_idx:]
  ctp_tmp = temperature[layer_bot_idx:]
  
  # Compute the difference between the environmental temperature profile and the MALR
  tdiff = (ctp_tmp-MALR)

  # Create a mask for the layer we want to integrate over
  p_mask = (ctp_prs<=pressure[layer_bot_idx])&(ctp_prs>=pressure[layer_top_idx])
  
  # Compute the Convective Triggering Potential (CTP) index
  CTP = mpconsts.Rd * units.Quantity(np.trapz(tdiff[p_mask].m,np.log(ctp_prs[p_mask].m)),'K')
  
  if plotskewt:
    import matplotlib.pyplot as plt
    from metpy.plots import SkewT
    fig = plt.figure(1, figsize=(22,15))
    skew = SkewT(fig=fig,rotation=45.0)
    skew.plot(pressure,temperature,'r',marker='.',linewidth=4)
    skew.ax.axhline(y=pressure[layer_bot_idx],xmin=-80,xmax=80,color='k',linewidth=2,linestyle='--')
    skew.ax.axhline(y=pressure[layer_top_idx],xmin=-80,xmax=80,color='k',linewidth=2,linestyle='--')
    skew.ax.fill_betweenx(pressure[layer_bot_idx:layer_top_idx+1],temperature[layer_bot_idx:layer_top_idx+1],MALR[0:(layer_top_idx-layer_bot_idx)+1])
    skew.plot(pressure[layer_bot_idx:],MALR,marker='.',linewidth=4,color='magenta')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_xlabel('Temperature (C)')
    plt.title('CTP = %5.5f J/kg' % (float(CTP.m)),loc='left')
    plt.title('STATION = %s' % (station_name))
    fig.savefig(os.path.join(plotdir,'CTP_%s.png' % (station)))
    plt.close()
   
  return CTP

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

def calc_humidity_index(pressure,temperature,dewpoint,station_index,start_pressure_hpa=-1,bot_pressure_hpa=50.0,top_pressure_hpa=150.0,interp=True,db=True):
  """ Function for computing the Humidity Index defined as:
      :math:`\\mathrm{HI_{low}} = (T_{950\\,\\text{hPa}} - D_{950\\,\\text{hPa}}) + (T_{850\\,\\text{hPa}} - D_{850\\,\\text{hPa}})`
  
  Args:
      pressure (pint.Quantity or xarray.DataArray): the pressure variable with units of Hectopascals (hPa).
      temperature (pint.Quantity or xarray.DataArray): the temperature variable with units of Kelvin (K).
      dewpoint (pint.Quantity or xarray.DataArray): the dewpoint temperature variable with units of Kelvin (K).
      station_index (int): the integer index of the station currently being processed. Use -1 if a single station is being passed.
      start_pressure_hpa (float, optional): the starting pressure to use. Default: -1 (bottom level in profile).
      bot_pressure_hpa (float, optional): bottom pressure value of the layer, subtracted from start_pressure_hpa. Default: 50 hPa.
      top_pressure_hpa (float, optional): top pressure value of the layer, subtracted from start_pressure_hpa. Default: 150 hPa.
      interp (bool): perform vertical interpolation to bot_pressure_hpa and top_pressure_hpa or use closest. Default: True.
      db (bool): Print debugging statements. Default: False

  Returns:
      float32
 
  Reference:
      Findell, K. L., and E. A. B. Eltahir, 2003: Atmospheric Controls on Soil Moisture–Boundary Layer Interactions. Part I: Framework Development. J. Hydrometeor., 4, 552–569, https://doi.org/10.1175/1525-7541(2003)004<0552:ACOSML>2.0.CO;2.
      
      Also see: https://www.pauldirmeyer.com/coupling-metrics for a summary of this and other land-atmosphere coupling metrics.

  Notes:
      Pressure and temperature can either be a 1D profile or a 3D array of data. If a 3D array, station_index needs to be a non-negative value
      indicating the single column to extract from the 3D data. If they are 1D profiles, use -1 for the station_index as that will be the only
      profile to use.

      Note also that the equation denotes 950 and 850 hPa, but the starting pressure in vertical profiles is not always 1000 hPa. Thus,
      the function is designed to focus on a consistent layer depth of 100 hPa starting at 50 hPa above the bottom pressure value and
      ending at 150 hP above the bottom pressure value rather than a 100 hPa layer spanning 950 hPa to 850 hPa.

  """

  # Import utility functions for CTP/HI 
  import metcalcpy.diagnostics.land_surface_ctphi_util as land_surface_ctphi_util

  # If the station index is a non-negative value, then extract the profile at the station index
  if station_index>=0:
    temperature=temperature.isel(sid=station_index).values*units('degK')
    pressure=pressure.isel(sid=station_index).values*units('hPa')
    dewpoint=dewpoint.isel(sid=station_index).values*units('degK')

  # Subset the profile to only levels where pressure is > min_prs_profile
  min_prs_profile = 100.0*units('hPa')
  temperature = temperature[pressure>=min_prs_profile]
  dewpoint = dewpoint[pressure>=min_prs_profile]
  pressure = pressure[pressure>=min_prs_profile]

  # A pressure difference for warning and a maximum value where the humidity index
  # won't be calculated. If warn_prs_diff is exceeded, a warning is printed but 
  # calculation proceeds. If max_prs_diff is exeeded, an error is printed
  # and missing data is returned. These are only used for the non-interpolation case.
  warn_prs_diff = 50.0*units('hPa')
  max_prs_diff = 100.0*units('hPa')

  # Find the starting pressure to use
  start_prs = land_surface_ctphi_util.find_start_pressure(start_pressure_hpa*units('hPa'),pressure,interp,max_prs_diff,db,'HI')
  if start_prs < 0:
    return(-9999.*units('J/kg'))
  start_prs = start_prs*units('hPa')

  # Based on the starting pressure, set the initial layer bottom and top pressures
  layer_bot_prs = start_prs-(bot_pressure_hpa*units('hPa'))
  layer_top_prs = start_prs-(top_pressure_hpa*units('hPa'))

  if db:
    print("")
    print(f"TARGET LAYER BOTTOM PRESSURE: {layer_bot_prs.m}")
    print(f"TARGET LAYER TOP PRESSURE: {layer_top_prs.m}\n")

  # Obtain information at the top and bottom of the layer
  if interp:

    # If the highest pressure in the sounding is < the layer_bot_prs, then skip this site
    if layer_bot_prs.m > np.max(pressure.m):
      print("ERROR! HIGHEST PRESSURE IN SOUNDING IS LOWER THAN REQUESTED BOTTOM PRESSURE.")
      print(f"max pressure: {np.max(pressure.m)}")
      print(f"requested bottom pressure: {layer_bot_prs.m}")
      print("UNABLE TO COMPUTE HI.")
      return(-9999.*units('degK'))
    
    if db:
      print("")
      print(f"INTERPOLATING TO BOTTOM PRESSURE: {layer_bot_prs.m}")
      print(f"INTERPOLATING TO TOP PRESSURE: {layer_top_prs.m}\n")

    layer_bottom_temperature, layer_bottom_dewpoint = log_interp_1d(layer_bot_prs,pressure,temperature,dewpoint)
    layer_top_temperature, layer_top_dewpoint = log_interp_1d(layer_top_prs,pressure,temperature,dewpoint)
    
    # log_interp_1d returns array-like, so get the float values
    layer_bottom_temperature = layer_bottom_temperature[0]
    layer_bottom_dewpoint = layer_bottom_dewpoint[0]
    layer_top_temperature = layer_top_temperature[0]
    layer_top_dewpoint = layer_top_dewpoint[0]

  else:

    # Find the index of the closest pressure value to the bottom and top values requested
    layer_bot_idx = np.argmin(np.abs(pressure-layer_bot_prs))
    layer_top_idx = np.argmin(np.abs(pressure-layer_top_prs))

    # Warn if the distance between the closest value to the bottom and top values exceeds warn_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=warn_prs_diff:
      print(f"WARNING! ACTUAL BOTTOM PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
      print(f"requested: layer_bot_prs = {layer_bot_prs.m} hPa")
      print(f"closest: layer_bot_prs = {pressure[layer_bot_idx].m} hPa")
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=warn_prs_diff:
      print(f"WARNING! ACTUAL TOP PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
      print(f"requested: layer_top_prs = {layer_top_prs.m} hPa")
      print(f"closest: layer_top_prs = {pressure[layer_top_idx].m} hPa")
    # Return missing data if the distance between the closest value to the bottom and top values exceeds max_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=max_prs_diff:
      print(f"ERROR! ACTUAL BOTTOM PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
      print(f"requested: layer_bot_prs = {layer_bot_prs.m} hPa")
      print(f"closest: layer_bot_prs = {pressure[layer_bot_idx].m} hPa")
      print("UNABLE TO COMPUTE HI.")
      return(-9999.*units('degK'))
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=max_prs_diff:
      print(f"ERROR! ACTUAL TOP PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
      print(f"requested: layer_top_prs = {layer_top_prs.m} hPa")
      print(f"closest: layer_top_prs = {pressure[layer_top_idx].m} hPa")
      print("UNABLE TO COMPUTE HI.")
      return(-9999.*units('degK'))
    if db:
      print("")
      print(f"USING DATA AT NEAREST BOTTOM PRESSURE: {pressure[layer_bot_idx].m}\n")
      print(f"USING DATA AT NEAREST TOP PRESSURE: {pressure[layer_top_idx].m}\n")

    layer_bottom_temperature = temperature[layer_bot_idx]
    layer_bottom_dewpoint = dewpoint[layer_bot_idx]
    layer_top_temperature = temperature[layer_top_idx]
    layer_top_dewpoint = dewpoint[layer_top_idx]

  return (layer_bottom_temperature-layer_bottom_dewpoint) + (layer_top_temperature-layer_top_dewpoint)
