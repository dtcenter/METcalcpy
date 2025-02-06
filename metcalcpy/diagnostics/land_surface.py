"""Diagnostics relevant to Land/Surface applications"""

import metpy.constants as mpconsts
import numpy as np
from metpy import calc as mpcalc
from metpy.interpolate import log_interpolate_1d as log_interp_1d
from metpy.units import units
import os
from pandas.core.series import Series
from xarray.core.dataarray import DataArray

def calc_ctp(pressure,temperature,station_index,start_pressure_hpa=-1,bot_pressure_hpa=100.0,top_pressure_hpa=300.0,interp=False,db=False,plotskewt=False,plotdir="",station_name=""):

  """ Function for computing the Convective Triggering Potential

  Args:
      pressure (pint.Quantity): the vertical pressure profile
      temperature (pint.Quantity): the vertical temperature profile
      station_index (int): the integer index of the station currently being processed. Use -1 if a single station is being passed.
      start_pressure_hpa (float, optional): the starting pressure to use. Default: -1 (bottom level in profile).
      bot_pressure_hpa (float, optional): bottom pressure value of the layer, added to start_pressure_hpa. Default: 100 hPa.
      top_pressure_hpa (float, optional): top pressure value of the layer, added to start_pressure_hpa. Default: 300 hPa.
      interp (bool): Whether to interpolate data to exact pressures or use the closest. Default: False.
      db (bool): Print debugging statements. Default: False
      plotskewt (bool): Plot a Skew-T Log-P graphic of the CTP calculation. Default: False.
      plotdir (string, optional): Directory where Skew-T plots should be written. Default: "".
      station_name (string, optional): Location ID string used for labeling the Skew-T plot and image file name. Default: "".

  Returns:
      float32
 
  Reference:
      TBD

  Notes:
      Lorem Ipsum

  """

  # If the station index is a non-negative value, then extract the profile at the station index
  if station_index>=0:
    temperature=temperature.isel(sid=station_index).values*units('degK')
    pressure=pressure.isel(sid=station_index).values*units('hPa')

  # Ensure there's no missing data values in the profile
  #print(temperature)
  #temperature = temperature[~np.isnan(temperature)]
  #print(temperature)

  # Subset the profile to only levels where pressure is > min_prs_profile
  min_prs_profile = 100.0*units('hPa')
  temperature = temperature[pressure>=min_prs_profile]
  pressure = pressure[pressure>=min_prs_profile]
  
  # Set a pressure difference for non-interpolation case
  # If the closest pressure to the bot_pressure_hpa is more
  # than this amount from the start_pressure_hpa, a warning will be
  # printed.
  max_prs_diff = 100.0*units('hPa')

  # Find the starting pressure in the profile
  if start_pressure_hpa < 0:
    start_prs = pressure[0]
    if db:
      print("")
      print("USING LOWEST STARTING PRESSURE: %f\n" % (start_prs.m))
  else:
    if interp:
      # If the requested starting pressure is greater than all the pressures in the
      # profile, then we won't be able to interpolate to the requested starting pressure.
      if start_pressure_hpa > np.max(pressure.m):
        if db:
          print("")
          print("ERROR! REQUESTED STARTING PRESSURE INVALID")
          print("UNABLE TO COMPUTE CTP.")
        return(-9999.*units('J/kg'))
      else:
        start_prs = start_pressure_hpa*units('hPa')
      if db:
        print("")
        print("USING ACTUAL REQUESTED STARTING PRESSURE: %f\n" % (start_prs.m))
    else:
      # Find the closest value. We'll just take the difference between the start pressure and pressure
      # and find the index of the minimum
      prs_diff = pressure-(start_pressure_hpa*units('hPa'))
      start_prs = pressure[np.argmin(np.abs(prs_diff))]
      if np.abs(start_pressure_hpa*units('hPa')-start_prs)>=max_prs_diff:
        print("INFO: ACTUAL STARTING PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED START PRESSURE." % (max_prs_diff.m))
        print("requested: start_pressure_hpa = %4.2f hPa" % (start_pressure_hpa))
        print("actual: start_pressure_hpa = %4.2f hPa" % (start_prs.m))
      if db:
        print("")
        print("USING NEAREST STARTING PRESSURE: %f\n" % (start_prs.m))

  # Based on the starting pressure, set the initial layer bottom and top pressures
  layer_bot_prs = start_prs-(bot_pressure_hpa*units('hPa'))
  layer_top_prs = start_prs-(top_pressure_hpa*units('hPa'))

  if db:
    print("")
    print("TARGET LAYER BOTTOM PRESSURE: %f\n" % (layer_bot_prs.m))
    print("TARGET LAYER TOP PRESSURE: %f\n" % (layer_top_prs.m))
  
  # Obtain information at the top and bottom of the layer
  if interp:
    
    #prsBot, tmpBot = log_interp_1d(layer_bot_prs.m,pressure.m,pressure.m,temperature.m)
    #prsTop, tmpTop = log_interp_1d(layer_top_prs.m,pressure.m,pressure.m,temperature.m)
    tmpBot = log_interp_1d(layer_bot_prs.m,pressure.m,temperature.m)
    tmpTop = log_interp_1d(layer_top_prs.m,pressure.m,temperature.m)
    prsBot = np.array([layer_bot_prs.m])
    prsTop = np.array([layer_top_prs.m])
    
    if db:
      print("")
      print("USING INTERPOLATED LAYER BOTTOM PRESSURE: %f\n" % (prsBot))
      print("USING INTERPOLATED LAYER TOP PRESSURE: %f\n" % (prsTop))
  
    # Find the top and bottom of the layer, where the interpolated values should be inserted
    if any(np.diff(pressure.m)[np.diff(pressure.m)>=0]):
      if db:
        print("ERROR! PRESSURES DO NOT MONOTONICALLY DECREASE!")
        print("UNABLE TO COMPUTE CTP.")
      return(-9999.*units('J/kg'))
    layer_bot_idx = len(pressure.m)-np.searchsorted(pressure.m[::-1],prsBot,side="left")[0]
    layer_top_idx = len(pressure.m)-np.searchsorted(pressure.m[::-1],prsTop,side="left")[0]
    if db:
      print("")
      print("INSERTING INTERPOLATED BOT DATA AT INDEX: %02d" % (int(layer_bot_idx)))
      print("INSERTING INTERPOLATED TOP DATA AT INDEX: %02d" % (int(layer_top_idx)))
    
    # Create a new sounding to use, which has the interpolated T/P at bottom/top inserted
    prs = np.append(np.append(np.append(np.append(pressure.m[0:layer_bot_idx],prsBot),pressure.m[layer_bot_idx:layer_top_idx]),prsTop),pressure.m[layer_top_idx:])
    tmp = np.append(np.append(np.append(np.append(temperature.m[0:layer_bot_idx],tmpBot),temperature.m[layer_bot_idx:layer_top_idx]),tmpTop),temperature.m[layer_top_idx:])
    # Assign units to the new sounding variables
    prs = prs*units('hPa')
    tmp = tmp*units('degK')

    # Reset the variables as if this was the true sounding
    pressure = prs
    temperature = tmp

    # Find the new layer top and bottom indices, which should be the indices of the interpolated values that were inserted above
    layer_bot_idx = np.where(pressure.m==prsBot)[0][0]
    layer_top_idx = np.where(pressure.m==prsTop)[0][0]
    if db:
      print("")
      print("INDEX OF LAYER BOT: %02d" % (int(layer_bot_idx)))
      print("INDEX OF LAYER TOP: %02d" % (int(layer_top_idx)))

    # Compute the moist adiabatic lapse rate
    try:
      MALR = mpcalc.moist_lapse(pressure[layer_bot_idx:],tmpBot*units('degK'),reference_pressure=prsBot*units('hPa'))
    except ValueError:
      print("UNABLE TO COMPUTE MALR IN calc_ctp()")
      return(-9999.*units('J/kg'))

  else:

    # Find the index of the closest value. 
    # We'll just take the difference between the top/bottom pressure and find the index of the minimum
    layer_bot_idx = np.argmin(np.abs(pressure-layer_bot_prs))
    layer_top_idx = np.argmin(np.abs(pressure-layer_top_prs))

    # Warn if the distance between the closest value to the bottom and top values exceeds max_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=max_prs_diff:
      print("INFO: ACTUAL BOTTOM PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED BOTTOM PRESSURE." % (max_prs_diff.m))
      print("requested: layer_bot_prs = %4.2f hPa" % (layer_bot_prs.m))
      print("actual: layer_bot_prs = %4.2f hPa" % (pressure[layer_bot_idx].m))
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=max_prs_diff:
      print("INFO: ACTUAL TOP PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED TOP PRESSURE." % (max_prs_diff.m))
      print("requested: layer_top_prs = %4.2f hPa" % (layer_top_prs.m))
      print("actual: layer_top_prs = %4.2f hPa" % (pressure[layer_top_idx].m))
    
    if db:
      print("")
      print("INDEX OF LAYER BOT: %02d" % (int(layer_bot_idx)))
      print("INDEX OF LAYER TOP: %02d" % (int(layer_top_idx)))

    prsBot = pressure.m[layer_bot_idx]
    prsTop = pressure.m[layer_top_idx]
    tmpBot = temperature.m[layer_bot_idx]
    tmpTop = temperature.m[layer_top_idx]

    if db:
      print("")
      print("USING NEAREST LAYER BOTTOM PRESSURE: %f\n" % (prsBot))
      print("USING NEAREST LAYER TOP PRESSURE: %f\n" % (prsTop))

    # Compute the moist adiabatic lapse rate
    try:
      MALR = mpcalc.moist_lapse(pressure[layer_bot_idx:],tmpBot*units('degK'),reference_pressure=prsBot*units('hPa'))
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

def calc_humidity_index(pressure,temperature,dewpoint,station_index,start_pressure_hpa=-1,bot_pressure_hpa=50.0,top_pressure_hpa=150.0,interp=False,db=False):
  """ Function for computing the Humidity Index
  
  Args:
      pressure (pint.Quantity): the vertical pressure profile
      temperature (pint.Quantity): the vertical temperature profile
      dewpoint (pint.Quantity): the vertical dewpoint temperature profile
      station_index (int): the integer index of the station currently being processed. Use -1 if a single station is being passed.
      start_pressure_hpa (float, optional): the starting pressure to use. Default: -1 (bottom level in profile).
      bot_pressure_hpa (float, optional): bottom pressure value of the layer, added to start_pressure_hpa. Default: 50 hPa.
      top_pressure_hpa (float, optional): top pressure value of the layer, added to start_pressure_hpa. Default: 150 hPa.
      interp (bool): perform vertical interpolation to bot_pressure_hpa and top_pressure_hpa or use closest. Default: False.
      db (bool): Print debugging statements. Default: False

  Returns:
      float32
 
  Reference:
      TBD

  Notes:
      Lorem Ipsum

  """

  # If the station index is a non-negative value, then extract the profile at the station index
  if station_index>=0:
    temperature=temperature.isel(sid=station_index).values*units('degK')
    pressure=pressure.isel(sid=station_index).values*units('hPa')
    dewpoint=dewpoint.isel(sid=station_index).values*units('degK')

  # Ensure there's no missing data values in the profile
  #print(temperature)
  #temperature = temperature[~np.isnan(temperature)]
  #print(temperature)

  # Subset the profile to only levels where pressure is > min_prs_profile
  min_prs_profile = 100.0*units('hPa')
  temperature = temperature[pressure>=min_prs_profile]
  dewpoint = dewpoint[pressure>=min_prs_profile]
  pressure = pressure[pressure>=min_prs_profile]

  # Set a pressure difference for non-interpolation case
  # If the closest pressure to the bot_pressure_hpa is more
  # than this amount from the start_pressure_hpa, a warning will be
  # printed.
  max_prs_diff = 100.0*units('hPa')

  # Find the starting pressure in the profile
  if start_pressure_hpa < 0:
    start_prs = pressure[0]
    if db:
      print("")
      print("USING LOWEST STARTING PRESSURE: %f\n" % (start_prs.m))
  else:
    if interp:
      # If the requested starting pressure is greater than all the pressures in the
      # profile, then we won't be able to interpolate to the requested starting pressure.
      if start_pressure_hpa > np.max(pressure.m):
        if db:
          print("")
          print("ERROR! REQUESTED STARTING PRESSURE INVALID")
          print("UNABLE TO COMPUTE CTP.")
        return(-9999.*units('J/kg'))
      else:
        start_prs = start_pressure_hpa*units('hPa')
      if db:
        print("")
        print("USING ACTUAL REQUESTED STARTING PRESSURE: %f\n" % (start_prs.m))
    else:
      # Find the closest value. We'll just take the difference between the start pressure and pressure
      # and find the index of the minimum
      prs_diff = pressure-(start_pressure_hpa*units('hPa'))
      start_prs = pressure[np.argmin(np.abs(prs_diff))]
      if np.abs(start_pressure_hpa*units('hPa')-start_prs)>=max_prs_diff:
        print("INFO: ACTUAL STARTING PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED START PRESSURE." % (max_prs_diff.m))
        print("requested: start_pressure_hpa = %4.2f hPa" % (start_pressure_hpa))
        print("actual: start_pressure_hpa = %4.2f hPa" % (start_prs.m))
      if db:
        print("")
        print("USING NEAREST STARTING PRESSURE: %f\n" % (start_prs.m))

  # Based on the starting pressure, set the initial layer bottom and top pressures
  layer_bot_prs = start_prs-(bot_pressure_hpa*units('hPa'))
  layer_top_prs = start_prs-(top_pressure_hpa*units('hPa'))

  if db:
    print("")
    print("TARGET LAYER BOTTOM PRESSURE: %f\n" % (layer_bot_prs.m))
    print("TARGET LAYER TOP PRESSURE: %f\n" % (layer_top_prs.m))

  # Obtain information at the top and bottom of the layer
  if interp:

    # If the highest pressure in the sounding is < the layer_bot_prs, then skip this site
    if layer_bot_prs.m > np.max(pressure.m):
      print("ERROR! HIGHEST PRESSURE IN SOUNDING IS LOWER THAN REQUESTED BOTTOM PRESSURE.")
      print("max pressure: %4.2f" % (np.max(pressure.m)))
      print("requested bottom pressure: %4.2f" % (layer_bot_prs.m))
      print("UNABLE TO COMPUTE HI.")
      return(-9999.*units('degK'))
    
    if db:
      print("")
      print("INTERPOLATING TO BOTTOM PRESSURE: %f\n" % (layer_bot_prs.m))
      print("INTERPOLATING TO TOP PRESSURE: %f\n" % (layer_top_prs.m))

    tmpBot, dewBot = log_interp_1d(layer_bot_prs,pressure,temperature,dewpoint)
    tmpTop, dewTop = log_interp_1d(layer_top_prs,pressure,temperature,dewpoint)
    
    # log_interp_1d returns array-like, so get the float values
    tmpBot = tmpBot[0]
    dewBot = dewBot[0]
    tmpTop = tmpTop[0]
    dewTop = dewTop[0]

  else:

    # Find the index of the closest pressure value to the bottom and top values requested
    layer_bot_idx = np.argmin(np.abs(pressure-layer_bot_prs))
    layer_top_idx = np.argmin(np.abs(pressure-layer_top_prs))

    # Warn if the distance between the closest value to the bottom and top values exceeds max_prs_diff
    if np.abs(pressure[layer_bot_idx]-layer_bot_prs)>=max_prs_diff:
      print("INFO: ACTUAL BOTTOM PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED BOTTOM PRESSURE." % (max_prs_diff.m))
      print("requested: layer_bot_prs = %4.2f hPa" % (layer_bot_prs.m))
      print("actual: layer_bot_prs = %4.2f hPa" % (pressure[layer_bot_idx].m))
    if np.abs(pressure[layer_top_idx]-layer_top_prs)>=max_prs_diff:
      print("INFO: ACTUAL TOP PRESSURE IS AT LEAST %3.2f hPa FROM REQUESTED TOP PRESSURE." % (max_prs_diff.m))
      print("requested: layer_top_prs = %4.2f hPa" % (layer_top_prs.m))
      print("actual: layer_top_prs = %4.2f hPa" % (pressure[layer_top_idx].m))
    
    if db:
      print("")
      print("USING DATA AT NEAREST BOTTOM PRESSURE: %f\n" % (pressure[layer_bot_idx].m))
      print("USING DATA AT NEAREST TOP PRESSURE: %f\n" % (pressure[layer_top_idx].m))

    tmpBot = temperature[layer_bot_idx]
    dewBot = dewpoint[layer_bot_idx]
    tmpTop = temperature[layer_top_idx]
    dewTop = dewpoint[layer_top_idx]

  return (tmpBot-dewBot) + (tmpTop-dewTop)
