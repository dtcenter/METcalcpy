import numpy as np

def find_start_pressure(user_start_pressure,pressure_profile,interp_bool,max_prs_diff,debug,metric):
  """ Function for finding the starting pressure to use for the Convective Triggering Potential (CTP)
      and Humidity Index (HI) calculations.

  Args:
      user_start_pressure (pint.Quantity): the starting pressure requested by the user. A negative value indicates use the lowest pressure. Units are hPa.
      pressure_profile (pint.Quantity): the pressure profile to use with units of Hectopascals (hPa).
      interp_bool (bool): whether the user has requested interpolation or not
      max_prs_diff (pint.Quantity): the maximum pressure difference between the user requested start pressure and the closest pressure in the profile.
      debug (bool): print debugging messages or not.
      metric (string): the metric that is being calculated when this function is called

  Returns:
      pint.Quantity

  Notes:

  """

  if user_start_pressure.m < 0:
    if debug:
      print("")
      print(f"USING LOWEST STARTING PRESSURE: {pressure_profile[0].m}")
    return(pressure_profile[0].m)
  else:
    if interp_bool:
      # If the requested starting pressure is greater than all the pressures in the
      # profile, then we won't be able to interpolate to the user_start_pressure.
      if user_start_pressure.m > np.max(pressure_profile.m):
        print("")
        print("ERROR! REQUESTED STARTING PRESSURE INVALID.")
        print(f"UNABLE TO COMPUTE {metric}.")
        return(-1)
      else:
        return(user_start_pressure.m)
    else:
      # Find the closest value if no interpolation is requested. Simply take the difference
      # between the start pressure and pressure profile and find the index of the minimum difference.
      prs_diff = pressure_profile-(user_start_pressure)
      closest_pressure = pressure_profile[np.argmin(np.abs(prs_diff))]
      if np.abs(user_start_pressure-closest_pressure)>=max_prs_diff:
        print(f"WARNING: ACTUAL STARTING PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED START PRESSURE.")
        print(f"requested: start_pressure_hpa = {start_pressure_hpa} hPa")
        print(f"closest: start_pressure_hpa = {start_prs.m} hPa")
      if debug:
        print("")
        print(f"USING NEAREST STARTING PRESSURE: {closest_pressure.m}\n")
      return(closest_pressure.m)

def find_closest_bottom_top_index(pressure,layer_bottom_pressure,layer_top_pressure,warn_prs_diff,max_prs_diff,debug,metric):
  """ Function for finding the closest index to the layer bottom pressure and layer top pressure to use for the
      Convective Triggering Potential (CTP) and Humidity Index (HI) calculations. This function is only used in the non-interpolation
      case.

  Args:
      pressure (pint.Quantity): the pressure profile to use with units of Hectopascals (hPa).
      layer_bottom_pressure (pint.Quantity): the pressure value of the layer bottom with units of Hectopascals (hPa)
      layer_top_pressure (pint.Quantity): the pressure value of the layer top pressure with units of Hectopascals (hPa)
      warn_prs_diff (pint.Quantity): the maximum pressure difference between either the layer bottom or top pressure that \
      the closest pressure can be before a warning is printed.
      max_prs_diff (pint.Quantity): the maximum pressure difference between either the layer bottom or top pressure that \
      the closest pressure can be before the calculation is blocked.
      debug (bool): whether debugging is desired or not.
      metric (string): the metric that is being calculated when this function is called

  Returns:
      int

  Notes:

  """

  # Find the index of the closest value to the layer top and bottom pressures.
  # Simply take the difference between the top and bottom pressure and the pressure profile
  # and find the index of the minimum of each.
  layer_bottom_index = np.argmin(np.abs(pressure-layer_bottom_pressure))
  layer_top_index = np.argmin(np.abs(pressure-layer_top_pressure))
  
  # Warn if the distance between the closest value to the bottom and top values exceeds warn_prs_diff
  if np.abs(pressure[layer_bottom_index]-layer_bottom_pressure)>=warn_prs_diff:
    print(f"WARNING! ACTUAL BOTTOM PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
    print(f"requested: layer_bottom_pressure = {layer_bottom_pressure.m} hPa")
    print(f"closest: layer_bottom_pressure = {pressure[layer_bottom_index].m} hPa")
  if np.abs(pressure[layer_top_index]-layer_top_pressure)>=warn_prs_diff:
    print(f"WARNING! ACTUAL TOP PRESSURE IS AT LEAST {warn_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
    print(f"requested: layer_top_pressure = {layer_top_pressure.m} hPa")
    print(f"closest: layer_top_pressure = {pressure[layer_top_index].m} hPa")
  # Return missing data if the distance between the closest value to the bottom and top values exceeds max_prs_diff
  if np.abs(pressure[layer_bottom_index]-layer_bottom_pressure)>=max_prs_diff:
    print(f"ERROR! ACTUAL BOTTOM PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED BOTTOM PRESSURE.")
    print(f"requested: layer_bottom_pressure = {layer_bottom_pressure.m} hPa")
    print(f"closest: layer_bottom_pressure = {pressure[layer_bottom_index].m} hPa")
    print(f"UNABLE TO COMPUTE {metric}.")
    return -1, -1
  if np.abs(pressure[layer_top_index]-layer_top_pressure)>=max_prs_diff:
    print(f"ERROR! ACTUAL TOP PRESSURE IS AT LEAST {max_prs_diff.m} hPa FROM REQUESTED TOP PRESSURE.")
    print(f"requested: layer_top_pressure = {layer_top_pressure.m} hPa")
    print(f"closest: layer_top_pressure = {pressure[layer_top_index].m} hPa")
    print(f"UNABLE TO COMPUTE {metric}.")
    return -1, -1
  if debug:
    print("")
    print(f"INDEX OF LAYER BOTTOM: {layer_bottom_index}")
    print(f"USING DATA AT NEAREST BOTTOM PRESSURE: {pressure[layer_bottom_index].m}\n")
    print(f"INDEX OF LAYER TOP: {layer_top_index}")
    print(f"USING DATA AT NEAREST TOP PRESSURE: {pressure[layer_top_index].m}\n")

  return layer_bottom_index, layer_top_index
