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

