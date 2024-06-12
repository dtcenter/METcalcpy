"""Tests the functions in diagnostics/land_surface.py"""

import numpy as np
import pandas as pd
import xarray as xr
from metcalcpy.diagnostics.land_surface import calc_tci
from xarray.testing import assert_equal

__author__ = "Daniel Adriaansen (NCAR)"

def test_calc_ctp():
  """
  Test that the output of the calc_ctp function is correct.
  
  Returns 
  -------
  None.
  """

  # Open sounding data for the three test sites
  site1 = pd.read_csv('2023031512_GDAS_Sounding_72210.csv')
  site2 = pd.read_csv('2023031512_GDAS_Sounding_76225.csv')
  site3 = pd.read_csv('2023031512_GDAS_Sounding_76458.csv')

  # 1. Open up CSV file of sounding with pressure/temperature data
  # 2. Test defaults
  # 3. Test with start_pressure_hpa provided
  # 4. Test 2 with interp=True
  # 4. Test 3 with interp=True

  # DEFAULT
  # 72210 --> 5.552989
  # 76225 --> 363.563595
  # 76458 --> 51.231849 

  # TEST2 (used start_pressure_hpa=925.0), rest default
  # 72210 --> 130.746506
  # 76225 --> 363.563595
  # 76458 --> -17.487427

  # TEST3 (DEFAULT, but interp=True)
  # 72210 --> 4.206843
  # 76225 --> 239.918212
  # 76458 --> 65.445686

  # TEST4 (TEST2, but interp=True)  
  # 72210 --> 75.682023
  # 76225 --> -9999. --> the bottom of this sounding is 861 hPa so interp fails.
  # 76458 --> -4.076987

def test_calc_humidity_index():
  """
  Test that the output of the calc_humidity_index function is correct.
  
  Returns
  -------
  None.
  """ 

  # 1. Open up CSV file of sounding with pressure/temperature/dewpoint data
  #    Can be the same sounding used for test_calc_ctp()
  # 2. Test defaults
  # 3. Test 2 with interp=True
  
  # DEFAULT
  # 72210 --> 32.083832
  # 76225 --> -9999. (lowest pressure > 950).
  # 76458 --> 9.759857 

  # INTERP=TRUE
  # 72210 --> 30.895859
  # 76225 --> NaN!
  # 76458 --> 11.099218


def test_calc_tci():
  """
  Test that the output of the calc_tci function is correct.

  Returns
  -------
  None.
  """

  doXarray = True
  doPandas = True

  if doXarray:
    ###### Xarray DataArray case
    # Input data for Xarray case
    xr_input = xr.open_dataset('data/calc_tci_jja_xarray_input.nc')

    # Output data for Xarray case
    xr_truth_var = '__xarray_dataarray_variable__'
    xr_truth = xr.open_dataset('data/calc_tci_jja_xarray_output.nc')
 
    # Compute TCI
    xr_test = calc_tci(xr_input['SOILWATER_10CM'],xr_input['LHFLX'])

    # Validate Xarray case
    assert_equal(xr_truth[xr_truth_var],xr_test)

  if doPandas:
    ###### Pandas DataFrame case
    # Input data for Pandas case
    pd_input = pd.read_csv('data/calc_tci_jja_pandas_input.csv')
 
    # There are three sites in the test data, each should have its own TCI value
    pd_test = np.array([])
    for name,site in pd_input.groupby('station_id'):
      pd_test = np.append(pd_test,calc_tci(site['SWC_F_MDS_1'],site['LE_F_MDS']))

    # The truth values
    pd_truth = np.array([-1.851168960504201,11.861239905560712,-2.0781980819945076])

    # Validate Pandas case
    for test,truth in tuple(zip(pd_test,pd_truth)):
      assert test==truth
    
if __name__ == "__main__":
  test_calc_tci()
  test_calc_ctp()
  test_calc_humidity_index()
