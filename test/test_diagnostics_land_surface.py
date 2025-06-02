"""Tests the functions in diagnostics/land_surface.py"""

import numpy as np
import pandas as pd
import pytest
import warnings
import xarray as xr
from metcalcpy.diagnostics.land_surface import calc_tci
from metcalcpy.diagnostics.land_surface import calc_ctp
from metcalcpy.diagnostics.land_surface import calc_humidity_index
from metpy.units import units
from numpy.testing import assert_almost_equal
from xarray.testing import assert_equal

import os

__author__ = "Daniel Adriaansen (NCAR)"

cwd = os.path.dirname(__file__)

def test_calc_ctp():
  """
  Test that the output of the calc_ctp function is correct.
  
  Returns 
  -------
  None.
  """

  # Open sounding data for the three test sites
  site1 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_72210.csv')
  site2 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_76225.csv')
  site3 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_76458.csv')

  # Save variables with units for testing
  s1prs = site1['pressure'].astype('float').values*units('hPa')
  s2prs = site2['pressure'].astype('float').values*units('hPa')
  s3prs = site3['pressure'].astype('float').values*units('hPa')
  s1tmp = site1['temperature'].astype('float').values*units('degK')
  s2tmp = site2['temperature'].astype('float').values*units('degK')
  s3tmp = site3['temperature'].astype('float').values*units('degK')

  # Test 1: default
  t1test = np.array([calc_ctp(s1prs,s1tmp,-1).m,\
                     calc_ctp(s2prs,s2tmp,-1).m,\
                     calc_ctp(s3prs,s3tmp,-1).m])

  # Test 2: provide a start_pressure_hpa
  t2test = np.array([calc_ctp(s1prs,s1tmp,-1,start_pressure_hpa=925.0).m,\
                     calc_ctp(s2prs,s2tmp,-1,start_pressure_hpa=925.0).m,\
                     calc_ctp(s3prs,s3tmp,-1,start_pressure_hpa=925.0).m])

  # Test 3: default, but with interp=False
  t3test = np.array([calc_ctp(s1prs,s1tmp,-1,interp=False).m,\
                     calc_ctp(s2prs,s2tmp,-1,interp=False).m,\
                     calc_ctp(s3prs,s3tmp,-1,interp=False).m])

  # Test 4: same as test 2, but with interp=False
  t4test = np.array([calc_ctp(s1prs,s1tmp,-1,start_pressure_hpa=925.0,interp=False).m,\
                     calc_ctp(s2prs,s2tmp,-1,start_pressure_hpa=925.0,interp=False).m,\
                     calc_ctp(s3prs,s3tmp,-1,start_pressure_hpa=925.0,interp=False).m])
 
  # Truth values
  # Ordered by [site1,site2,site3]
  t1truth = np.array([2.45332902,247.83157506,65.74518711])
  t2truth = np.array([8.04560060e+01,-9.99900000e+03,-3.60593336e+00])
  t3truth = np.array([5.45993016,363.42711415,51.12524706])
  t4truth = np.array([130.5569771,363.42711415,-17.54853487])

  print("CALC CTP TEST VALUES:")
  print(t1test)
  print(t2test)
  print(t3test)
  print(t4test)
  
  # Validate test 1
  assert_almost_equal(t1test,t1truth,decimal=0)

  # Validate test 2
  assert_almost_equal(t2test,t2truth,decimal=0)

  # Validate test 3
  assert_almost_equal(t3test,t3truth,decimal=0)
  
  # Validate test 4
  assert_almost_equal(t4test,t4truth,decimal=0)

def test_calc_humidity_index():
  """
  Test that the output of the calc_humidity_index function is correct.
  
  Returns
  -------
  None.
  """

  # Open sounding data for the three test sites
  site1 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_72210.csv')
  site2 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_76225.csv')
  site3 = pd.read_csv(f'{cwd}/data/2023031512_GDAS_Sounding_76458.csv')

  # Save variables with units for testing
  s1prs = site1['pressure'].astype('float').values*units('hPa')
  s2prs = site2['pressure'].astype('float').values*units('hPa')
  s3prs = site3['pressure'].astype('float').values*units('hPa')
  s1tmp = site1['temperature'].astype('float').values*units('degK')
  s2tmp = site2['temperature'].astype('float').values*units('degK')
  s3tmp = site3['temperature'].astype('float').values*units('degK')
  s1dew = site1['dewpoint'].astype('float').values*units('degK')
  s2dew = site2['dewpoint'].astype('float').values*units('degK')
  s3dew = site3['dewpoint'].astype('float').values*units('degK')

  # Test 1: default
  t1test = np.array([calc_humidity_index(s1prs,s1tmp,s1dew,-1).m,\
                     calc_humidity_index(s2prs,s2tmp,s2dew,-1).m,\
                     calc_humidity_index(s3prs,s3tmp,s3dew,-1).m])
  
  # Test 2: provide a start_pressure_hpa
  t2test = np.array([calc_humidity_index(s1prs,s1tmp,s1dew,-1,start_pressure_hpa=925.0).m,\
                     calc_humidity_index(s2prs,s2tmp,s2dew,-1,start_pressure_hpa=925.0).m,\
                     calc_humidity_index(s3prs,s3tmp,s3dew,-1,start_pressure_hpa=925.0).m])

  # Test 3: default, but with interp=False
  t3test = np.array([calc_humidity_index(s1prs,s1tmp,s1dew,-1,interp=False).m,\
                     calc_humidity_index(s2prs,s2tmp,s2dew,-1,interp=False).m,\
                     calc_humidity_index(s3prs,s3tmp,s3dew,-1,interp=False).m])

  # Test 4: same as test 2, but with interp=False
  t4test = np.array([calc_humidity_index(s1prs,s1tmp,s1dew,-1,start_pressure_hpa=925.0,interp=False).m,\
                     calc_humidity_index(s2prs,s2tmp,s2dew,-1,start_pressure_hpa=925.0,interp=False).m,\
                     calc_humidity_index(s3prs,s3tmp,s3dew,-1,start_pressure_hpa=925.0,interp=False).m])

  # Truth values
  # Ordered by [site1,site2,site3]
  t1truth = np.array([30.356453,48.6300781,12.8117684])
  t2truth = np.array([20.99006982,-9999.,30.89811178])
  t3truth = np.array([27.0415954,48.230834,9.7646789])
  t4truth = np.array([24.027771,48.23083496,27.01660156])

  # Validate test 1
  assert_almost_equal(t1test,t1truth,decimal=5)
  
  # Validate test 2
  assert_almost_equal(t2test,t2truth,decimal=5)
  
  # Validate test 3
  assert_almost_equal(t3test,t3truth,decimal=5)
  
  # Validate test 4
  assert_almost_equal(t4test,t4truth,decimal=5)

@pytest.mark.filterwarnings("ignore:Degrees of freedom")
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
    xr_input = xr.open_dataset(f'{cwd}/data/calc_tci_jja_xarray_input.nc')

    # Output data for Xarray case
    xr_truth_var = '__xarray_dataarray_variable__'
    xr_truth = xr.open_dataset(f'{cwd}/data/calc_tci_jja_xarray_output.nc')
 
    # Compute TCI
    xr_test = calc_tci(xr_input['SOILWATER_10CM'],xr_input['LHFLX'])

    # Validate Xarray case
    assert_equal(xr_truth[xr_truth_var],xr_test)

  if doPandas:
    ###### Pandas DataFrame case
    # Input data for Pandas case
    pd_input = pd.read_csv(f'{cwd}/data/calc_tci_jja_pandas_input.csv')
 
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
