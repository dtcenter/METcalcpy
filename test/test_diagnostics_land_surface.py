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

__author__ = "Daniel Adriaansen (NCAR)"

def test_calc_ctp():
  """
  Test that the output of the calc_ctp function is correct.
  
  Returns 
  -------
  None.
  """

  # Open sounding data for the three test sites
  site1 = pd.read_csv('data/2023031512_GDAS_Sounding_72210.csv')
  site2 = pd.read_csv('data/2023031512_GDAS_Sounding_76225.csv')
  site3 = pd.read_csv('data/2023031512_GDAS_Sounding_76458.csv')

  # Save variables with units for testing
  s1prs = site1['pressure'].astype('float').values*units('hPa')
  s2prs = site2['pressure'].astype('float').values*units('hPa')
  s3prs = site3['pressure'].astype('float').values*units('hPa')
  s1tmp = site1['temperature'].astype('float').values*units('degK')
  s2tmp = site2['temperature'].astype('float').values*units('degK')
  s3tmp = site3['temperature'].astype('float').values*units('degK')

  # Test 1: default
  t1test = np.array([calc_ctp(s1prs,s1tmp).m,\
                     calc_ctp(s2prs,s2tmp).m,\
                     calc_ctp(s3prs,s3tmp).m])

  # Test 2: provide a start_pressure_hpa
  t2test = np.array([calc_ctp(s1prs,s1tmp,start_pressure_hpa=925.0).m,\
                     calc_ctp(s2prs,s2tmp,start_pressure_hpa=925.0).m,\
                     calc_ctp(s3prs,s3tmp,start_pressure_hpa=925.0).m])

  # Test 3: Default, but with interp=True
  t3test = np.array([calc_ctp(s1prs,s1tmp,interp=True).m,\
                     calc_ctp(s2prs,s2tmp,interp=True).m,\
                     calc_ctp(s3prs,s3tmp,interp=True).m])

  # Test 4: Same as test 2, but with interp=True
  t4test = np.array([calc_ctp(s1prs,s1tmp,start_pressure_hpa=925.0,interp=True).m,\
                     calc_ctp(s2prs,s2tmp,start_pressure_hpa=925.0,interp=True).m,\
                     calc_ctp(s3prs,s3tmp,start_pressure_hpa=925.0,interp=True).m])
 
  # Truth values
  # Ordered by [site1,site2,site3]
  t1truth = np.array([5.55298893,363.56359537,51.23184928])
  t2truth = np.array([130.74650626,363.56359537,-17.48742726])
  t3truth = np.array([4.20684335,239.91821201,65.44568564])
  t4truth = np.array([7.56820228e+01,-9.99900000e+03,-4.07698660e+00])
  
  # Validate test 1
  assert_almost_equal(t1test,t1truth,decimal=5)

  # Validate test 2
  assert_almost_equal(t2test,t2truth,decimal=5)

  # Validate test 3
  assert_almost_equal(t3test,t3truth,decimal=5)
  
  # Validate test 4
  assert_almost_equal(t4test,t4truth,decimal=5)

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
  #test_calc_tci()
  test_calc_ctp()
  #test_calc_humidity_index()
