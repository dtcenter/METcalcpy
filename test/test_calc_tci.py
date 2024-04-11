"""Tests the calc_tci() function in diagnostics/land_surface.py"""

from metcalcpy.diagnostics.land_surface import calc_tci
from xarray.testing import assert_equal
import xarray as xr
import pandas as pd
import numpy as np

__author__ = "Daniel Adriaansen (NCAR)"

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
