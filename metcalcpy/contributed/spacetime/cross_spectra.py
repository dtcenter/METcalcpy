# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
This is an example script for cross-spectral analysis. Computes power and cross-spectra for multiple input
data sets.
"""
import argparse
import xarray as xr
import numpy as np

"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from spacetime import mjo_cross
from spacetime import get_symmasymm
from spacetime_utils import save_Spectra

"""
Set parameters for the spectra calculation.
spd:  Number of observations per day.
nperseg:  Number of data points per segment.
segOverlap:  How many data points of overlap between segments. If negative there is overlap
             If positive skip that number of values between segments.
Symmetry:    Can be "symm", "asymm" for symmetric or anti-symmetric across the equator. "latband" 
             if the spectra averaged across the entire latitude band are requested.
latMin:      Minimum latitude to include.
latMax:      Maximum latitude to include.
"""
spd = 2
nperseg = 46 * spd
segOverLap = -20 * spd
Symmetry = "symm"
latMin = -15.
latMax = 15.
datestrt = '2015-12-01'  # plot start date, format: yyyy-mm-dd
datelast = '2016-03-31'  # plot end date, format: yyyy-mm-dd

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str,
    help='data path')
parser.add_argument('--pathout', type=str,
    help='output path')
args = parser.parse_args()

# datapath = '../data/'
# path to the location where to save the output spectra
# pathout = '../data/'
datapath = args.datapath
pathout = args.pathout

print("reading data from file:")
""" 
Read in data here. Example:
"""
ds = xr.open_dataset(datapath+'precip.trmm.'+str(spd)+'x.1p0.v7a.fillmiss.comp.2014-2016.nc')
z = ds.precip
z = z.sel(lat=slice(latMin, latMax))
z = z.sel(time=slice(datestrt, datelast))
z = z.squeeze()
latC = ds.lat.sel(lat=slice(latMin, latMax))

ds = xr.open_dataset(datapath+'precip.erai.sfc.1p0.'+str(spd)+'x.2014-2016.nc')
x = ds.precip
x = x.sel(lat=slice(latMin, latMax))
x = x.sel(time=slice(datestrt, datelast))
x = x.squeeze()
latA = ds.lat.sel(lat=slice(latMin, latMax))
ds.close()

ds = xr.open_dataset(datapath+'div.erai.850.1p0.'+str(spd)+'x.2014-2016.nc')
y = ds.div
y = y.sel(lat=slice(latMin, latMax))
y = y.sel(time=slice(datestrt, datelast))
y = y.squeeze()
latB = ds.lat.sel(lat=slice(latMin, latMax))

ds = xr.open_dataset(datapath+'div.erai.200.1p0.'+str(spd)+'x.2014-2016.nc')
w = ds.div
w = y.sel(lat=slice(latMin, latMax))
w = y.sel(time=slice(datestrt, datelast))
w = y.squeeze()
latB = ds.lat.sel(lat=slice(latMin, latMax))



if any(latA - latB) != 0:
    print("Latitudes must be the same for both variables! Check latitude ordering.")

print("get symmetric/anti-symmetric components:")
if Symmetry == "symm" or Symmetry == "asymm":
    X = get_symmasymm(x, latA, Symmetry)
    Y = get_symmasymm(y, latB, Symmetry)
    Z = get_symmasymm(z, latC, Symmetry)
    W = get_symmasymm(w, latB, Symmetry)
else:
    X = x
    Y = y
    Z = z
    W = w

print("compute cross-spectrum:")
"""
The output from mjo_cross includes:
STC = [8,nfreq,nwave]
STC[0,:,:] : Power spectrum of x
STC[1,:,:] : Power spectrum of y
STC[2,:,:] : Co-spectrum of x and y
STC[3,:,:] : Quadrature-spectrum of x and y
STC[4,:,:] : Coherence-squared spectrum of x and y
STC[5,:,:] : Phase spectrum of x and y
STC[6,:,:] : Phase angle v1
STC[7,:,:] : Phase angle v2
freq 
wave
number_of_segments
dof
prob
prob_coh2
"""
result = mjo_cross(X, Y, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_ERAI_P_D850_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)

result = mjo_cross(X, W, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_ERAI_P_D200_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)

result = mjo_cross(X, Z, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_ERAI_TRMM_P_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)
