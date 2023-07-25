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
import os
import xarray as xr
import numpy as np

"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from spacetime import mjo_cross
from spacetime import get_symmasymm
from spacetime_utils import save_Spectra

import metcalcpy.util.read_env_vars_in_config as readconfig


# Read in the YAML config file
# user can use their own, if none specified at the command line,
# use the "default" example YAML config file, spectra_plot_coh2.py
# Using a custom YAML reader so we can use environment variables
cross_config_file = os.getenv("COMP_SPECTRA_YAML_CONFIG_NAME","spectra_comp.yaml")

config_dict = readconfig.parse_config(cross_config_file)

# Retrieve settings from config file
#pathdata is now set in the METplus conf file
pathout = config_dict['pathout'][0]
print("Output path ",pathout)
model = config_dict['model']
print("Model ",model)

# Make output directory if it does not exist
if not os.path.exists(pathout):
    os.makedirs(pathout)

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
#spd = 4
#nperseg = 15 * spd
#segOverLap = -5 * spd
#Symmetry = "symm"
#latMin = -15.
#latMax = 15.
#datestrt = '2014-04-01T06:00:00'  # plot start date, format: yyyy-mm-dd
#datelast = '2014-05-04T18:00:00'  # plot end date, format: yyyy-mm-dd
spd = config_dict['spd']
nperseg = config_dict['nperseg']*spd
segOverLap = config_dict['segOverLap']*spd
Symmetry = config_dict['Symmetry']
latMin = config_dict['latMin']
latMax = config_dict['latMax']
datestrt = config_dict['datestrt']
datelast = config_dict['datelast']
print('spd: ', spd,', nperseg: ',nperseg,', segOverLap: ',segOverLap)
print('symmetry: ',Symmetry,', latMin: ',latMin,', latMax: ',latMax)
print('datestrt: ',datestrt,', datelast: ',datelast)

print("reading data from file:")
""" 
Read in data here. Example:
"""
filenames = os.environ.get("COMP_SPECTRA_INPUT_FILE_NAMES","P_verif,P_model,Vlev1_model,Vlev2_model").split(",")
print("Filename ",filenames[1])
ds = xr.open_dataset(filenames[0])
z = ds.precip
z = z.sel(lat=slice(latMin, latMax))
z = z.sel(time=slice(datestrt, datelast))
z = z.squeeze()
latC = ds.lat.sel(lat=slice(latMin, latMax))

ds = xr.open_dataset(filenames[1])
x = ds.prate
x = x.sel(lat=slice(latMin, latMax))
x = x.sel(time=slice(datestrt, datelast))
x = x.squeeze()
latA = ds.lat.sel(lat=slice(latMin, latMax))
ds.close()

ds = xr.open_dataset(filenames[2])
ds = ds.rename({'latitude':'lat'})
ds = ds.sortby('lat',ascending=True)
y = ds.u
y = y.sel(lat=slice(latMin, latMax))
y = y.sel(time=slice(datestrt, datelast))
y = y.squeeze()
latB = ds.lat.sel(lat=slice(latMin, latMax))

ds = xr.open_dataset(filenames[3])
ds = ds.rename({'latitude':'lat'})
ds = ds.sortby('lat',ascending=True)
w = ds.u
w = w.sel(lat=slice(latMin, latMax))
w = w.sel(time=slice(datestrt, datelast))
w = w.squeeze()
latB = ds.lat.sel(lat=slice(latMin, latMax))

print(x.shape)
print(y.shape)
print(z.shape)

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
fileout = 'SpaceTimeSpectra_'+model+'_P_D850_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)

result = mjo_cross(X, W, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_'+model+'_P_D200_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)

spd = 2
result = mjo_cross(X.sel(time=Z.time), Z, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_ERAI_'+model+'_P_' + Symmetry + '_' + str(spd) + 'spd'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)
