# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Compute OMI index from input OLR data. 
"""

import numpy as np
import xarray as xr
import datetime
import pandas as pd

import compute_mjo_indices as cmi
from plot_mjo_indices import phase_diagram

# set dates to read
datestrt = '1979-01-01'
datelast = '2012-12-31'

spd = 1 # number of obs per day
time = np.arange(datestrt,datelast, dtype='datetime64[D]')
ntim = len(time)

# read OLR from file
ds = xr.open_dataset('/data/mgehne/OLR/olr.1x.7920.nc')
olr = ds['olr'].sel(lat=slice(-20,20),time=slice(datestrt,datelast))
lat = ds['lat'].sel(lat=slice(-20,20))
lon = ds['lon']
print(olr.min(), olr.max())

# project OLR onto EOFs
PC1, PC2 = cmi.omi(olr[0:ntim,:,:], time, spd, './data/')

print(PC1.min(), PC1.max())

# set dates to plot
datestrt = '2012-01-01'
datelast = '2012-03-31'

time = np.arange(datestrt,datelast, dtype='datetime64[D]')
ntim = len(time)
PC1 = PC1.sel(time=slice(datestrt,datelast))
PC2 = PC2.sel(time=slice(datestrt,datelast))
PC1 = PC1[0:ntim]
PC2 = PC2[0:ntim]

months = []
days = []
for idx, val in enumerate(time):
    date = pd.to_datetime(val).timetuple()
    month = date.tm_mon
    day = date.tm_mday
    months.append(month)
    days.append(day)

# plot the PC phase diagram 
phase_diagram('OMI',PC1,PC2,time,months,days,'OMI_comp_phase','png')