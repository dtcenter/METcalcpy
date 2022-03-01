# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Compute RMM index from input U850, U200 and OLR data. Data is averaged from 20S-20N
"""

import numpy as np
import xarray as xr
import datetime
import pandas as pd

import compute_mjo_indices as cmi
import plot_mjo_indices as pmi

# set dates to read
datestrt = '2000-01-01'
datelast = '2002-12-31'

spd = 1 # number of obs per day
time = np.arange(datestrt,datelast, dtype='datetime64[D]')
ntim = len(time)

#######################################
# read RMM EOFs from file and plot
EOF1, EOF2 = cmi.read_rmm_eofs('./data/')
pmi.plot_rmm_eofs(EOF1, EOF2, 'RMM_EOFs','png')

#######################################
# read data from file
ds = xr.open_dataset('/data/mgehne/OLR/olr.1x.7920.anom7901.nc')
olr = ds['olr'].sel(lat=slice(-15,15),time=slice(datestrt,datelast))
lon = ds['lon']
olr = olr.mean('lat')
print(olr.min(), olr.max())

ds = xr.open_dataset('/data/mgehne/ERAI/uwnd.erai.an.2p5.850.daily.anom7901.nc')
u850 = ds['uwnd'].sel(lat=slice(-15,15),time=slice(datestrt,datelast))
u850 = u850.mean('lat')
print(u850.min(), u850.max())

ds = xr.open_dataset('/data/mgehne/ERAI/uwnd.erai.an.2p5.200.daily.anom7901.nc')
u200 = ds['uwnd'].sel(lat=slice(-15,15),time=slice(datestrt,datelast))
u200 = u200.mean('lat')
print(u200.min(), u200.max())

########################################
# project data onto EOFs
PC1, PC2 = cmi.rmm(olr[0:ntim,:], u850[0:ntim,:], u200[0:ntim,:], time, spd, './data/')

print(PC1.min(), PC1.max())


########################################
# plot phase diagram
datestrt = '2002-01-01'
datelast = '2002-12-31'

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
pmi.phase_diagram('RMM',PC1,PC2,time,months,days,'RMM_comp_phase','png')

# plot PC time series
pmi.pc_time_series('RMM',PC1,PC2,time,months,days,'RMM_time_series','png')