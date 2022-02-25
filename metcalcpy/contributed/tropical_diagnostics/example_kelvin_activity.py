# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
import numpy as np
import xarray as xr
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
import ccew_activity as ccew

"""
Paths to plot and data directories. The annual EOF files for the waves are provided. The eofpath should point 
to the location of these files. The filenames need to match what is being read in in ccew_activity.waveact
"""
# plotpath = '../plots/'
# eofpath = '../data/EOF/'
# datapath = '../data/'
plotpath = '/Users/fillmore/working/'
eofpath = '/Volumes/d1/fillmore/Data/TropicalDiagnostics/'
datapath = '/Volumes/d1/fillmore/Data/TropicalDiagnostics/'
"""
Parameters to set for plotting Kelvin activity index.
"""
wave = 'Kelvin'
datestrt = '2015-12-01 00:00:00'
datelast = '2016-03-31 13:00:00'


print("reading ERAI data from file:")
# spd = 1
spd = 2
ds = xr.open_dataset(datapath+'/precip.erai.sfc.1p0.'+str(spd)+'x.2014-2016.nc')
A = ds.precip
print("extracting time period:")
A = A.sel(time=slice(datestrt, datelast))
A = A.squeeze()
timeA = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
A = A * 1000/4
A.attrs['units'] = 'mm/d'

print("project data onto wave EOFs")
waveactA = ccew.waveact(A, wave, eofpath, spd, '1p0', 180, 'annual')
print(waveactA.min(), waveactA.max())


print("reading observed precipitation data from file:")
# spd = 1
spd = 2
ds = xr.open_dataset(datapath+'/precip.trmm.'+str(spd)+'x.1p0.v7a.fillmiss.comp.2014-2016.nc')
B = ds.precip
print("extracting time period:")
B = B.sel(time=slice(datestrt, datelast))
B = B.squeeze()
timeB = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
B.attrs['units'] = 'mm/d'

print("project data onto wave EOFs")
waveactB = ccew.waveact(B, wave, eofpath, spd, '1p0', 180, 'annual')
print(waveactB.min(), waveactB.max())

exps = [0, 1]
explabels = ['trmm', 'erai']
nexps = len(exps)

print("computing skill")
skill = ccew.wave_skill(B)


##### maybe this next routine needs to be moved to METplotpy? ############
# ccew.plot_skill(skill, wave, explabels, plotpath)
