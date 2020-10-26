import xarray as xr
import numpy as np
import sys

sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from spacetime import mjo_cross
from spacetime import get_symmasymm
from spacetime import save_Spectra

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
spd = 1
nperseg = 46 * spd
segOverLap = -20 * spd
Symmetry = "symm"
latMin = -15.
latMax = 15.
datestrt = '2015-12-01'  # plot start date, format: yyyy-mm-dd
datelast = '2016-03-31'  # plot end date, format: yyyy-mm-dd

print("reading data from file:")
""" 
Read in data here. Example:
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
x = ds.u
x = x.sel(lat=slice(latMin,latMax)) 
x = x.squeeze()
"""
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p0.1x.1979-2016.nc')
x = ds.precip
x = x.sel(lat=slice(latMin, latMax))
x = x.sel(time=slice(datestrt, datelast))
x = x.squeeze()
latA = ds.lat.sel(lat=slice(latMin, latMax))
ds.close()

ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/div.erai.850.1p0.1x.1979-2016.nc')
y = ds.div
y = y.sel(lat=slice(latMin, latMax))
y = y.sel(time=slice(datestrt, datelast))
y = y.squeeze()
latB = ds.lat.sel(lat=slice(latMin, latMax))

ds = xr.open_dataset('/data/mgehne/Precip/MetricsObs/precip.trmm.1x.1p0.v7a.fillmiss.comp.1998-2016.nc')
z = ds.precip
z = z.sel(lat=slice(latMin, latMax))
z = z.sel(time=slice(datestrt, datelast))
z = z.squeeze()
latC = ds.lat.sel(lat=slice(latMin, latMax))

if any(latA - latB) != 0:
    print("Latitudes must be the same for both variables! Check latitude ordering.")

print("get symmetric/anti-symmetric components:")
if Symmetry == "symm" or Symmetry == "asymm":
    X = get_symmasymm(x, latA, Symmetry)
    Y = get_symmasymm(y, latB, Symmetry)
    Z = get_symmasymm(z, latC, Symmetry)
else:
    X = x
    Y = y
    Z = z

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
pathout = '../data/'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)

result = mjo_cross(X, Z, nperseg, segOverLap)
STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
freq = result['freq']
freq = freq * spd
wnum = result['wave']
# save spectra in netcdf file
fileout = 'SpaceTimeSpectra_ERAI_TRMM_P_' + Symmetry + '_' + str(spd) + 'spd'
pathout = '../data/'
print('saving spectra to file: ' + pathout + fileout + '.nc')
save_Spectra(STC, freq, wnum, fileout, pathout)


ds = xr.open_dataset('/data/mgehne/Precip/MetricsObs/precip.trmm.1x.1deg.lats180.v7a.fillmiss.comp.1998-201806.nc')
z = ds.precip
z = z.sel(lat=slice(latMin, latMax))
z = z.sel(time=slice(datestrt, datelast))
z = z.squeeze()
latC = ds.lat.sel(lat=slice(latMin, latMax))


print("compute cross-spectra at all lead times")
spd = 1
res1 = 'C128'
path1 = '/data/mgehne/FV3/replay_exps/C128/ERAI_free-forecast_C128/STREAM_2015103100/MODEL_DATA/SST_INITANOMALY2CLIMO-90DY/ALLDAYS/'
filebaseP = 'prcp_avg6h_fhr'  #720_C128_180x360.nc
filebaseD = 'div850_sh_cdf_f'  #00.nc'

fchrs = np.array([0,24])
print(fchrs)
nfchr = len(fchrs)

for ff in fchrs:
    fstr = f"{ff:02d}"
    print('Reading fhr='+fstr)
    ds = xr.open_dataset(path1 + filebaseP + fstr + '_C128_180x360.nc')
    prcp = ds.prcp
    prcp = prcp.sel(time=slice(datestrt, datelast))
    prcp = prcp.sel(lat=slice(latMin, latMax))
    prcp = prcp*3600
    prcp.attrs['units'] = 'mm/d'
    lat = ds.lat.sel(lat=slice(latMin, latMax))
    ds.close()
    ds = xr.open_dataset(path1 + filebaseD + fstr + '.nc')
    div = ds.div
    div = div.sel(time=slice(datestrt, datelast))
    div = div.sel(lat=slice(latMin, latMax))
    ds.close()

    print("get symmetric/anti-symmetric components:")
    if Symmetry == "symm" or Symmetry == "asymm":
        P = get_symmasymm(prcp, lat, Symmetry)
        D = get_symmasymm(div, lat, Symmetry)
        X = get_symmasymm(z, latC, Symmetry)
    else:
        P = prcp
        D = div
        X = z

    print('compute cross spectra model precip and model div')
    result = mjo_cross(P, D, nperseg, segOverLap)
    STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
    freq = result['freq']
    freq = freq * spd
    wnum = result['wave']
    # save spectra in netcdf file
    fileout = 'SpaceTimeSpectra_FV3_P_D850_' + Symmetry + '_' + str(spd) + 'spd_fhr'+ fstr
    pathout = '../data/'
    print('saving spectra to file: ' + pathout + fileout + '.nc')
    save_Spectra(STC, freq, wnum, fileout, pathout)

    print('compute cross spectra model precip and obs precip')
    print(P.shape, X.shape)
    result = mjo_cross(P, X, nperseg, segOverLap)
    STC = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
    freq = result['freq']
    freq = freq * spd
    wnum = result['wave']
    # save spectra in netcdf file
    fileout = 'SpaceTimeSpectra_FV3_TRMM_P_' + Symmetry + '_' + str(spd) + 'spd_fhr'+ fstr
    pathout = '../data/'
    print('saving spectra to file: ' + pathout + fileout + '.nc')
    save_Spectra(STC, freq, wnum, fileout, pathout)
