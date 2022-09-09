# Importing required libraries
import numpy as np
import xarray as xr
import copy
import sys
import pandas as pd

def calc_tau_MJO(flxa,flx_eof,filt1,filt2):

     # # reshape the data
     flxa_rg = flxa.stack(grid=("lat","lon")).transpose("time","grid").astype('float32')

     # find indices where the data is not missing and subset the uflxi
     indx = np.squeeze(np.argwhere(~np.isnan(flxa_rg[10,:].data)))
     flx = flxa_rg[:,indx]


     print('Calculating 30-90-day band pass filtered daily UFLX anomalies')

     flxi = copy.deepcopy(flx)
     for i in range(filt1.shape[1]):

        LowPass = np.convolve(flx[:,i],filt1[:,i],mode='same')
        HighPass = flx[:,i] - LowPass
        flxi[:,i] = np.convolve(HighPass,filt2[:,i],mode='same')

     flx_bpf = copy.deepcopy(flxa_rg)	
     flx_bpf[:,indx] = flxi
     flx_bpf = flx_bpf.unstack()
     flx_bpf = flx_bpf.rename('flx_bpf')

     flx_eof_rg = flx_eof.stack(grid=("lat","lon")).transpose("time","grid").astype('float32')
     un = flx_eof_rg[:,indx]

     ## Projecting EOFs onto PCs of FLX 30-90 day anomalies

     pct = (un.data@flxi.T.data)/flxi.T.shape[0]
     pc = np.transpose(pct)
     xx = np.std(pc,axis=0,ddof=1)

     flxMJO = np.zeros(flx.T.shape)

     for j in range(4):
      
        r = flxi.T.data@pc[:,j]
        r = r/xx[j]
    
        tt = np.expand_dims(r,1) * np.expand_dims(pct[j,:],0)
    
        flxMJO = flxMJO + tt
    
     flx_MJO = copy.deepcopy(flxa_rg.T)
     flx_MJO[indx,:] = flxMJO

     # Reshape to (time,lat,lon)
     flx_MJO = flx_MJO.unstack()

     flx_MJO = flx_MJO.rename('flx_mjo')

     return flx_MJO

def calc_wpower_MJO(u,v,uflx_mjo,vflx_mjo):

     lat_ws = u.lat
     lon_ws = u.lon

     print('calculating the meridional structure of Kelvin wave')
     import math

     beta=2.28E-11
     hequiv=90
     ynd=lat_ws[len(lat_ws)-1]
     ysd=lat_ws[0]

     cwave=np.sqrt(9.81E-2*hequiv)

     eleq = np.sqrt(cwave/beta)/111000
     yn = ynd/eleq
     ys = ysd/eleq

     dy = (yn-ys)/len(lat_ws)
     dyinv = 2/dy
     enorm = np.sqrt(np.pi)
     anorm = (enorm*(math.erf(yn)+math.erf(-ys)))**(-0.5)

     yy = np.zeros(len(lat_ws))

     for j in range(len(lat_ws)):
         yy[j] = ys+((j+1)-0.5)*dy

     jnot = np.argmin(abs(yy))

     jnot1 = jnot+1

     phik = np.zeros(len(lat_ws))
     phik[jnot] = anorm*np.exp(-0.5*yy[jnot]*yy[jnot])

     for j in range(jnot1,len(lat_ws)):
         phik[j] = phik[j-1]*(dyinv-yy[j-1])/(dyinv+yy[j])

     for j in range(jnot+1):
         k=jnot-j
         phik[k]=phik[k+1]*(dyinv+yy[k+1])/(dyinv-yy[k])


     phik2d = np.ones((len(lat_ws),len(lon_ws)))
     phik2d = phik2d * np.expand_dims(phik,1)

     print('Projecting the ocean current anomalies onto the meridional structure of Kelvin wave')
     u_day_prim_kelvin = u*phik2d
     v_day_prim_kelvin = v*phik2d

     # print('calculating the MJO wind power')
     wmjoks = u_day_prim_kelvin*uflx_mjo.data + v_day_prim_kelvin*vflx_mjo.data
     wmjoks = wmjoks.rename('wmjoks')

     return wmjoks

def make_maki(sst,wmjoks,meofs):

    nday = sst.time.shape[0]
    lon = sst.lon
    nlon = len(lon)

    print('Normalizing the SST and MJO wind power data')
    sst_std = sst.std(dim='time',skipna=True)
    sst_norm = sst/sst_std

    wmjoks_std = wmjoks.std(dim='time',skipna=True)
    wmjoks_norm = wmjoks/wmjoks_std

    print('Creating multivariable matrix')
    dat  = np.zeros((nday,nlon*2))
    dat[:,0:nlon] = wmjoks_norm
    dat[:,nlon:nlon*2] = sst_norm

    print('Calculating the multivariate PCs')

    # from geocat.ncomp import eofunc, eofunc_ts

    # neof = 2
    # eofs = eofunc(dat.T, neof)
    # eofs = eofs.rename({'evn':'eofs','dim_0':'lonn'})
    # eofs = eofs.rename('meofs')
    # eofs.to_netcdf('cfs_multivarEOFs.nc')


    pcs = meofs.data@dat.T

    pc1 = -pcs[0,:]
    pc2 = -pcs[1,:]

    PC1 = pc1/np.std(pc1)  # cfsv2 hindcast?
    PC2 = pc2/np.std(pc2)

    # Calculate MaKE and MaKI index

    pcmake = PC1 + np.fabs(PC2)
    pcmaki = PC1 + PC2

    pcmake = xr.DataArray(pcmake,name='pcmake',coords=[wmjoks.time], dims="time")
    pcmaki = xr.DataArray(pcmaki,name='pcmaki',coords=[wmjoks.time], dims="time")

    ###### Two Options #######

    # Minimum of 3 MONTHS. For example, April value = min (A,M,J)
    # Minimum of 90 days. For example, April value = min (April 1st - June 29th)

    ###########################

    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    startYear=sst['time'][90]
    print(str(startYear.values)[0:10])
    endYear=sst['time'][-1]
    date_format = '%Y-%m-%d'
    stdt = datetime.strptime(str(startYear.values)[0:10], date_format)
    edt = datetime.strptime(str(endYear.values)[0:10], date_format)

    datemin = datetime(stdt.year, stdt.month, stdt.day)
    datemax = datetime(edt.year, edt.month, edt.day)
    
    print('Calculating MaKE index and MaKI index')
    make = list()
    maki = list()

    while stdt <= edt:
        padt = stdt - relativedelta(days=90)
        make.append(pcmake.sel(time=slice(padt.strftime(date_format),stdt.strftime(date_format))).min('time',skipna='True').data.tolist())
        maki.append(pcmaki.sel(time=slice(padt.strftime(date_format),stdt.strftime(date_format))).min('time',skipna='True').data.tolist())
        stdt = stdt + relativedelta(months=1)

    time_mon = pd.date_range(datemin, datemax, freq='MS')#.to_pydatetime().tolist()
    make = xr.DataArray(make,name='make',coords=[time_mon],dims="time")
    maki = xr.DataArray(maki,name='maki',coords=[time_mon],dims="time")

    make_n = make/np.std(make)
    maki_n = maki/np.std(maki)

    return make_n,maki_n


