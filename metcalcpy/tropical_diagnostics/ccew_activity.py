"""
Contains functions to project data onto CCEW EOFs.

List of functions:

waveact:

rem_seas_cyc:

waveproj:

"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from netCDF4 import num2date


def waveact(data: object, wave: str, eofpath: str, spd: int, res: str, nlat: int, opt=False):
    """
    Main script to compute the wave activity index.
    :param data: DataArray containing the raw data
    :param wave: string describing the wave name
    :param eofpath: file path to EOFs
    :param spd: number of obs per day
    :param res: resolution of the data ('1p0')
    :param nlat: number of latitudes in the data (180 or 181)
    :param opt: optional parameter, currently not used
    :return: DataArray containing the time series of wave activity
    """
    # read EOFs from file
    if (wave == 'Kelvin' or wave == 'kelvin'):
        eofname = 'EOF_1-4_130-270E_-15S-15N_persiann_cdr_'+res+'_nlat'+str(nlat)+'_fillmiss8314_1983-2016_Kelvinband_'
    elif (wave == 'ER' or wave == 'er'):
        eofname = 'EOF_1-4_60-220E_-21S-21N_persiann_cdr_'+res+'_nlat'+str(nlat)+'_fillmiss8314_1983-2016_ERband_'

    ds = xr.open_dataset(eofpath + eofname + '01.nc')
    nlat = len(ds.lat)
    nlon = len(ds.lon)
    eofnum = np.arange(4) + 1
    neof = 4
    month = np.arange(12) + 1
    nmon = 12
    ntim = len(data['time'])

    eofseas = xr.DataArray(0., coords=[month, eofnum, ds.lat, ds.lon], dims=['month', 'eofnum', 'lat', 'lon'])

    for ss in month:
        monthnum = f"{ss:02d}"
        ds = xr.open_dataset(eofpath + eofname + monthnum + '.nc')
        eofseas[ss - 1, :, :, :] = ds.eof
    ds.close()

    # remove mean annual cycle
    print("remove annual cycle")
    doy = data['time.dayofyear']
    doy_oad = doy[0::spd]
    nd = len(doy_oad) - len(np.unique(doy_oad))
    if nd > 0:
        data_anom = rem_seas_cyc(data)
    else:
        data_anom = rem_seas_mean(data)

    # compute projection
    print("project onto EOFs")
    tswave = waveproj(data_anom, eofseas)

    # compute activity
    waveact = xr.DataArray(0., coords=[data_anom.time], dims=['time'])
    waveact.values = np.sqrt(np.sum(np.square(tswave), 0))
    try:
        waveact.attrs['units'] = data.attrs['units']
    except KeyError:
        print('Data has no units attribute, cannot attach units to activity')
    waveact.attrs['name'] = wave+' activity'

    del data, data_anom

    return waveact


def rem_seas_cyc(data: object, opt: object = False) -> object:
    """
    Read in a xarray data array with time coordinate containing daily data. Compute anomalies from daily climatology.
    :type data: xr.DataArray
    :param data: xarray
    :param opt: optional parameter, not currently used
    :return: xr.DataArray containing anomalies from daily climatology
    """
    da = xr.DataArray(np.arange(len(data['time'])), coords=[data['time']], dims=['time'])
    month_day_str = xr.DataArray(da.indexes['time'].strftime('%m-%d'), coords=da.coords, name='month_day_str')
    time = data['time']

    data = data.rename({'time': 'month_day_str'})
    month_day_str = month_day_str.rename({'time': 'month_day_str'})
    data = data.assign_coords(month_day_str=month_day_str)
    clim = data.groupby('month_day_str').mean('month_day_str')

    data_anom = data.groupby('month_day_str') - clim
    data_anom = data_anom.rename({'month_day_str': 'time'})
    data_anom = data_anom.assign_coords(time=time)

    return data_anom

def rem_seas_mean(data: object, opt: object = False) -> object:
    """
    Read in a xarray data array with time coordinate containing daily data. Compute anomalies from time mean.
    :type data: xr.DataArray
    :param data: xarray
    :param opt: optional parameter, not currently used
    :return: xr.DataArray containing anomalies from daily climatology
    """
    clim = data.mean('time')
    data_anom = data - clim

    return data_anom


def waveproj(data_anom: object, eofseas: object):
    """
    Compute the projection onto the CCEW EOFS
    :param data_anom: anomalies of precipitation
    :param eofseas: xarray dataarray containing the monthly EOF patterns
    :return: wave time series projected onto each EOF
    """
    data_anom = data_anom.sel(lat=slice(eofseas.lat.min(), eofseas.lat.max()),
                              lon=slice(eofseas.lon.min(), eofseas.lon.max()))
    mm = data_anom.time.dt.month
    ntim = len(data_anom.time)
    neof = len(eofseas.eofnum)
    proj_wave: object = xr.DataArray(0., coords=[eofseas.eofnum, data_anom.time], dims=['eofnum', 'time'])
    tswave: object = xr.DataArray(0., coords=[data_anom.time], dims=['time'])
    for tt in range(ntim):
        eof = eofseas[mm[tt] - 1, :, :, :]
        for ee in range(neof):
            proj_wave[ee, tt] = eof[ee, :, :] @ data_anom[tt, :, :]

    return proj_wave


def wave_skill(act):
    """

    :param act:
    :type act:
    :return:
    :rtype:
    """
    nfchr, nlines, ntim = act.shape
    skill = act[:, 1::, 0].squeeze()

    for ff in np.arange(nfchr):
        for ll in np.arange(nlines-1):
            tmpskill = np.corrcoef(act[ff, ll+1, :], act[ff, 0, :])
            skill[ff, ll] = tmpskill[0, 1]

    return skill

def get_timestr(time):
    """
    Generate time string for y-axis labels.
    :param time: time coordinate
    :type time: datetime object
    :return: timestr
    :rtype: str
    """
    ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
    date = num2date(ts, 'hours since 1970-01-01T00:00:00Z')
    timestr = [i.strftime("%Y-%m-%d %H:%M") for i in date]

    return timestr


def plot_activity(act, wavename, labels, plotpath, fchr=[]):
    """
    Plot pattern correlation curves as a function of lead time.
    :param act:
    :type act:
    :param labels:
    :type labels:
    :return:
    :rtype:
    """

    plttype = "png"
    plotname = plotpath + wavename + "Activity." + plttype
    if fchr:
        plotname = plotpath + wavename + "Activity_f" + f"{fchr:03d}" + "." + plttype

    nlines, ntim = act.shape

    timestr = get_timestr(act['time'])

    fig = go.Figure()
    for ll in np.arange(nlines):
        fig.add_trace(go.Scatter(x=timestr, y=act[ll, :].values,
                                mode='lines',
                                name=labels[ll]))

    fig.update_layout(
        title=wavename + " FH" + f"{fchr:03d}",
        yaxis=dict(range=[0, 25]))

    #fig.update_xaxes(ticks="", tick0=0, dtick=12, title_text='date')
    fig.update_yaxes(ticks="", tick0=0, dtick=1., title_text='activity')

    fig.write_image(plotname)


def plot_skill(skill, wavename, labels, plotpath):
    """
    Plot pattern correlation curves as a function of lead time.
    :param skill:
    :type skill:
    :param labels:
    :type labels:
    :return:
    :rtype:
    """

    plttype = "png"
    plotname = plotpath + wavename + "Skill." + plttype

    nfchr, nlines = skill.shape

    fig = go.Figure()
    for ll in np.arange(nlines):
        fig.add_trace(go.Scatter(x=skill['fchrs'], y=skill[:, ll],
                                mode='lines',
                                name=labels[ll]))

    fig.update_layout(
        title=wavename + " skill",
        yaxis=dict(range=[0, 1]))

    fig.update_xaxes(ticks="", tick0=0, dtick=24, title_text='lead time (h)')
    fig.update_yaxes(ticks="", tick0=0, dtick=0.1, title_text='skill correlation')

    fig.write_image(plotname)
