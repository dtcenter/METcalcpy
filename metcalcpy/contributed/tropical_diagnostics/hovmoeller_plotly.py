"""
Hovmoeller plots using plotly module.
"""

import numpy as np
import plotly.graph_objects as go
from netCDF4 import num2date
from kaleido.scopes.plotly import PlotlyScope


def hov_resources(pltvarname):
    """
    Set the colormap to be used for the Hovmoeller plot.
    :param pltvarname: name of variable to be plotted
    :type pltvarname: string
    :return: rgb string colorscale
    :rtype: color string
    """
    if pltvarname == 'precip':
        cmap_rgb = [[0, "rgb(255, 255, 255)"], [0.111111, "rgb(255, 255, 255)"],
                    [0.111111, "rgb(135, 206, 250)"], [0.222222, "rgb(135, 206, 250)"],
                    [0.222222, "rgb(30, 144, 255)"], [0.333333, "rgb(30, 144, 255)"],
                    [0.333333, "rgb(0, 0, 238)"], [0.444444, "rgb(0, 0, 238)"],
                    [0.444444, "rgb(131, 111, 255)"], [0.666667, "rgb(131, 111, 255)"],
                    [0.666667, "rgb(171, 130, 255)"], [0.888889, "rgb(171, 130, 255)"],
                    [0.888889, "rgb(145, 44, 238)"], [1, "rgb(145, 44, 238)"]]
    elif pltvarname == 'uwnd':
        cmap_rgb = [[0.0, "rgb(49,54,149)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.5, "rgb(255,255,255)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [1.0, "rgb(165,0,38)"]]
    elif pltvarname == 'vwnd':
        cmap_rgb = [[0.0, "rgb(49,54,149)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.5, "rgb(255,255,255)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [1.0, "rgb(165,0,38)"]]
    elif pltvarname == 'div':
        cmap_rgb = [[0.0, "rgb(49,54,149)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.5, "rgb(255,255,255)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [1.0, "rgb(165,0,38)"]]
    elif pltvarname == 'olr':
        cmap_rgb = [[0, "rgb(145, 44, 238)"], [0.14, "rgb(145, 44, 238)"],
                    [0.14, "rgb(171, 130, 255)"], [0.28, "rgb(171, 130, 255)"],
                    [0.28, "rgb(131, 111, 255)"], [0.42, "rgb(131, 111, 255)"],
                    [0.42, "rgb(0, 0, 238)"], [0.56, "rgb(0, 0, 238)"],
                    [0.56, "rgb(30, 144, 255)"], [0.71, "rgb(30, 144, 255)"],
                    [0.71, "rgb(135, 206, 250)"], [0.85, "rgb(135, 206, 250)"],
                    [0.85, "rgb(255, 255, 255)"], [1, "rgb(255, 255, 255)"]]
    else:
        cmap_rgb = [[0.0, "rgb(49,54,149)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.5, "rgb(255,255,255)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [1.0, "rgb(165,0,38)"]]

    return cmap_rgb


def get_clevels(pltvarname):
    """
    Set contour levels for given variable.
    :param pltvarname: name of variable to be plotted
    :type pltvarname: string
    :return: cmin, cmax, cspc
    :rtype: float
    """
    if pltvarname == "precip":
        cmin = 0.2
        cmax = 1.6
        cspc = 0.2
    elif (pltvarname == 'uwnd') or (pltvarname == 'vwnd'):
        cmin = -21.
        cmax = 21.
        cspc = 2.
    elif pltvarname == 'div':
        cmin = -0.000011
        cmax = 0.000011
        cspc = 0.000002
    elif pltvarname == 'olr':
        cmin = 160.
        cmax = 240.
        cspc = 20.
    else:
        print('Warning: Not a default variable name (' + pltvarname + '). To ensure best plotting results please '
                                                                      'specify min, max and spacing for contour levels.')
        cmin = -21.
        cmax = 21.
        cspc = 2.

    return cmin, cmax, cspc


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


def get_latstring(lats, latn):
    """
    Generate string describing the latitude band averaged over.
    :param lats: southern latitude limit of the average
    :type lats: float
    :param latn: northern latitude limit of the average
    :type latn: float
    :return: latstr
    :rtype: str
    """
    if lats < 0:
        hems = 'S'
        lats = -lats
    else:
        hems = 'N'
    if latn < 0:
        hemn = 'S'
        latn = -latn
    else:
        hemn = 'N'

    latstr = str(lats) + hems + " - " + str(latn) + hemn

    return latstr


def hovmoeller(data, lon, time, datestrt, datelast, plotpath, lats, latn, spd, source, pltvarname, lev=[],
               cmin=[], cmax=[], cspc=[]):
    """
    Main driver for plotting Hovmoeller diagrams.
    :param data: input data, should be (time, lon)
    :type data: numeric
    :param lon: longitude coordinate of data
    :type lon: float
    :param time: time coordinate of data
    :type time: datetime
    :param datestrt: start date for Hovmoeller, used in plot file name
    :type datestrt: str
    :param datelast: end date for Hovmoeller, used in plot file name
    :type datelast: str
    :param plotpath: path for saving the figure
    :type plotpath: str
    :param lats: southern latitude limit of the average
    :type lats: float
    :param latn: northern latitude limit of the average
    :type latn: float
    :param spd: number of observations per day
    :type spd: int
    :param source: source of the data, e.g. (ERAI, TRMM, ...), used in plot file name
    :type source: str
    :param pltvarname: name of variable to be plotted
    :type pltvarname: str
    :param lev: vertical level of data (optional)
    :type lev: str
    :param cmin: contour level minimum (optional)
    :type cmin: float
    :param cmax: contour level maximum (optional)
    :type cmax: float
    :param cspc: contour spacing (optional)
    :type cspc: float
    :return: none
    :rtype: none
    """

    """
    Set plot type and plot file name.
    """
    plttype = "png"
    plotname = plotpath + "Hovmoeller_" + source + pltvarname + lev + "_" + str(datestrt) + "-" + str(
        datelast) + "." + plttype

    """
    Set plot resources: colormap, time string, contour levels, latitude band string
    """
    cmap_rgb = hov_resources(pltvarname)

    timestr = get_timestr(time)

    if (not cmin) or (not cmax) or (not cspc):
        cmin, cmax, cspc = get_clevels(pltvarname)

    latstring = get_latstring(lats, latn)

    """
    Generate the Hovmoeller plot.
    """
    scope = PlotlyScope()
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            z=data.values,
            x=lon,
            y=timestr,
            colorscale=cmap_rgb,
            contours=dict(start=cmin, end=cmax, size=cspc,
                          showlines=False),
            colorbar=dict(title=data.attrs['units'],
                          len=0.6,
                          lenmode='fraction')
        )
    )

    fig.update_layout(
        title=source + " " + pltvarname + lev,
        width=600,
        height=900,
        annotations=[
            go.layout.Annotation(
                x=300,
                y=timestr[5],
                xref="x",
                yref="y",
                text=latstring,
                showarrow=False,
                bgcolor="white",
                opacity=0.9,
                font=dict(size=16)
            )
        ]
    )

    fig.update_xaxes(ticks="inside", tick0=0, dtick=30, title_text='longitude')
    fig.update_yaxes(autorange="reversed", ticks="inside", nticks=11)

    with open(plotname, "wb") as f:
        f.write(scope.transform(fig, format=plttype))

    return


def plot_pattcorr(PC, labels, plotpath, lats, latn):
    """
    Plot pattern correlation curves as a function of lead time.
    :param PC:
    :type PC:
    :param labels:
    :type labels:
    :param plotpath:
    :type plotpath:
    :param region:
    :type region: str
    :return:
    :rtype:
    """

    latstring = get_latstring(lats, latn)

    plttype = "png"
    plotname = plotpath + "PatternCorrelationHovmoeller." + plttype
    nlines = len(labels)

    colors = ['black', 'dodgerblue', 'orange', 'seagreen', 'firebrick']

    scope = PlotlyScope()
    fig = go.Figure()
    for ll in np.arange(0, nlines):
        fig.add_trace(go.Scatter(x=PC['fchrs'], y=PC[:, ll],
                                 mode='lines',
                                 name=labels[ll],
                                 line=dict(color=colors[ll], width=2)))

    fig.update_layout(title=latstring)

    fig.update_xaxes(ticks="", tick0=0, dtick=24, title_text='lead time (h)')
    fig.update_yaxes(ticks="", tick0=0, dtick=0.1, title_text='correlation')

    with open(plotname, "wb") as f:
        f.write(scope.transform(fig, format=plttype))
