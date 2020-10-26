# -*- coding: utf-8 -*-

"""
Utilities for plotting Space-time spectra.
Created by: Maria Gehne
maria.gehne@noaa.gov
2019
"""
import numpy as np
from scipy.optimize import fsolve
from functools import reduce
import pandas as pd
import Ngl as ngl
import string

"""
local scripts
"""
import matsuno_plot as mp

pi = np.pi
re = 6.371008e6  # Earth's radius in meters
g = 9.80665  # Gravitational acceleration [m s^{-2}]
omega = 7.292e-05  # Angular speed of rotation of Earth [rad s^{-1}]
deg2rad = pi / 180  # Degrees to Radians
sec2day = 1. / (24. * 60. * 60.)  # Seconds to Days

"""
The following utilities include a zonal background flow for mid-latitude Rossby wave dispersion
curves.
Created by: Maria Gehne
2019
"""


def dispersion_bg(w, k, n, he, beta, u):
    """
    Dispersion relationship for Matsuno Modes with a non-zero background flow.
    The roots of this function correspond to the angular frequencies of the
    Matsuno modes for a given k.
    :param w:
        Angular Frequency
    :param k:
        Longitudinal Wavenumber
    :param n:
        Meridional Mode Number
    :param he:
        Equivalent Depth
    :param beta:
        Beta-Plane Parameter
    :param u:
        Background flow in m/s
    :type w: Float
    :type k: Float
    :type n: Integer
    :type he: Float
    :type beta: Float
    :return: Zero if w and k corresponds to a Matsuno Mode.
    :rtype: Float
    """
    w = -u * k + w
    disp = w ** 3 - g * he * (k ** 2 + (beta * (2. * n + 1.) / np.sqrt(g * he))) * w - k * beta * g * he
    return disp


def er_n_bg(he, n, latitude=0., max_wn=50, n_wn=500, u=30):
    """
    Function that calculates the dispersion curve for the beta-plane Rossby wave
    for a given Equivalent Depth and background wind.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :param u:
        Background wind in m/s
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta, perimeter) = mp.beta_parameters(latitude)
    wn = mp.wn_array(max_wn, n_wn)  # Global Wavenumber
    k = mp.wn2k(wn, perimeter)  # Wavenumber[rad m^{-1}]
    # Use the Approximation to the Equatorial Rossby dispersion relationship as
    # a seed for the solver function
    angular_frequency = -beta * k / ((k * k) + (2. * n + 1.) * (beta / np.sqrt(g * he)))
    angular_frequency[:] = fsolve(dispersion_bg, angular_frequency[:], \
                                  args=(k[:], n, he, beta, u))
    (period, frequency) = mp.afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'ER(n=' + str(n) + ',he=' + str(he) + 'm)'
    df = pd.DataFrame(data={name: frequency}, index=wn)
    df.index.name = 'Wavenumber'
    return df


def matsuno_modes_wk_bg(he=[3000, 7000, 10000], n=[1, ], latitude=0., max_wn=20, n_wn=500, u=30):
    """
    Creates a dataframe with Rossby modes for a non-zero background flow for a given set
    of meridional mode numbers given in a list.
    :param he:
        Equivalent Depth
    :param n:
        Meridional Mode Number
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :param u:
        Background wind in m/s
    :type he: Float
    :type n: List of integers (e.g. [1,2,3])
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    matsuno_modes = {}
    for h in he:
        df = []
        for nn in n:
            df.append(er_n_bg(h, nn, latitude, max_wn, n_wn, u))
        df = reduce(lambda left, right: pd.merge(left, right, on='Wavenumber'), df)
        matsuno_modes[h] = df
    return matsuno_modes


"""
Plotting utilities using the NCAR NGL python module to generate space-time plots.
"""


def text_labels(Symmetry="symm"):
    """
    Create text labels and their locations on the space-time plot
    :param Symmetry:
        symmetric, anti-symmetric or midlatude dispersion curves
    :return: name of the waves in the dataframe, textlabels, textlocations
    :rtype: strings, floats
    """
    if Symmetry == "symm":
        matsuno_names = ["Kelvin(he=", "ER(n=1,he=", "EIG(n=1,he=", "WIG(n=1,he="]
        textlabels = ["Kelvin", "n=1 ER", "n=1 IG", "h=50", "h=25", "h=12"]
        textlocsX = [11.5, -10.7, -3.0, -14.0, -14.0, -14.0]
        textlocsY = [0.4, 0.07, 0.45, 0.78, 0.6, 0.46]
    else:
        if Symmetry == "asymm":
            matsuno_names = ["MRG(he=", "EIG(n=2,he=", "EIG(n=0,he=", "WIG(n=2,he="]
            textlabels = ["MRG", "n=2 IG", "n=0 EIG", "h=50", "h=25", "h=12"]
            textlocsX = [-10.0, -3.0, 6.5, -10.0, -10.0, -10.0]
            textlocsY = [0.15, 0.58, 0.4, 0.78, 0.63, 0.51]
        else:
            if Symmetry == "midlat":
                matsuno_names = ["ER(n=1,he="]
                textlabels = ["n=1 ER", "h=10000", "h=7000", "h=3000"]
                textlocsX = [9.0, -6.0, -6.0, -6.0]
                textlocsY = [0.12, 0.18, 0.14, 0.06]
            else:
                matsuno_names = ["Kelvin(he=", "ER(n=1,he=", "EIG(n=1,he=", "WIG(n=1,he="]
                textlabels = ["Kelvin", "n=1 ER", "n=1 IG", "h=50", "h=25", "h=12"]
                textlocsX = [11.5, -10.7, -3.0, -14.0, -14.0, -14.0]
                textlocsY = [0.4, 0.07, 0.45, 0.78, 0.6, 0.46]

    return (matsuno_names, textlabels, textlocsX, textlocsY)


def coh_resources(contourmin=0.05, contourmax=0.55, contourspace=0.05, FillMode="AreaFill", flim=0.8, nWavePlt=20):
    # fspace = 0.05
    # fspaceminor = 0.01
    # nclev = int((flim - 0.) / fspaceminor + fspace / fspaceminor)
    # wspace = 5
    res = ngl.Resources()
    res.nglDraw = False
    res.nglFrame = False
    res.nglMaximize = False
    res.cnLinesOn = False
    res.cnFillOn = True
    res.cnFillMode = FillMode
    res.cnLineLabelsOn = False
    res.cnInfoLabelOn = False
    res.lbLabelBarOn = False
    res.cnFillPalette = "WhiteBlueGreenYellowRed"
    res.cnLevelSelectionMode = "ManualLevels"
    res.cnMinLevelValF = contourmin
    res.cnMaxLevelValF = contourmax
    res.cnLevelSpacingF = contourspace
    res.tiYAxisString = "frq (cpd)"
    res.tiXAxisString = "zonal wavenumber"
    res.sfYCStartV = 0.
    res.sfYCEndV = flim
    res.sfXCStartV = -nWavePlt
    res.sfXCEndV = nWavePlt

    return res


def phase_resources(flim=0.8, nWavePlt=20):
    res_a = ngl.Resources()
    res_a.nglDraw = False
    res_a.nglFrame = False
    res_a.vfYCStartV = 0.
    res_a.vfYCEndV = flim
    res_a.vfXCStartV = -nWavePlt
    res_a.vfXCEndV = nWavePlt
    res_a.vcRefMagnitudeF = 1.0  # define vector ref mag
    res_a.vcRefLengthF = 0.015  # define length of vec ref
    res_a.vcRefAnnoOrthogonalPosF = -1.0  # move ref vector
    res_a.vcRefAnnoArrowLineColor = "black"  # ref vector color
    res_a.vcMinDistanceF = 0.03  # thin out vectors
    res_a.vcMapDirection = False
    res_a.vcRefAnnoOn = False  # do not draw
    res_a.vcLineArrowThicknessF = 4
    res_a.vcLineArrowHeadMinSizeF = 0.005
    res_a.vcLineArrowHeadMaxSizeF = 0.005

    return res_a


def panel_resources(nplot=4, abc=['a', 'b', 'c', 'd']):
    res_p = ngl.Resources()
    res_p.nglFrame = True
    res_p.nglMaximize = True
    res_p.nglPanelLabelBar = True
    # res_p.nglPanelRight = 0.9
    # res_p.nglPanelLeft = 0.15
    # res_p.nglPanelBottom = 0.05
    res_p.lbOrientation = "vertical"
    res_p.nglPanelLabelBarLabelFontHeightF = 0.02
    res_p.nglPanelLabelBarHeightF = 0.27
    res_p.nglPanelLabelBarParallelPosF = 0.01
    res_p.nglPanelFigureStrings = abc[0:nplot]
    res_p.nglPanelFigureStringsJust = "TopLeft"

    return res_p


def plot_coherence(cohsq, phase1, phase2, symmetry=("symm"), source="", vars1="", vars2="", plotpath="./", wkstype="png",
                   flim=0.5, nwaveplt=20, cmin=0.05, cmax=0.55, cspc=0.05, plotxy=[1, 1], N=[1, 2]):

    dims = cohsq.shape
    nplot = dims[0]

    FillMode = "AreaFill"

    # text labels
    abc = list(string.ascii_lowercase)

    # plot resources
    #wkstype = wkstype
    wks = ngl.open_wks(wkstype, plotpath + "SpaceTimeCoherence_")
    plots = []

    # coherence2 plot resources
    res = coh_resources(cmin, cmax, cspc, FillMode, flim, nwaveplt)
    # phase arrow resources
    res_a = phase_resources(flim, nwaveplt)

    # dispersion curve resources
    dcres = ngl.Resources()
    dcres.gsLineThicknessF = 2.0
    dcres.gsLineDashPattern = 0

    # text box resources
    txres = ngl.Resources()
    txres.txPerimOn = True
    txres.txFontHeightF = 0.013
    txres.txBackgroundFillColor = "Background"

    # panel plot resources
    res_p = panel_resources(nplot, abc)

    # plot contours and phase arrows
    pp = 0
    while pp < nplot:
        coh2 = cohsq[pp, :, :]
        phs1 = phase1[pp, :, :]
        phs2 = phase2[pp, :, :]
        if len(symmetry) == nplot:
            Symmetry = symmetry[pp]
        else:
            Symmetry = symmetry
        var1 = vars1[pp]
        var2 = vars2[pp]

        res.tiMainString = source + "    coh^2(" + var1 + "," + var2 + ") " + Symmetry
        plot = ngl.contour(wks, coh2, res)
        plot_a = ngl.vector(wks, phs1, phs2, res_a)
        ngl.overlay(plot, plot_a)

        (matsuno_names, textlabels, textlocsX, textlocsY) = text_labels(Symmetry)
        nlabel = len(textlocsX)

        # generate matsuno mode dispersion curves
        if Symmetry == "midlat":
            He = [3000, 7000, 10000]
            matsuno_modes = matsuno_modes_wk_bg(he=He, n=N, latitude=0., max_wn=nwaveplt, n_wn=500, u=30)
        else:
            He = [12, 25, 50]
            matsuno_modes = mp.matsuno_modes_wk(he=He, n=N, latitude=0., max_wn=nwaveplt, n_wn=500)

        # add polylines for dispersion curves
        for he in matsuno_modes:
            df = matsuno_modes[he]
            wn = df.index.values
            for wavename in df:
                for matsuno_name in matsuno_names:
                    if wavename == (matsuno_name + str(he) + "m)"):
                        wave = df[wavename].values
                        wnwave = wn[~np.isnan(wave)]
                        wave = wave[~np.isnan(wave)]
                        ngl.add_polyline(wks, plot, wnwave, wave, dcres)

        # add text boxes
        ll = 0
        while ll < nlabel:
            ngl.add_text(wks, plot, textlabels[ll], textlocsX[ll], textlocsY[ll], txres)
            ll += 1

        plots.append(plot)
        pp += 1


    # panel plots
    ngl.panel(wks, plots, plotxy, res_p)
    ngl.delete_wks(wks)
    ngl.end()

    return


def plot_power(Pow, symmetry=("symm"), source="", var1="", plotpath="./", flim=0.5, nWavePlt=20, cmin=0.05, cmax=0.55,
               cspc=0.05, nplot=1, N=[1, 2]):
    FillMode = "AreaFill"

    # text labels
    abc = list(string.ascii_lowercase)

    # plot resources
    wkstype = "png"
    wks = ngl.open_wks(wkstype, plotpath + "SpaceTimePower_" + source + var1)
    plots = []

    # coherence2 plot resources
    res = coh_resources(cmin, cmax, cspc, FillMode, flim, nWavePlt)
    # phase arrow resources
    resA = phase_resources(flim, nWavePlt)

    # dispersion curve resources
    dcres = ngl.Resources()
    dcres.gsLineThicknessF = 2.0
    dcres.gsLineDashPattern = 0

    # text box resources
    txres = ngl.Resources()
    txres.txPerimOn = True
    txres.txFontHeightF = 0.013
    txres.txBackgroundFillColor = "Background"

    # panel plot resources
    resP = panel_resources(nplot, abc)

    # plot contours and phase arrows
    pp = 0
    while pp < nplot:
        if nplot == 1:
            coh2 = Pow
            Symmetry = symmetry
        else:
            coh2 = Pow[pp, :, :]
            Symmetry = symmetry[pp]

        res.tiMainString = source + "    log10( Power(" + var1 + "))           " + Symmetry
        plot = ngl.contour(wks, coh2, res)

        (matsuno_names, textlabels, textlocsX, textlocsY) = text_labels(Symmetry)
        nlabel = len(textlocsX)

        # generate matsuno mode dispersion curves
        if Symmetry == "midlat":
            He = [3000, 7000, 10000]
            matsuno_modes = mp.matsuno_modes_wk_bg(he=He, n=N, latitude=0., max_wn=nWavePlt, n_wn=500, u=25)
        else:
            He = [12, 25, 50]
            matsuno_modes = mp.matsuno_modes_wk(he=He, n=N, latitude=0., max_wn=nWavePlt, n_wn=500)

        # add polylines for dispersion curves
        for he in matsuno_modes:
            df = matsuno_modes[he]
            wn = df.index.values
            for wavename in df:
                for matsuno_name in matsuno_names:
                    if wavename == (matsuno_name + str(he) + "m)"):
                        wave = df[wavename].values
                        wnwave = wn[~np.isnan(wave)]
                        wave = wave[~np.isnan(wave)]
                        ngl.add_polyline(wks, plot, wnwave, wave, dcres)

        # add text boxes
        ll = 0
        while ll < nlabel:
            ngl.add_text(wks, plot, textlabels[ll], textlocsX[ll], textlocsY[ll], txres)
            ll += 1

        plots.append(plot)
        pp += 1

        # panel plots
    ngl.panel(wks, plots, [nplot // 2 + 1, nplot // 2 + 1], resP)
    ngl.delete_wks(wks)
    #ngl.end()

    return
