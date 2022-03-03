# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
# -*- coding: utf-8 -*-

"""
Utilities for plotting the Matsuno Dispersion Curves
Created by: Alejandro Jaramillo
ajaramillomoreno@gmail.com
2018


References:

Matsuno, T. (1966). Quasi-Geostrophic Motions in the Equatorial Area.
Journal of the Meteorological Society of Japan.
Ser. II, 44(1), 25–43.
https://doi.org/10.2151/jmsj1965.44.1_25

Wheeler, M., & Kiladis, G. N. (1999).
Convectively Coupled Equatorial Waves: Analysis of Clouds and Temperature in
the Wavenumber–Frequency Domain. Journal of the Atmospheric Sciences,
56(3), 374–399.
https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2

Kiladis, G. N., Wheeler, M. C., Haertel, P. T., Straub, K. H.,
& Roundy, P. E. (2009). Convectively coupled equatorial waves.
Reviews of Geophysics, 47(2), RG2003.
https://doi.org/10.1029/2008RG000266

Wheeler, M. C., & Nguyen, H. (2015).
TROPICAL METEOROLOGY AND CLIMATE | Equatorial Waves.
In Encyclopedia of Atmospheric Sciences (pp. 102–112). Elsevier.
https://doi.org/10.1016/B978-0-12-382225-3.00414-X
"""
import numpy as np
from scipy.optimize import fsolve
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
pi = np.pi
re    = 6.371008e6 # Earth's radius in meters
g     = 9.80665 # Gravitational acceleration [m s^{-2}]
omega = 7.292e-05 # Angular speed of rotation of Earth [rad s^{-1}]
deg2rad = pi/180 # Degrees to Radians
sec2day = 1./(24.*60.*60.) #Seconds to Days

def beta_parameters(latitude):
    """
    Function that calculates the beta-plane parameters that are a Functions
    of the latitude, where **beta** is the beta-plane parameter, **perimeter**
    is the perimeter in meters of the Earths's circunference at the given
    latitude.
    :param latitude:
        Latitude
    :type latitude: Float
    :return: (beta, perimeter)
    :rtype: tuple
    """
    beta  = 2.*omega*np.cos(abs(latitude)*deg2rad)/re
    perimeter = 2.*pi*re*np.cos(abs(latitude)*deg2rad)
    return (beta,perimeter)

def wn_array(max_wn,n_wn):
    """
    Creates an array with wavenumbers in the range (-max_wn,max_wn).
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: Array of Global Wavenumbers
    :rtype: Numpy Array
    """
    maxwn = abs(int(max_wn)) # maxwn is Positive Integer (maxwn > 0)
    n_wn = abs(int(n_wn))
    wn = np.linspace(-max_wn,max_wn,n_wn)
    return wn

def wn2k(wn,perimeter):
    """
    Converts an array of Global wave numbers to wavenumber in [rad m^{-1}].
    :param wn:
        Array of Global wavenumbers.
    :param perimeter:
        Perimeter in meters of the Earths's circunference at the given latitude
    :type wn: Numpy Array
    :type perimeter: Float
    :return: Array of Wavenumbers in [rad m^{-1}]
    :rtype: Numpy Array
    """
    wavelength = perimeter/wn # Wavekength[m]
    k  = 2.*pi/wavelength # Wavenumber[rad m^{-1}]
    return k

def afreq2freq(angular_frequency):
    """
    Convert angular frequency in [rad s^{-1}] to frequency in Cycles per
    Day(CPD).
    :param angular_frequency:
        Angular Frequency
    :type angular_frequency: Numpy Array
    :return: (Period,Frequency)
        Period in [days/cycle]
        Frequency in [cycles/day] Cycles per Day(CPD)
    :rtype: tuple
    """
    period = (2.*pi/angular_frequency)*sec2day # Period in [days/cycle]
    frequency = 1./period #[cycles/day] Cycles per Day(CPD)
    return (period,frequency)

def kelvin_mode(he,latitude=0,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Kelvin
    Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    k[k<=0] = np.nan
    angular_frequency = np.sqrt(g*he)*k # [rad s^{-1}]
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'Kelvin(he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def mrg_mode(he,latitude=0,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    #wn = wn[wn<0] # Extract only wn < 0
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    k[k>=0] = np.nan
    angular_frequency = np.sqrt(g*he)*k*(0.5-0.5*np.sqrt(1.+\
                        (4*beta/(k*k*np.sqrt(g*he))))) # [rad s^{-1}]
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'MRG(he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df
def eig_n_0(he,latitude=0,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
    :param he:
        Equivalent Depth
    :param latitude:
        Latitude
    :param max_wn:
        Max global wave number.
        The global wave number range is (-max_wn,max_wn)
    :param n_wn:
        Number of global wave numbers in the range (-max_wn,max_wn)
    :type he: Float
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    k[k<=0] = np.nan
    angular_frequency = np.sqrt(g*he)*k*(0.5+0.5*np.sqrt(1.+\
                        (4*beta/(k*k*np.sqrt(g*he))))) # [rad s^{-1}]
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'EIG(n=0,he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def er_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Mixed
    Rossby Gravity Wave for a given Equivalent Depth.
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
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the Equatorial Rossby dispersion relationship as
    # a seed for the solver function
    angular_frequency = -beta*k/((k*k)+(2.*n+1.)*(beta/np.sqrt(g*he)))
    angular_frequency[k>=0] = np.nan
    angular_frequency[k<0] = fsolve(dispersion,angular_frequency[k<0],\
                                    args=(k[k<0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'ER(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def eig_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Easterly
    Gravity Wave for a given Equivalent Depth.
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
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the EIG dispersion relationship as
    # a seed for the solver function
    angular_frequency = np.sqrt((2.*n+1.)*beta*np.sqrt(g*he)+(k**2)*g*he)
    angular_frequency[k<=0] = np.nan
    angular_frequency[k>0] = fsolve(dispersion,angular_frequency[k>0],\
                                    args=(k[k>0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'EIG(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def wig_n(he,n,latitude=0.,max_wn=50,n_wn=500):
    """
    Function that calculates the dispersion curve for the Equatorial Westerly
    Gravity Wave for a given Equivalent Depth.
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
    :type he: Float
    :type n: Integer
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    (beta,perimeter) = beta_parameters(latitude)
    wn = wn_array(max_wn,n_wn) #Global Wavenumber
    k  = wn2k(wn,perimeter) # Wavenumber[rad m^{-1}]
    # Use the Approximation to the WIG dispersion relationship as
    # a seed for the solver function
    angular_frequency = np.sqrt((2.*n+1.)*beta*np.sqrt(g*he)+(k**2)*g*he)
    angular_frequency[k>=0] = np.nan
    angular_frequency[k<0] = fsolve(dispersion,angular_frequency[k<0],\
                                    args=(k[k<0],n,he,beta))
    (period,frequency) = afreq2freq(angular_frequency)
    # Period in [days/cycle]
    # Frequency [cycles/day] Cycles per Day(CPD)
    name = 'WIG(n='+str(n)+',he='+str(he)+'m)'
    df = pd.DataFrame(data={name:frequency},index=wn)
    df.index.name = 'Wavenumber'
    return df

def dispersion(w,k,n,he,beta):
    """
    Dispersion relationship for Matsuno Modes(See Wheeler and Nguyen (2015)
    , Eq (13)).
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
    :type w: Float
    :type k: Float
    :type n: Integer
    :type he: Float
    :type beta: Float
    :return: Zero if w and k corresponds to a Matsuno Mode.
    :rtype: Float
    """
    disp = w**3-g*he*(k**2+(beta*(2.*n+1.)/np.sqrt(g*he)))*w-k*beta*g*he
    return disp

def matsuno_dataframe(he,n=[1,2,3],latitude=0.,max_wn=50,n_wn=500):
    """
    Creates a dataframe with all Matsuno modes for a given set of meridional
    mode numbers given in a list.
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
    :type he: Float
    :type n: List of integers (e.g. [1,2,3])
    :type latitude: Float
    :type maxwn: Positive Integer (max_wn > 0)
    :type n_wn: Integer
    :return: DataFrame with wn and frequency
    :rtype: DataFrame
    """
    df = []
    df.append(kelvin_mode(he,latitude,max_wn,n_wn))
    df.append(mrg_mode(he,latitude,max_wn,n_wn))
    df.append(eig_n_0(he,latitude,max_wn,n_wn))

    for nn in n:
        df.append(er_n(he,nn,latitude,max_wn,n_wn))
        df.append(eig_n(he,nn,latitude,max_wn,n_wn))
        df.append(wig_n(he,nn,latitude,max_wn,n_wn))

    df = reduce(lambda left,right: pd.merge(left,right,on='Wavenumber'), df)
    return df

def standar_plot(he,size=12,figsize=(8, 8),mx_wn=20,mx_freq=1.,labels='on'):
    """
    Creates a standard plot with all Matsuno modes for a given Equivalent Depth
    he.  This function plots the dispersion curves for the meridional mode
    numbers n = [1,2,3]. Some configuration is possible although with several
    limitations.
    :param he:
        Equivalent Depth
    :param size(optional):
        Text size
    :param figsize(optional):
        Figure size
    :param mx_wn(optional):
        Upper Limit of the Zonal Wave number range. The range is given
        by (-mx_wn,mx_wn)
    :param mx_freq(optional):
        Upper Limit for the y axis given as frequency in CPD.
    :param labels(optional):
        Plot the name of the Matsuno Modes in the figureself.
    :type he: Float
    :type size: Integer
    :type figsize: Tuple
    :type mx_wn: Integer
    :type mx_freq: Float
    :type mx_freq: String
    :return: Plot with Matsuno Modes
    :rtype: Matplotlib Figure
    """

    plt.rc('font', size=size)          # controls default text sizes
    plt.rc('axes', titlesize=size)     # fontsize of the axes title
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title

    df = matsuno_dataframe(he,n=[1,2,3])
    wn = df.index.values
    fig,ax = plt.subplots(figsize=figsize)

    for column in df:
        ax.plot(wn,df[column].values,color='k')

    ax.set_xlim(-mx_wn,mx_wn)
    ax.set_ylim(0,mx_freq)
    ax.set_xlabel('ZONAL WAVENUMBER')
    ax.set_ylabel('FREQUENCY (CPD)')
    plt.text(mx_wn-2*0.25*mx_wn,-0.06,'EASTWARD',fontsize=size-2)
    plt.text(-mx_wn+0.25*mx_wn,-0.06,'WESTWARD',fontsize=size-2)

    if labels=='on':
        # Print Kelvin Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.3*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][0],'Kelvin', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print MRG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.7*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][1],'MRG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print EIG(n=0) Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.1*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][2],'EIG(n=0)', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print ER Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.7*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][3]+0.01,'ER', \
        bbox={'facecolor':'none','edgecolor':'none'},fontsize=size+1)

        # Print EIG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int((len(p_wn)/2)+0.3*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][7],'EIG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print WIG Label
        p_wn = wn[np.logical_and(wn>=-mx_wn,wn<=mx_wn)]
        i = int(0.55*(len(p_wn)/2))
        i, = np.where(wn == p_wn[i])[0]
        plt.text(wn[i]-1,df.iloc[i][8],'WIG', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

        # Print n Labels
        p_wn = wn[wn>=0]
        i, = np.where(wn == p_wn[0])[0]
        plt.text(-1,df.iloc[i][4],'n=1', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)
        plt.text(-1,df.iloc[i][7],'n=2', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)
        plt.text(-1,df.iloc[i][10],'n=3', \
        bbox={'facecolor':'white','edgecolor':'none'},fontsize=size+1)

    plt.show()
    return fig

def matsuno_modes_wk(he=[12,25,50],n=[1,],latitude=0.,max_wn=20,n_wn=500):
    """
    Creates a dataframe with all Matsuno modes for a given set of meridional
    mode numbers given in a list.
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
        df.append(kelvin_mode(h,latitude,max_wn,n_wn))
        df.append(mrg_mode(h,latitude,max_wn,n_wn))
        df.append(eig_n_0(h,latitude,max_wn,n_wn))

        for nn in n:
            df.append(er_n(h,nn,latitude,max_wn,n_wn))
            df.append(eig_n(h,nn,latitude,max_wn,n_wn))
            df.append(wig_n(h,nn,latitude,max_wn,n_wn))

        df = reduce(lambda left,right: pd.merge(left,right,on='Wavenumber'), df)
        matsuno_modes[h] = df
    return matsuno_modes
