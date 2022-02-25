# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Contains functions to compute space-time power, coherence and phase spectra.
Mirrors the capabilities of NCL.

List of functions:

mjo_cross_segment:

mjo_cross_segment_realfft:

get_symmasymm:

mjo_cross_coh2pha:

smooth121:

smooth121_1D:

window_cosbell:

mjo_cross:
  The main function to compute cross-spectra is called as
  result = mjo_cross(X,Y,nperseg,segoverlap). This splits the data in X and Y
  into segments of length nperseg, the segments overlap by segoverlap.

kf_filter_mask:

kf_filter:

"""

import numpy as np
from scipy import signal

pi = np.pi
re = 6.371008e6  # Earth's radius in meters
g = 9.80665  # Gravitational acceleration [m s^{-2}]
omega = 7.292e-05  # Angular speed of rotation of Earth [rad s^{-1}]
beta = 2. * omega / re  # beta parameter at the equator


def mjo_cross_segment(XX, YY, opt=False):
    """
    Compute the FFT to get the power and cross-spectra for one time segment.
    :param XX: Input array (time, lat, lon)
    :param YY: Input array (time, lat, lon)
    :param opt: Optional parameter, not currently used. Set to False.
    :return STC: Spectra array of shape (8, nfreq, nwave). Last 4 entries are blank and need to be computed by calling
    mjo_cross_coh2pha. The first 4 entries contain power spectra for XX, power spectra for YY, co-spectra between XX
    and YY, quadrature spectra between XX and YY.
    """
    NT, NM, NL = XX.shape
    # compute fourier decomposition in time and longitude
    Xfft = np.fft.fft2(XX, axes=(0, 2))
    Yfft = np.fft.fft2(YY, axes=(0, 2))
    # normalize by # time samples
    Xfft = Xfft / (NT * NL)
    Yfft = Yfft / (NT * NL)
    # shift 0 frequency and 0 wavenumber to the center
    Xfft = np.fft.fftshift(Xfft, axes=(0, 2))
    Yfft = np.fft.fftshift(Yfft, axes=(0, 2))

    # average the power spectra across all latitudes
    PX = np.average(np.square(np.abs(Xfft)), axis=1)
    PY = np.average(np.square(np.abs(Yfft)), axis=1)

    # compute co- and quadrature spectrum
    PXY = np.average(np.conj(Yfft) * Xfft, axis=1)
    CXY = np.real(PXY)
    QXY = np.imag(PXY)

    PX = PX[:, ::-1]
    PY = PY[:, ::-1]
    CXY = CXY[:, ::-1]
    QXY = QXY[:, ::-1]

    # test if time and longitude are odd or even, fft algorithm
    # returns the Nyquist frequency once for even NT or NL and twice
    # if they are odd
    if NT % 2 == 1:
        nfreq = NT
        if NL % 2 == 1:
            nwave = NL
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
        else:
            nwave = NL + 1
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[:, :, 0] = STC[:, :, NL]
    else:
        nfreq = NT + 1
        if NL % 2 == 1:
            nwave = NL
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
            STC[:, NT, :] = STC[:, 0, :]
        else:
            nwave = NL + 1
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[0, NT, :] = STC[0, 0, :]
            STC[:, :, 0] = STC[:, :, NL]

    return STC


def mjo_cross_segment_realfft(XX, YY, opt=False):
    """
    Compute the FFT to get the power and cross-spectra for one time segment.
    :param XX: Input array (time, lat, lon)
    :param YY: Input array (time, lat, lon)
    :param opt: Optional parameter, not currently used. Set to False.
    :return STC: Spectra array of shape (8, nfreq, nwave). Last 4 entries are blank and need to be computed by calling
    mjo_cross_coh2pha. The first 4 entries contain power spectra for XX, power spectra for YY, co-spectra between XX
    and YY, quadrature spectra between XX and YY.
    """
    NT, NM, NL = XX.shape

    XX = np.transpose(XX, axes=[1, 2, 0])  # is now (lat, lon, time)
    YY = np.transpose(YY, axes=[1, 2, 0])  # is now (lat, lon, time)

    # compute fourier decomposition in time and longitude
    Xfft = np.fft.rfft2(XX, axes=(1, 2))  # (lat, nlon, ntim)
    Yfft = np.fft.rfft2(YY, axes=(1, 2))   # (lat, nlon, ntim)

    # return array to (time, lat, lon)
    Xfft = np.transpose(Xfft, axes=[2, 0, 1])
    Yfft = np.transpose(Yfft, axes=[2, 0, 1])

    # normalize by # time samples
    Xfft = Xfft / (NT * NL)
    Yfft = Yfft / (NT * NL)

    # shift 0 wavenumber to the center
    Xfft = np.fft.fftshift(Xfft, axes=2)
    Yfft = np.fft.fftshift(Yfft, axes=2)

    # average the power spectra across all latitudes
    PX = np.average(np.square(np.abs(Xfft)), axis=1)
    PY = np.average(np.square(np.abs(Yfft)), axis=1)

    # compute co- and quadrature spectrum
    PXY = np.conj(Xfft) * Yfft
    CXY = np.average(np.real(PXY), axis=1)
    QXY = np.average(np.imag(PXY), axis=1)

    PX = PX[:, ::-1]
    PY = PY[:, ::-1]
    CXY = CXY[:, ::-1]
    QXY = QXY[:, ::-1]

    # test if time and longitude are odd or even, fft algorithm
    # returns the Nyquist frequency once for even NT or NL and twice
    # if they are odd
    NT = int(NT / 2) + 1
    if NT % 2 == 1:
        nfreq = NT
        if NL % 2 == 1:
            nwave = NL
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
        else:
            nwave = NL + 1
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[:, :, 0] = STC[:, :, NL]
    else:
        nfreq = NT + 1
        if NL % 2 == 1:
            nwave = NL
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
            STC[:, NT, :] = STC[:, 0, :]
        else:
            nwave = NL + 1
            STC = np.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[:, NT, :] = STC[:, 0, :]
            STC[:, :, 0] = STC[:, :, NL]

    return STC


def get_symmasymm(X, lat, opt=False):
    """
    Split the data in X into symmetric and anti-symmetric
    (across the equator) parts. Return only the part we are
    interested in.
    :param X: Array (time, lat, lon).
    :param lat: Latitude values corresponding to dimension 1 of X.
    :param opt: Parameter to choose symmetric or anti-symmetric part across the equator.
    :return : symmetric or anti-symmetric part of X
    """
    if opt:
        NT, NM, NL = X.shape
        if opt == 'symm':
            x = X[:, lat[:] >= 0, :]
            if len(lat) % 2 == 1:
                for ll in range(NM // 2 + 1):
                    x[:, ll, :] = 0.5 * (X[:, ll, :] + X[:, NM - ll - 1, :])
            else:
                for ll in range(NM // 2):
                    x[:, ll, :] = 0.5 * (X[:, ll, :] + X[:, NM - ll - 1, :])
        else:
            if opt == 'asymm' or opt == 'anti-symm':
                x = X[:, lat[:] > 0, :]
                if len(lat) % 2 == 1:
                    for ll in range(NM // 2):
                        x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
                else:
                    for ll in range(NM // 2):
                        x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
    else:
        print("Please provide a valid option: symm or asymm.")

    return x


def mjo_cross_coh2pha(STC, opt=False):
    """
    Compute coherence squared and phase spectrum from averaged power and
    cross-spectral estimates.
    :param STC: Spectra array.
    :return STC: Spectra array of the same size with entries 4-7 (coherence squared, phase angle, phase component 1,
    phase component 2) recomputed based on the power and cross-spectra in entries 0-3.
    """

    nvar, nfreq, nwave = STC.shape

    PX = STC[0, :, :]
    PY = STC[1, :, :]
    CXY = STC[2, :, :]
    QXY = STC[3, :, :]

    PY[PY == 0] = np.nan
    COH2 = (np.square(CXY) + np.square(QXY)) / (PX * PY)
    PHAS = np.arctan2(QXY, CXY)

    V1 = -QXY / np.sqrt(np.square(QXY) + np.square(CXY))
    V2 = CXY / np.sqrt(np.square(QXY) + np.square(CXY))

    STC[4, :, :] = COH2
    STC[5, :, :] = PHAS
    STC[6, :, :] = V1
    STC[7, :, :] = V2

    return (STC)


def smooth121(STC, freq, opt=False):
    """
    This function takes a coherence spectra array STC [ 8, nfreq, nwave] and smoothes the first
    4 entries ( Xpower, Ypower, Co-spectrum, Quadrature-spectrum ). Entries 4-7 of STC need to
    be recomputed after smoothing to have the matching phase, coherence-squared, and phase angle
    components.
    Smooth only in frequency and only positive frequencies.
    :param array_in:
        Input array
    :type array_in: np array
    :return: STC
    :rtype: np array
    """
    nvar, nfreq, nwave = STC.shape

    # find time-mean index
    indfreqzero = int(np.where(freq == 0.)[0])

    for wv in range(0, nwave):
        STC[0, indfreqzero + 1:, wv] = smooth121_1D(STC[0, indfreqzero + 1:, wv])
        STC[1, indfreqzero + 1:, wv] = smooth121_1D(STC[1, indfreqzero + 1:, wv])
        STC[2, indfreqzero + 1:, wv] = smooth121_1D(STC[2, indfreqzero + 1:, wv])
        STC[3, indfreqzero + 1:, wv] = smooth121_1D(STC[3, indfreqzero + 1:, wv])

    return (STC)


def smooth121_1D(array_in):
    """
    Smoothing function that takes a 1D array and passes it through a 1-2-1 filter.
    This function is a modified version of the wk_smooth121 from NCL.
    The weights for the first and last points are  3-1 (1st) or 1-3 (last) conserving the total sum.
    :param array_in:
        Input array
    :type array_in: np array
    :return: array_out
    :rtype: np array
    """

    temp = np.copy(array_in)
    array_out = np.copy(temp) * 0.0
    #weights = np.array([1.0, 2.0, 1.0]) / 4.0
    #sma = np.convolve(temp, weights, 'valid')
    #array_out[1:-1] = sma

    for i in np.arange(0, len(temp), 1):
        if np.isnan(temp[i]):
            array_out[0] = np.nan
        elif i == 0 or np.isnan(temp[i-1]):
            array_out[i] = (3*temp[i]+temp[i+1])/4
        elif i == (len(temp)-1) or np.isnan(temp[i+1]):
            array_out[i] = (3 * temp[i] + temp[i-1]) / 4
        else:
            array_out[i] = (temp[i+1] + 2 * temp[i] + temp[i-1]) / 4

    return array_out


def window_cosbell(N, pct, opt=False):
    """
    Compute an equivalent tapering window to the NCL taper function.
    :param N: Length of the time series to be tapered.
    :param pct: Percent of the time series to taper.
    :return x: Array of length N and values 1 with pct/2 beginning and end values tapered to zero.
    """

    x = np.ones(N, dtype='double')
    M = int((pct * N + 0.5) / 2)
    if M < 1:
        M = 1

    for i in range(1, M + 1):
        wgt = 0.5 - 0.5 * np.cos(np.pi / M * (i - 0.5))
        x[i - 1] = x[i - 1] * wgt
        x[-i] = x[-i] * wgt

    return x


def mjo_cross(X, Y, segLen, segOverLap, opt=False):
    """
    MJO cross spectrum function. This function calls the above functions to compute
    cross spectral estimates for each segment of length segLen. Segments overlap by
    segOverLap. This function mirrors the NCL routine mjo_cross.
    Return value is a dictionary.
    :param X: Input data 3D array ( time, lat, lon).
    :param Y: Input data 3D array ( time, lat, lon).
    :param segLen: Length of the time segments.
    :param segOverLap: Length of the overlap between time segments.
    :param opt: Optional parameter. Not currently used, set to False.
    :return dict: Dictionary containing the spectral array (STC), frequency array (freq), zonal wavenumber array (wave),
    the number of segments used (nseg), the estimated degrees of freedom (dof), the probability levels (p),
    the coherence squared values corresponding to the probability levels (prob_coh2)
    """

    ntim, nlat, mlon = X.shape
    ntim1, nlat1, mlon1 = Y.shape

    if any([ntim - ntim1, nlat - nlat1, mlon - mlon1]) != 0:
        print("mjo_cross: X and Y must be same size")
        print("           dimX=" + [ntim, nlat, mlon] + "   dimY=" + [ntim1, nlat1, mlon1])
        return

    # make a local copy (time,lat,lon)
    x = X
    y = Y

    # detrend overall series in time, dimension 0
    x = signal.detrend(x, 0)
    y = signal.detrend(y, 0)

    # generate Tukey window, taper 10% of series
    pct = 0.10
    window = window_cosbell(segLen, pct)
    window = np.tile(window, [1, 1, 1])
    window = np.transpose(window)
    window = np.tile(window, [1, nlat, mlon])

    # test if time and longitude are odd or even, fft algorithm
    # returns the Nyquist frequency once for even NT or NL and twice
    # if they are odd
    if (segLen/2+1) % 2 == 1:
        nfreq = int(segLen/2) + 1
    else:
        nfreq = int(segLen/2) + 2
    if mlon % 2 == 1:
        nwave = mlon
    else:
        nwave = mlon + 1

    # initialize spectrum array
    STC = np.zeros([8, nfreq, nwave], dtype='double')
    wave = np.arange(-int(nwave / 2), int(nwave / 2) + 1, 1.)
    freq = np.linspace(0, 0.5, num=nfreq)

    # find time-mean index
    indfreq0 = np.where(freq == 0.)[0]

    # loop through segments and compute cross-spectra
    kseg = 0
    ntStrt = 0
    switch = True
    while switch:
        ntLast = ntStrt + segLen
        if (ntLast > (ntim - 1)):
            switch = False
            continue

        XX = x[ntStrt:ntLast, :, :] * window
        YY = y[ntStrt:ntLast, :, :] * window
        STCseg = mjo_cross_segment_realfft(XX, YY, 0)
        # set time-mean power to NaN
        STCseg[:, indfreq0, :] = np.nan
        # apply 1-2-1 smoother in frequency
        smooth121(STCseg, freq)
        # sum segment spectra
        STC = STC + STCseg

        kseg = kseg + 1
        ntStrt = ntLast + segOverLap - 1

    STC = STC / kseg

    # compute phase and coherence from averaged spectra
    mjo_cross_coh2pha(STC)

    # conservative estimate for DOFs, 2.667 is for 1-2-1 smoother
    dof = 2.667 * kseg
    p = [0.80, 0.85, 0.90, 0.925, 0.95, 0.99]  # probability levels
    prob = p
    prob_coh2 = 1 - (1 - np.power(p, (0.5 * dof - 1)))

    return {'STC': STC, 'freq': freq, 'wave': wave, 'nseg': kseg, 'dof': dof, 'p': prob, 'prob_coh2': prob_coh2}


def kf_filter_mask(fftIn, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName):
    """
    Generate a filtered mask array based on the FFT array and the wave information. Set all values
    outside the specified wave dispersion curves to zero.
    :param fftData: Array of fft coefficients ( wavenumber x freq ), has to be 2 dimensional.
    :param obsPerDay: Number of observations per day.
    :param tMin: Minimum period to include in filtering region.
    :param tMax: Maximum period to include in filtering region.
    :param kMin: Minimum wavenumber to include in filtering region.
    :param kMax: Maximum wavenumber to include in filtering region.
    :param hMin: Minimum equivalent depth to include in filtering region.
    :param hMax: Maximum equivalent depth to include in filtering region.
    :param waveName: Name of the wave to filter for.
    :return: Array containing the fft coefficients of the same size as the input data, with coefficients outside the
    desired region set to zero.
    """
    fftData = np.copy(fftIn)
    fftData = np.transpose(fftData)
    nf, nk = fftData.shape  # frequency, wavenumber array
    fftData = fftData[:, ::-1]

    nt = (nf - 1) * 2
    jMin = int(round(nt / (tMax * obsPerDay)))
    jMax = int(round(nt / (tMin * obsPerDay)))
    jMax = np.array([jMax, nf]).min()

    if kMin < 0:
        iMin = int(round(nk + kMin))
        iMin = np.array([iMin, nk // 2]).max()
    else:
        iMin = int(round(kMin))
        iMin = np.array([iMin, nk // 2]).min()

    if kMax < 0:
        iMax = int(round(nk + kMax))
        iMax = np.array([iMax, nk // 2]).max()
    else:
        iMax = int(round(kMax))
        iMax = np.array([iMax, nk // 2]).min()

    # set the appropriate coefficients outside the frequency range to zero
    # print(fftData[:, 0])
    if jMin > 0:
        fftData[0:jMin, :] = 0
    if jMax < nf:
        fftData[jMax + 1:nf, :] = 0
    if iMin < iMax:
        # Set things outside the wavenumber range to zero, this is more normal
        if iMin > 0:
            fftData[:, 0:iMin] = 0
        if iMax < nk:
            fftData[:, iMax + 1:nk] = 0
    else:
        # Set things inside the wavenumber range to zero, this should be somewhat unusual
        fftData[:, iMax + 1:iMin] = 0

    c = np.empty([2])
    if hMin == -9999:
        c[0] = np.nan
        if hMax == -9999:
            c[1] = np.nan
    else:
        if hMax == -9999:
            c[1] = np.nan
        else:
            c = np.sqrt(g * np.array([hMin, hMax]))

    spc = 24 * 3600. / (2 * pi * obsPerDay)  # seconds per cycle

    # Now set things to zero that are outside the wave dispersion. Loop through wavenumbers
    # and find the limits for each one.
    for i in range(nk):
        if i < (nk / 2):
            # k is positive
            k = i / re
        else:
            # k is negative
            k = -(nk - i) / re

        freq = np.array([0, nf]) / spc
        jMinWave = 0
        jMaxWave = nf
        if (waveName == "Kelvin") or (waveName == "kelvin") or (waveName == "KELVIN"):
            ftmp = k * c
            freq = np.array(ftmp)
        if (waveName == "ER") or (waveName == "er"):
            ftmp = -beta * k / (k ** 2 + 3 * beta / c)
            freq = np.array(ftmp)
        if (waveName == "MRG") or (waveName == "IG0") or (waveName == "mrg") or (waveName == "ig0"):
            if k == 0:
                ftmp = np.sqrt(beta * c)
                freq = np.array(ftmp)
            else:
                if k > 0:
                    ftmp = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                    freq = np.array(ftmp)
                else:
                    ftmp = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                    freq = np.array(ftmp)
        if (waveName == "IG1") or (waveName == "ig1"):
            ftmp = np.sqrt(3 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)
        if (waveName == "IG2") or (waveName == "ig2"):
            ftmp = np.sqrt(5 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)

        if hMin == -9999:
            jMinWave = 0
        else:
            jMinWave = int(np.floor(freq[0] * spc * nt))
        if hMax == -9999:
            jMaxWave = nf
        else:
            jMaxWave = int(np.ceil(freq[1] * spc * nt))
        jMaxWave = np.array([jMaxWave, 0]).max()
        jMinWave = np.array([jMinWave, nf]).min()

        # set appropriate coefficients to zero
        if jMinWave > 0:
            fftData[0:jMinWave, i] = 0
        if jMaxWave < nf:
            fftData[jMaxWave + 1:nf, i] = 0

    fftData = fftData[:, ::-1]
    fftData = np.transpose(fftData)

    return fftData


def kf_filter(data, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName):
    """
    Filter 2D (time x lon) input data for a convectively coupled equatorial wave region in
    wavenumber - frequency space.
    :param Data: Input data ( time x lon ), has to be 2 dimensional.
    :param obsPerDay: Number of observations per day.
    :param tMin: Minimum period to include in filtering region.
    :param tMax: Maximum period to include in filtering region.
    :param kMin: Minimum wavenumber to include in filtering region.
    :param kMax: Maximum wavenumber to include in filtering region.
    :param hMin: Minimum equivalent depth to include in filtering region.
    :param hMax: Maximum equivalent depth to include in filtering region.
    :param waveName: Name of the wave to filter for.
    :return: Array containing the filtered data of the same size as the input data.
    """

    # reorder to (lon x time) to be able to use rfft on the time dimension
    data = np.transpose(data, axes=[1, 0])
    fftdata = np.fft.rfft2(data, axes=(0, 1))

    fftfilt = kf_filter_mask(fftdata, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName)

    datafilt = np.fft.irfft2(fftfilt, axes=(0, 1))
    datafilt = np.transpose(datafilt, axes=[1, 0])

    return datafilt
