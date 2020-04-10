"""
Contains functions to compute space-time power, coherence and phase spectra.
Mirrors the capabilities of NCL.

List of functions:

mjo_cross:
  The main function to compute cross-spectra is called as
  result = mjo_cross(X,Y,nperseg,segoverlap). This splits the data in X and Y
  into segments of length nperseg, the segments overlap by segoverlap.

mjo_cross_segment:

get_symmasymm:

mjo_cross_coh2pha:

smooth121:

smooth121_1D:

window_cosbell:

kf_filter_mask:

"""

import numpy
from scipy import signal

pi = numpy.pi
re = 6.371008e6  # Earth's radius in meters
g = 9.80665  # Gravitational acceleration [m s^{-2}]
omega = 7.292e-05  # Angular speed of rotation of Earth [rad s^{-1}]
beta = 2. * omega / re  # beta parameter at the equator


def mjo_cross_segment(XX, YY, opt=False):
    """
  Compute the FFT to get the power and cross-spectra for one time segment.
  """
    NT, NM, NL = XX.shape
    # compute fourier decomposition in time and longitude
    Xfft = numpy.fft.fft2(XX, axes=(0, 2))
    Yfft = numpy.fft.fft2(YY, axes=(0, 2))
    # normalize by # time samples
    Xfft = Xfft / (NT * NL)
    Yfft = Yfft / (NT * NL)
    # shift 0 frequency and 0 wavenumber to the center
    Xfft = numpy.fft.fftshift(Xfft, axes=(0, 2))
    Yfft = numpy.fft.fftshift(Yfft, axes=(0, 2))

    # average the power spectra across all latitudes
    PX = numpy.average(numpy.square(numpy.abs(Xfft)), axis=1)
    PY = numpy.average(numpy.square(numpy.abs(Yfft)), axis=1)

    # compute co- and quadrature spectrum
    PXY = numpy.average(numpy.conj(Yfft) * Xfft, axis=1)
    CXY = numpy.real(PXY)
    QXY = numpy.imag(PXY)

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
            STC = numpy.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
        else:
            nwave = NL + 1
            STC = numpy.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[:, :, 0] = STC[:, :, NL]
    else:
        nfreq = NT + 1
        if NL % 2 == 1:
            nwave = NL
            STC = numpy.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, :NL] = PX
            STC[1, :NT, :NL] = PY
            STC[2, :NT, :NL] = CXY
            STC[3, :NT, :NL] = QXY
            STC[:, NT, :] = STC[:, 0, :]
        else:
            nwave = NL + 1
            STC = numpy.zeros([8, nfreq, nwave], dtype='double')
            STC[0, :NT, 1:NL + 1] = PX
            STC[1, :NT, 1:NL + 1] = PY
            STC[2, :NT, 1:NL + 1] = CXY
            STC[3, :NT, 1:NL + 1] = QXY
            STC[0, NT, :] = STC[0, 0, :]
            STC[:, :, 0] = STC[:, :, NL]

    return (STC)


def get_symmasymm(X, lat, opt=False):
    """
  Split the data in X into symmetric and anti-symmetric
  (across the equator) parts. Return only the part we are
  interested in.
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
            if opt == 'asymm':
                x = X[:, lat[:] > 0, :]
                if len(lat) % 2 == 1:
                    for ll in range(NM // 2):
                        x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
                else:
                    for ll in range(NM // 2):
                        x[:, ll, :] = 0.5 * (X[:, ll, :] - X[:, NM - ll - 1, :])
    else:
        print("Please provide a valid option: symm or asymm.")

    return (x)


def mjo_cross_coh2pha(STC, opt=False):
    """
  Compute coherence squared and phase spectrum from averaged power and
  cross-spectral estimates.
  """

    nvar, nfreq, nwave = STC.shape

    PX = STC[0, :, :]
    PY = STC[1, :, :]
    CXY = STC[2, :, :]
    QXY = STC[3, :, :]

    PY[PY == 0] = numpy.nan
    COH2 = (numpy.square(CXY) + numpy.square(QXY)) / (PX * PY)
    PHAS = numpy.arctan2(QXY, CXY)

    V1 = -QXY / numpy.sqrt(numpy.square(QXY) + numpy.square(CXY))
    # V1[:,0:nwave//2+1]  = -1*QXY[:,0:nwave//2+1]/numpy.sqrt( numpy.square(QXY[:,0:nwave//2+1])+numpy.square(CXY[:,0:nwave//2+1]) )
    # QXY[:,0:nwave//2+1] = -1*QXY[:,0:nwave//2+1]

    V2 = CXY / numpy.sqrt(numpy.square(QXY) + numpy.square(CXY))

    # STC[3,:,:] = QXY
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
    :type array_in: Numpy array
    :return: STC
    :rtype: Numpy array
    """
    nvar, nfreq, nwave = STC.shape

    # find time-mean index
    indfreqzero = int(numpy.where(freq == 0.)[0])

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
    :type array_in: Numpy array
    :return: array_out
    :rtype: Numpy array
    """

    temp = numpy.copy(array_in)
    array_out = numpy.copy(temp) * 0.0
    weights = numpy.array([1.0, 2.0, 1.0]) / 4.0
    sma = numpy.convolve(temp, weights, 'valid')
    array_out[1:-1] = sma

    # Now its time to correct the borders
    if (numpy.isnan(temp[1])):
        if (numpy.isnan(temp[0])):
            array_out[0] = numpy.nan
        else:
            array_out[0] = temp[0]
    else:
        if (numpy.isnan(temp[0])):
            array_out[0] = numpy.nan
        else:
            array_out[0] = (temp[1] + 3.0 * temp[0]) / 4.0
    if (numpy.isnan(temp[-2])):
        if (numpy.isnan(temp[-1])):
            array_out[-1] = numpy.nan
        else:
            array_out[-2] = array_out[-2]
    else:
        if (numpy.isnan(temp[-1])):
            array_out[-1] = numpy.nan
        else:
            array_out[-1] = (temp[-2] + 3.0 * temp[-1]) / 4.0

    return array_out


def window_cosbell(N, pct, opt=False):
    """
    Compute an equivalent tapering window to the NCL taper function.
    """

    x = numpy.ones(N, dtype='double')
    M = int((pct * N + 0.5) / 2)
    if M < 1:
        M = 1

    for i in range(1, M + 1):
        wgt = 0.5 - 0.5 * numpy.cos(numpy.pi / M * (i - 0.5))
        x[i - 1] = x[i - 1] * wgt
        x[-i] = x[-i] * wgt

    return (x)


def mjo_cross(X, Y, segLen, segOverLap, opt=False):
    """
  MJO cross spectrum function. This function calls the above functions to compute
  cross spectral estimates for each segment of length segLen. Segments overlap by
  segOverLap. This function mirrors the NCL routine mjo_cross.
  Return value is a dictionary.
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
    window = numpy.tile(window, [1, 1, 1])
    window = numpy.transpose(window)
    window = numpy.tile(window, [1, nlat, mlon])

    # test if time and longitude are odd or even, fft algorithm
    # returns the Nyquist frequency once for even NT or NL and twice
    # if they are odd
    if segLen % 2 == 1:
        nfreq = segLen
    else:
        nfreq = segLen + 1
    if mlon % 2 == 1:
        nwave = mlon
    else:
        nwave = mlon + 1

    # initialize spectrum array
    STC = numpy.zeros([8, nfreq, nwave], dtype='double')
    wave = numpy.arange(-int(nwave / 2), int(nwave / 2) + 1, 1.)
    freq = numpy.arange(-1. * int(segLen / 2), 1. * int(segLen / 2) + 1., 1) / (segLen)

    # find time-mean index
    indfreq0 = numpy.where(freq == 0.)[0]

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
        STCseg = mjo_cross_segment(XX, YY, 0)
        # set time-mean power to NaN
        STCseg[:, indfreq0, :] = numpy.nan
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
    prob_coh2 = 1 - (1 - numpy.power(p, (0.5 * dof - 1)))

    return {'STC': STC, 'freq': freq, 'wave': wave, 'nseg': kseg, 'dof': dof, 'p': prob, 'prob_coh2': prob_coh2}


def kf_filter_mask(fftData, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName):
    """
    Generate a filter mask array based on the FFT array and the wave information. Set all values
    outside the specified wave dispersion curves to zero.
    """
    nf, nk = fftData.shape  # frequency, wavenumber array
    nt = (nf - 1) * 2
    jMin = int(round(nt / (tMax * obsPerDay)))
    jMax = int(round(nt / (tMin * obsPerDay)))
    jMax = numpy.array([jMax, nf]).min()

    if kMin < 0:
        iMin = int(round(nk + kMin))
        iMin = numpy.array([iMin, nk // 2]).max()
    else:
        iMin = int(round(kMin))
        iMin = numpy.array([iMin, nk // 2]).min()

    if kMax < 0:
        iMax = int(round(nk + kMax))
        iMax = numpy.array([iMax, nk // 2]).max()
    else:
        iMax = int(round(kMax))
        iMax = numpy.array([iMax, nk // 2]).min()

    # set the appropriate coefficients outside the frequency range to zero
    if jMin > 0:
        fftData[0:jMin, :] = 0
    if jMax < (nf - 1):
        fftData[jMax:, :] = 0
    if iMin < iMax:
        # Set things outside the wavenumber range to zero, this is more normal
        if iMin > 0:
            fftData[:, 0:iMin] = 0
        if iMax < (nk - 1):
            fftData[:, iMax + 1:] = 0
        else:
            # Set things inside the wavenumber range to zero, this should be somewhat unusual
            fftData[:, iMax + 1:iMin] = 0

    c = numpy.sqrt(g * [hMin, hMax])
    spc = 24 * 3600. / (2 * pi * obsPerDay)  # seconds per cycle

    # Now set things to zero that are outside the wave dispersion. Loop through wavenumbers
    # and find the limits for each one.
    for i in range(nk):
        if i < (nk / 2):
            # k is negative
            k = -i / re
        else:
            # k is positive
            k = (nk - i) / re
        freq = numpy.array([0, nf]) / spc
        jMinWave = 0
        jMaxWave = nf
        if ((waveName == "Kelvin") or (waveName == "kelvin") or (waveName == "KELVIN")):
            ftmp = k * c
            freq = numpy.array([ftmp, ftmp])
        if ((waveName == "ER") or (waveName == "er")):
            ftmp = -beta * k / (k ^ 2 + 3. * beta / c)
            freq = numpy.array([ftmp, ftmp])
        if ((waveName == "MRG") or (waveName == "IG0") or (waveName == "mrg") or (waveName == "ig0")):
            if (k == 0):
                ftmp = numpy.sqrt(beta * c)
                freq = numpy.array([ftmp, ftmp])
            else:
                if (k > 0):
                    ftmp = k * c * (0.5 + 0.5 * numpy.sqrt(1 + 4 * beta / (k ^ 2 * c)))
                    freq = numpy.array([ftmp, ftmp])
                else:
                    ftmp = k * c * (0.5 - 0.5 * numpy.sqrt(1 + 4 * beta / (k ^ 2 * c)))
                    freq = numpy.array([ftmp, ftmp])
        if ((waveName == "IG1") or (waveName == "ig1")):
            ftmp = numpy.sqrt(3 * beta * c + k ^ 2 * c ^ 2)
            freq = numpy.array([ftmp, ftmp])
        if ((waveName == "IG2") or (waveName == "ig2")):
            ftmp = numpy.sqrt(5 * beta * c + k ^ 2 * c ^ 2)
            freq = numpy.array([ftmp, ftmp])

        if (hMin == -9999):
            jMinWave = 0
        else:
            jMinWave = int(numpy.floor(freq[0] * spc * nt))
        if (hMax == -9999):
            jMaxWave = nf
        else:
            jMaxWave = int(numpy.ceil(freq[1] * spc * nt))
        jMaxWave = numpy.array([jMaxWave, 0]).max()
        jMinWave = numpy.array([jMinWave, freqDim]).min()
        # set appropriate coefficients to zero
        if (jMinWave > 0):
            fftData[:jMinWave, i] = 0
        if (jMaxWave < (nf - 1)):
            fftData[jMaxWave + 1:, i] = 0

    return fftData
