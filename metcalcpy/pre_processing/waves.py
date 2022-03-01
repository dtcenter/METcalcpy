# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** CIRES, Regents of the University of Colorado
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
import numpy as np
import xarray as xr
import scipy
import xrft


def zonal_wave_coeffs(dat, dimvar='longitude', *, waves=None, fftpkg='scipy'):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.
    This is a primarily a driver function that shifts the data depending
    on the specified fftpkg.
    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension over which to compute coefficients that spans 
        all 360 degrees
    dimvar: Name of the dimension to compute the wave coefficients.  Longitude is the
        default if it's not specified
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension, for specified waves only.
    """

    if fftpkg not in ['scipy', 'xrft']:
        msg = 'fftpkg keyword arg must be one of scipy or xarray'
        raise ValueError(msg)

    funcs = {
        'scipy': _zonal_wave_coeffs_scipy,
        'xrft': _zonal_wave_coeffs_xrft
    }

    nlons = dat[dimvar].size

    fc = funcs[fftpkg](dat, dimvar)

    fc.attrs['nlons'] = nlons
    fc.attrs['lon0'] = dat[dimvar].values[0]
    if (waves is not None):
        fc = fc.sel(lon_wavenum=waves)

    return fc


def _zonal_wave_coeffs_scipy(dat, dimvar):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.
    Uses scipy.fft.rfft to perform the calculation.
    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension over which to compute coefficients that spans
        all 360 degrees
    dimvar: Name of the dimension to compute the wave coefficients.
    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.
    """
    nlons = dat[dimvar].size
    lon_ax = dat.get_axis_num(dimvar)

    new_dims = list(dat.dims)
    new_dims[lon_ax] = 'lon_wavenum'

    new_coords = dict(dat.coords)
    new_coords.pop(dimvar)
    new_coords['lon_wavenum'] = np.arange(0, nlons//2 + 1)

    fc = scipy.fft.rfft(dat.values, axis=lon_ax)
    fc = xr.DataArray(fc, coords=new_coords, dims=new_dims)

    return fc


def _zonal_wave_coeffs_xrft(dat, dimvar):
    r"""Calculate the Fourier coefficients of waves in the zonal direction.
    Uses xrft.fft to perform the calculation.
    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension over which to compute coefficients that spans
        all 360 degrees
    dimvar: Name of the dimension to compute the wave coefficients.
    Returns
    -------
    `xarray.DataArray`
        Output of the rFFT along the longitude dimension.
    """

    fc = xrft.fft(dat, dim=dimvar, real_dim=dimvar,
                  true_phase=False, true_amplitude=False)

    fc = fc.rename({'freq_longitude': 'lon_wavenum'})
    fc = fc.assign_coords({'lon_wavenum': np.arange(fc.lon_wavenum.size)})

    return fc


def zonal_wave_ampl_phase(dat, dimvar='longitude', waves=None, phase_deg=False, fftpkg='scipy'):
    r"""Calculates the amplitudes and relative phases of waves in the zonal direction.
    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension to compute wave amplitudes and phases that spans 
        all 360 degrees
    dimvar: Name of the dimension to compute the amplitude and phase.  Longitude is the
        default if it's not specified
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    phase_deg : boolean, optional
        Whether to return the relative phases in radians or degrees.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
    Returns
    -------
    Tuple of two `xarray.DataArray`
        Tuple contains (amplitudes, phases)
    See Also
    --------
    zonal_wave_coeffs
    """

    fc = zonal_wave_coeffs(dat, dimvar, waves=waves, fftpkg=fftpkg)

    # where the longitudinal wavenumber is 0, `where' will
    # mask to nan, so np.isfinite will return False in those
    # spots and true everywhere else. Thus, add 1 to get
    # the multiplying mask that keeps in mind the "zeroth"
    # mode (which would be the zonal mean, if kept)
    #
    # this is necessary because of the symmetric spectrum,
    # so all other wavenumbers except the 0th need to
    # be multipled by 2 to get the right amplitude
    mult_mask = np.isfinite(fc.where(fc.lon_wavenum != 0)) + 1

    ampl = mult_mask*np.abs(fc) / fc.nlons
    phas = np.angle(fc, deg=phase_deg)

    return (ampl.astype(dat.dtype), phas.astype(dat.dtype))


def zonal_wave_contributions(dat, dimvar='longitude', waves=None, fftpkg='scipy'):
    r"""Computes contributions of waves with zonal wavenumber k to the input field.
    Parameters
    ----------
    dat : `xarray.DataArray`
        data containing a dimension to compute wave contributions that spans all 
        360 degrees
    dimvar: Name of the dimension to compute the contributions of waves.  Longitude 
        is the default if it's not specified
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
    Returns
    -------
    `xarray.DataArray`
    See Also
    --------
    zonal_wave_coeffs
    """
    fc = zonal_wave_coeffs(dat, dimvar, waves=waves, fftpkg=fftpkg)

    if (waves is None):
        waves = fc.lon_wavenum.values

    recons = []
    if (fftpkg == 'scipy'):
        new_dims = list(dat.dims)
        new_dims += ['lon_wavenum']
        new_coords = dict(dat.coords)
        new_coords['lon_wavenum'] = waves

        for k in waves:
            mask = np.isnan(fc.where(fc.lon_wavenum != k))

            kcont = scipy.fft.irfft((fc*mask).values, axis=fc.get_axis_num('lon_wavenum'))
            recons.append(kcont[..., np.newaxis])

        recons = np.concatenate(recons, axis=-1)
        recons = xr.DataArray(recons, dims=new_dims, coords=new_coords)

    elif (fftpkg == 'xarray'):
        fc = fc.rename({'lon_wavenum': 'freq_longitude'})

        for k in waves:
            mask = np.isnan(fc.where(fc.lon_wavenum != k))

            kcont = xrft.ifft((fc*mask).values, dim='lon_wavenum', real_dim='lon_wavenum')
            recons.append(kcont)

        recons = xr.concat(recons, dim='lon_wavenum')
        recons = recons.assign_coords({'lon_wavenum': waves, dimvar: dat[dimvar]})

    return recons.astype(dat.dtype)


def zonal_wave_covariance(dat1, dat2, dimvar1='longitude', dimvar2='longitude', waves=None, fftpkg='scipy'):
    r"""Calculates the covariance of two fields partititioned into zonal wavenumbers.
    Parameters
    ----------
    dat1 : `xarray.DataArray`
        field containing a dimension over which to compute wave covariance that 
        spans all 360 degrees.  Should have the same shape as dat2.
    dat2 : `xarray.DataArray`
        another field also containing a dimension over which to compute wave covariance 
        that spans all 360 degrees. Should have the same shape as dat1.
    dimvar1: Name of the dimension to compute the wave covariance for dat1.  Longitude 
        is the default if it's not specified
    dimvar2: Name of the dimension to compute the wave ccovariance for dat2.  Longitude 
        is the default if it's not specified
    waves : array-like, optional
        The zonal wavenumbers to maintain in the output. Defaults to None for all.
    fftpkg : string, optional
        String that specifies how to perform the FFT on the data. Options are
        scipy or xrft. Specifying scipy uses some operations that are memory-eager
        and leverages scipy.fft.rfft. Specifying xrft should leverage the benefits
        of xarray/dask for large datasets by using xrft.fft. Defaults to scipy.
    Returns
    -------
    `xarray.DataArray`
    See Also
    --------
    zonal_wave_coeffs
    TO DO
    -----
    * Check for consistency between dat1 and dat2 and throw errors
    """

    nlons = dat1[dimvar].size

    fc1 = zonal_wave_coeffs(dat1, dimvar1, waves=waves, fftpkg=fftpkg)
    fc2 = zonal_wave_coeffs(dat2, dimvar2, waves=waves, fftpkg=fftpkg)

    mult_mask = np.isfinite(fc1.where(fc1.lon_wavenum != 0)) + 1
    cov = mult_mask*np.real(fc1 * fc2.conj())/(nlons**2)

    return cov
