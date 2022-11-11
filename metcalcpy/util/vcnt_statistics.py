# ============================*
# ** Copyright UCAR (c) 2020
# ** University Corporation for Atmospheric Research (UCAR)
# ** National Center for Atmospheric Research (NCAR)
# ** Research Applications Lab (RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ============================*


"""
Program Name: vcnt_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.met_stats import calc_direction, calc_speed
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'

from metcalcpy.util.vl1l2_statistics import calculate_vl1l2_fvar, calculate_vl1l2_ovar


def calculate_vcnt_fbar(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_FBAR - Mean value of forecast wind speed

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_FBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'f_speed_bar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_obar(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_OBAR - Mean value of observed wind speed

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_OBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'o_speed_bar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fs_rms(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_FS_RMS - Root mean square forecast wind speed

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_FS_RMS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        result = np.sqrt(uvffbar)
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_os_rms(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_OS_RMS - Root mean square observed wind speed

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_OS_RMS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        result = np.sqrt(uvoobar)
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_msve(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_MSVE - Mean squared length of the vector
    difference between the forecast and observed winds

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_MSVE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        uvfobar = sum_column_data_by_name(input_data, columns_names, 'uvfobar') / total
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        mse = uvffbar - 2 * uvfobar + uvoobar
        if mse < 0:
            result = None
        else:
            result = round_half_up(mse, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_rmsve(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_RMSVE - Square root of Mean squared length of the vector
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_RMSVE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        msve = calculate_vcnt_msve(input_data, columns_names, aggregation)
        result = np.sqrt(msve)
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fstdev(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_FSTDEV - Standard deviation of the forecast wind speed
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_FSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_vl1l2_fvar(input_data, columns_names, aggregation))
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_ostdev(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_OSTDEV - Standard deviation of the observed wind speed
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_OSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_vl1l2_ovar(input_data, columns_names, aggregation))
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fdir(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_FDIR - Direction of the average forecast wind vector
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_FDIR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        fdir = calc_direction(-ufbar, -vfbar)
        result = round_half_up(fdir, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_odir(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_ODIR - Direction of the average observed wind vector
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_ODIR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        odir = calc_direction(-uobar, -vobar)
        result = round_half_up(odir, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fbar_speed(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_FBAR_SPEED - Length (speed) of the average forecast wind vector
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_FBAR_SPEED as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        fspd = calc_speed(ufbar, vfbar)
        result = round_half_up(fspd, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_obar_speed(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_OBAR_SPEED - Length (speed) of the average observed wind vector
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_OBAR_SPEED as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        fspd = calc_speed(uobar, vobar)
        result = round_half_up(fspd, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_vdiff_speed(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_VDIFF_SPEED - Length (speed)  of the vector deference between
    the average forecast and average observed wind vectors

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_VDIFF_SPEED as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        vdiff_spd = calc_speed(ufbar - uobar, vfbar - vobar)
        result = round_half_up(vdiff_spd, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_vdiff_dir(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_VDIFF_DIR - Direction of the vector deference between
    the average forecast and average wind vector

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_VDIFF_DIR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        vdiff_dir = calc_direction(-(ufbar - uobar), -(vfbar - vobar))
        result = round_half_up(vdiff_dir, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_speed_err(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_SPEED_ERR - Deference between
        the length of the average forecast wind vector
     and the average observed wind vector (in the sense F - O)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_SPEED_ERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        speed_bias = calculate_vcnt_fbar_speed(input_data, columns_names, aggregation) \
                     - calculate_vcnt_obar_speed(input_data, columns_names, aggregation)
        result = round_half_up(speed_bias, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_speed_abserr(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_SPEED_ABSERR - Absolute value of diference between the length
     of the average forecast wind vector
     and the average observed wind vector (in the sense F - O)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_SPEED_ABSERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        spd_abserr = abs(calculate_vcnt_speed_err(input_data, columns_names, aggregation))
        result = round_half_up(spd_abserr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_dir_err(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_DIR_ERR - Signed angle between the directions
        of the average forecast and observed wind vectors.
        Positive if the forecast wind vector is counter clockwise from the observed wind vector

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_DIR_ERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        f_len = calculate_vcnt_fbar_speed(input_data, columns_names, aggregation)
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        uf = ufbar / f_len
        vf = vfbar / f_len

        o_len = calculate_vcnt_obar_speed(input_data, columns_names)
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        uo = uobar / o_len
        vo = vobar / o_len

        a = vf * uo - uf * vo
        b = uf * uo + vf * vo

        dir_err = calc_direction(a, b)

        result = round_half_up(dir_err, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_dir_abser(input_data, columns_names, aggregation=False):
    """Performs calculation of VCNT_DIR_ABSERR - Absolute value of
        signed angle between the directions of the average forecast
        and observed wind vectors. Positive if the forecast wind vector
        is counter clockwise from the observed wind vector

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VCNT_DIR_ABSERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        ang_btw = abs(calculate_vcnt_dir_err(input_data, columns_names, aggregation))
        result = round_half_up(ang_btw, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_anom_corr(input_data, columns_names, aggregation=False):
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation) # n
        fa_speed_bar = sum_column_data_by_name(input_data, columns_names, 'fa_speed_bar')   # f
        oa_speed_bar = sum_column_data_by_name(input_data, columns_names, 'oa_speed_bar')   # o
        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar')   # ff
        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar')  # fo
        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar')   # oo

        v = (total * uvffabar - fa_speed_bar * fa_speed_bar) * (total * uvooabar - oa_speed_bar * oa_speed_bar)
        result = ((total * uvfoabar) - (fa_speed_bar * oa_speed_bar)) / np.sqrt(v)

        # Check the computed range
        if result > 1:
            result = 1.0
        elif result < -1:
            result = -1.0

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_anom_corr_uncntr(input_data, columns_names, aggregation=False):
    warnings.filterwarnings('error')
    try:
        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar')   # ff
        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar')   # oo
        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar')   # fo

        v = uvffabar * uvooabar
        result = uvfoabar / np.sqrt(v)

        # Check the computed range
        if result > 1:
            result = 1.0
        elif result < -1:
            result = -1.0

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result
