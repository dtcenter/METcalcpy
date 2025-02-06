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
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values, \
    get_total_dir_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'

from metcalcpy.util.vl1l2_statistics import calculate_vl1l2_fvar, calculate_vl1l2_ovar


def calculate_vcnt_fbar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_FBAR.")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values for VCNT_FBAR calculation: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'f_speed_bar') / total

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_FBAR value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_FBAR: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_obar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_OBAR.")

        total = get_total_dir_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values for VCNT_OBAR calculation: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'o_speed_bar') / total

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_OBAR value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_OBAR: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fs_rms(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_FS_RMS.")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values for VCNT_FS_RMS calculation: {total}")

        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        safe_log(logger, "debug", f"Calculated uvffbar value: {uvffbar}")

        result = np.sqrt(uvffbar)

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_FS_RMS value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_FS_RMS: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_os_rms(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_OS_RMS.")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values for VCNT_OS_RMS calculation: {total}")

        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        safe_log(logger, "debug", f"Calculated uvoobar value: {uvoobar}")

        result = np.sqrt(uvoobar)

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_OS_RMS value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_OS_RMS: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_msve(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_MSVE.")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values for VCNT_MSVE calculation: {total}")

        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        uvfobar = sum_column_data_by_name(input_data, columns_names, 'uvfobar') / total
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total

        safe_log(logger, "debug", f"Calculated uvffbar: {uvffbar}, uvfobar: {uvfobar}, uvoobar: {uvoobar}")

        mse = uvffbar - 2 * uvfobar + uvoobar
        safe_log(logger, "debug", f"Calculated MSE value: {mse}")

        if mse < 0:
            safe_log(logger, "warning", "MSE value is negative, setting result to None.")
            result = None
        else:
            result = round_half_up(mse, PRECISION)
            safe_log(logger, "debug", f"Rounded VCNT_MSVE value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_MSVE: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_rmsve(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_RMSVE.")

        msve = calculate_vcnt_msve(input_data, columns_names, aggregation, logger=logger)
        
        safe_log(logger, "debug", f"Calculated MSVE value: {msve}")
        result = np.sqrt(msve)
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_RMSVE value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_RMSVE: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fstdev(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_FSTDEV.")

        fvar = calculate_vl1l2_fvar(input_data, columns_names, aggregation, logger=logger)
        
        safe_log(logger, "debug", f"Calculated FVAR value: {fvar}")
        result = np.sqrt(fvar)
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_FSTDEV value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_FSTDEV: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_ostdev(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_OSTDEV.")

        ovar = calculate_vl1l2_ovar(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated OVAR value: {ovar}")
        result = np.sqrt(ovar)
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded VCNT_OSTDEV value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_OSTDEV: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fdir(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_FDIR.")

        total = get_total_values(input_data, columns_names, aggregation)

        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        safe_log(logger, "debug", f"Calculated UFBar: {ufbar}, VFBar: {vfbar}")

        fdir = calc_direction(-ufbar, -vfbar, logger=logger)
        result = round_half_up(fdir, PRECISION)
        safe_log(logger, "debug", f"Calculated VCNT_FDIR: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_FDIR: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_odir(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_ODIR.")

        total = get_total_values(input_data, columns_names, aggregation)

        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"Calculated UOBar: {uobar}, VOBar: {vobar}")

        odir = calc_direction(-uobar, -vobar, logger=logger)
        result = round_half_up(odir, PRECISION)
        safe_log(logger, "debug", f"Calculated VCNT_ODIR: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_ODIR: {str(e)}")
        result = Non
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_fbar_speed(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_FBAR_SPEED.")
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        safe_log(logger, "debug", f"Calculated UFBar: {ufbar}, VFBar: {vfbar}")
        fspd = calc_speed(ufbar, vfbar, logger=logger)
        result = round_half_up(fspd, PRECISION)
        safe_log(logger, "debug", f"Calculated VCNT_FBAR_SPEED: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_FBAR_SPEED: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_obar_speed(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of VCNT_OBAR_SPEED.")
        total = get_total_values(input_data, columns_names, aggregation)
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"Calculated UOBar: {uobar}, VOBar: {vobar}")

        fspd = calc_speed(uobar, vobar, logger=logger)
        result = round_half_up(fspd, PRECISION)
        safe_log(logger, "debug", f"Calculated VCNT_OBAR_SPEED: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_OBAR_SPEED: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_vdiff_speed(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "debug", "Starting calculation of VCNT_VDIFF_SPEED.")
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"Calculated UFBAR: {ufbar}, UOBAR: {uobar}, VFBAR: {vfbar}, VOBAR: {vobar}")
        vdiff_spd = calc_speed(ufbar - uobar, vfbar - vobar, logger=logger)
        result = round_half_up(vdiff_spd, PRECISION)
        safe_log(logger, "debug", f"Calculated VCNT_VDIFF_SPEED: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of VCNT_VDIFF_SPEED: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_vdiff_dir(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "info", "Starting calculation of VCNT_VDIFF_DIR.")
    warnings.filterwarnings('error')

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")
        
        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        safe_log(logger, "debug", f"ufbar: {ufbar}")
        
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        safe_log(logger, "debug", f"uobar: {uobar}")
        
        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        safe_log(logger, "debug", f"vfbar: {vfbar}")
        
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"vobar: {vobar}")
        
        vdiff_dir = calc_direction(-(ufbar - uobar), -(vfbar - vobar), logger=logger)
        safe_log(logger, "debug", f"Calculated direction: {vdiff_dir}")
        
        result = round_half_up(vdiff_dir, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    safe_log(logger, "info", "Finished calculation of VCNT_VDIFF_DIR.")
    warnings.filterwarnings('ignore')
    return result


def calculate_vcnt_speed_err(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "info", "Starting calculation of VCNT_SPEED_ERR.")
    warnings.filterwarnings('error')
    try:
        fbar_speed = calculate_vcnt_fbar_speed(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated fbar_speed: {fbar_speed}")
        
        obar_speed = calculate_vcnt_obar_speed(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated obar_speed: {obar_speed}")
        
        speed_bias = fbar_speed - obar_speed
        safe_log(logger, "debug", f"Calculated speed_bias (F - O): {speed_bias}")
        
        result = round_half_up(speed_bias, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_SPEED_ERR.")
    return result


def calculate_vcnt_speed_abserr(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "info", "Starting calculation of VCNT_SPEED_ABSERR.")
    warnings.filterwarnings('error')
    try:
        speed_err = calculate_vcnt_speed_err(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated speed_err: {speed_err}")
        
        spd_abserr = abs(speed_err)
        safe_log(logger, "debug", f"Calculated absolute speed error: {spd_abserr}")
        
        result = round_half_up(spd_abserr, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_SPEED_ABSERR.")
    return result

def calculate_vcnt_dir_err(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_ERR.")
    warnings.filterwarnings('error')
    try:
        f_len = calculate_vcnt_fbar_speed(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated f_len: {f_len}")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        safe_log(logger, "debug", f"ufbar: {ufbar}")

        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        safe_log(logger, "debug", f"vfbar: {vfbar}")

        uf = ufbar / f_len
        vf = vfbar / f_len
        safe_log(logger, "debug", f"Normalized forecast wind vector components: uf={uf}, vf={vf}")

        o_len = calculate_vcnt_obar_speed(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated o_len: {o_len}")

        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        safe_log(logger, "debug", f"uobar: {uobar}")

        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"vobar: {vobar}")

        uo = uobar / o_len
        vo = vobar / o_len
        safe_log(logger, "debug", f"Normalized observed wind vector components: uo={uo}, vo={vo}")

        a = vf * uo - uf * vo
        b = uf * uo + vf * vo
        safe_log(logger, "debug", f"Components a={a}, b={b} for direction calculation")

        dir_err = calc_direction(a, b, logger=logger)
        safe_log(logger, "debug", f"Calculated direction error: {dir_err}")

        result = round_half_up(dir_err, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_ERR.")
    return result


def calculate_vcnt_dir_abser(input_data, columns_names, aggregation=False, logger=None):
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
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_ABSERR.")
    warnings.filterwarnings('error')
    try:
        dir_err = calculate_vcnt_dir_err(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated dir_err: {dir_err}")

        ang_btw = abs(dir_err)
        safe_log(logger, "debug", f"Calculated absolute direction error: {ang_btw}")

        result = round_half_up(ang_btw, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_ABSERR.")
    return result


def calculate_vcnt_anom_corr(input_data, columns_names, aggregation=False, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_ANOM_CORR.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation) # n
        safe_log(logger, "debug", f"Total value calculated: {total}")

        fa_speed_bar = sum_column_data_by_name(input_data, columns_names, 'fa_speed_bar')   # f
        safe_log(logger, "debug", f"Calculated fa_speed_bar: {fa_speed_bar}")

        oa_speed_bar = sum_column_data_by_name(input_data, columns_names, 'oa_speed_bar')   # o
        safe_log(logger, "debug", f"Calculated oa_speed_bar: {oa_speed_bar}")

        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar')   # ff
        safe_log(logger, "debug", f"Calculated uvffabar: {uvffabar}")

        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar')  # fo
        safe_log(logger, "debug", f"Calculated uvfoabar: {uvfoabar}")

        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar')   # oo
        safe_log(logger, "debug", f"Calculated uvooabar: {uvooabar}")

        v = (total * uvffabar - fa_speed_bar * fa_speed_bar) * (total * uvooabar - oa_speed_bar * oa_speed_bar)
        safe_log(logger, "debug", f"Intermediate variable v calculated: {v}")

        result = ((total * uvfoabar) - (fa_speed_bar * oa_speed_bar)) / np.sqrt(v)
        safe_log(logger, "debug", f"Raw anomaly correlation calculated: {result}")

        # Check the computed range
        if result > 1:
            result = 1.0
        elif result < -1:
            result = -1.0
        safe_log(logger, "info", f"Final anomaly correlation result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_ANOM_CORR.")
    return result


def calculate_vcnt_anom_corr_uncntr(input_data, columns_names, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_ANOM_CORR_UNCNTR.")
    warnings.filterwarnings('error')
    try:
        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar')   # ff
        safe_log(logger, "debug", f"Calculated uvffabar: {uvffabar}")

        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar')   # oo
        safe_log(logger, "debug", f"Calculated uvooabar: {uvooabar}")

        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar')   # fo
        safe_log(logger, "debug", f"Calculated uvfoabar: {uvfoabar}")

        v = uvffabar * uvooabar
        safe_log(logger, "debug", f"Intermediate variable v calculated: {v}")

        result = uvfoabar / np.sqrt(v)
        safe_log(logger, "debug", f"Raw uncentered anomaly correlation calculated: {result}")

        # Check the computed range
        if result > 1:
            result = 1.0
        elif result < -1:
            result = -1.0
        safe_log(logger, "info", f"Final uncentered anomaly correlation result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_ANOM_CORR_UNCNTR.")
    return result


def calculate_vcnt_dir_me(input_data, columns_names, aggregation=False, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_ME.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_me') / total
        safe_log(logger, "debug", f"Raw mean error of direction calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_ME.")
    return result

def calculate_vcnt_dir_mae(input_data, columns_names, aggregation=False, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_MAE.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_mae') / total
        safe_log(logger, "debug", f"Raw mean absolute error of direction calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_MAE.")
    return result

def calculate_vcnt_dir_mse(input_data, columns_names, aggregation=False, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_MSE.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_mse') / total
        safe_log(logger, "debug", f"Raw mean squared error of direction calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_MSE.")
    return result


def calculate_vcnt_dir_rmse(input_data, columns_names, aggregation=False, logger=None):
    safe_log(logger, "info", "Starting calculation of VCNT_DIR_RMSE.")
    warnings.filterwarnings('error')
    try:
        mse_result = calculate_vcnt_dir_mse(input_data, np.array(columns_names), aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated mean squared error: {mse_result}")

        result = np.sqrt(mse_result)
        safe_log(logger, "debug", f"Calculated root mean squared error: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VCNT_DIR_RMSE.")
    return result