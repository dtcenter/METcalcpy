# ============================*
# ** Copyright UCAR (c) 2020
# ** University Corporation for Atmospheric Research (UCAR)
# ** National Center for Atmospheric Research (NCAR)
# ** Research Applications Lab (RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ============================*


"""
Program Name: vl1l2_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.met_stats import calc_speed
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values, \
    get_total_dir_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_vl1l2_bias(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_BIAS -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_BIAS as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_BIAS.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        safe_log(logger, "debug", f"Calculated uvffbar: {uvffbar}")

        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        safe_log(logger, "debug", f"Calculated uvoobar: {uvoobar}")

        bias = np.sqrt(uvffbar) - np.sqrt(uvoobar)
        safe_log(logger, "debug", f"Calculated bias: {bias}")

        result = round_half_up(bias, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_BIAS.")
    return result

def calculate_vl1l2_fvar(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_FVAR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_FVAR as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_FVAR.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        safe_log(logger, "debug", f"Calculated uvffbar: {uvffbar}")

        f_speed_bar = sum_column_data_by_name(input_data, columns_names, 'f_speed_bar') / total
        safe_log(logger, "debug", f"Calculated f_speed_bar: {f_speed_bar}")

        result = uvffbar - f_speed_bar * f_speed_bar
        safe_log(logger, "debug", f"Calculated forecast variance: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_FVAR.")
    return result


def calculate_vl1l2_ovar(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_OVAR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_OVAR as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_OVAR.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        safe_log(logger, "debug", f"Calculated uvoobar: {uvoobar}")

        o_speed_bar = sum_column_data_by_name(input_data, columns_names, 'o_speed_bar') / total
        safe_log(logger, "debug", f"Calculated o_speed_bar: {o_speed_bar}")

        result = uvoobar - o_speed_bar * o_speed_bar
        safe_log(logger, "debug", f"Calculated observed variance: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_OVAR.")
    return result


def calculate_vl1l2_fspd(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_FSPD -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_FSPD as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_FSPD.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        ufbar = sum_column_data_by_name(input_data, columns_names, 'ufbar') / total
        safe_log(logger, "debug", f"Calculated ufbar: {ufbar}")

        vfbar = sum_column_data_by_name(input_data, columns_names, 'vfbar') / total
        safe_log(logger, "debug", f"Calculated vfbar: {vfbar}")

        fspd = calc_speed(ufbar, vfbar, logger=logger)
        safe_log(logger, "debug", f"Calculated forecast speed: {fspd}")

        result = round_half_up(fspd, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_FSPD.")
    return result


def calculate_vl1l2_ospd(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_OSPD -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_OSPD as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_OSPD.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        safe_log(logger, "debug", f"Calculated uobar: {uobar}")

        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        safe_log(logger, "debug", f"Calculated vobar: {vobar}")

        ospd = calc_speed(uobar, vobar, logger=logger)
        safe_log(logger, "debug", f"Calculated observed speed: {ospd}")

        result = round_half_up(ospd, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_OSPD.")
    return result


def calculate_vl1l2_speed_err(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_SPEED_ERR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_SPEED_ERR as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_SPEED_ERR.")
    warnings.filterwarnings('error')
    try:
        fspd = calculate_vl1l2_fspd(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated forecast speed (VL1L2_FSPD): {fspd}")

        ospd = calculate_vl1l2_ospd(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated observed speed (VL1L2_OSPD): {ospd}")

        speed_bias = fspd - ospd
        safe_log(logger, "debug", f"Calculated speed bias (VL1L2_SPEED_ERR): {speed_bias}")

        result = round_half_up(speed_bias, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_SPEED_ERR.")
    return result


def calculate_vl1l2_msve(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_MSVE -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VL1L2_MSVE as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_MSVE.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total value calculated: {total}")

        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        safe_log(logger, "debug", f"Calculated uvffbar: {uvffbar}")

        uvfobar = sum_column_data_by_name(input_data, columns_names, 'uvfobar') / total
        safe_log(logger, "debug", f"Calculated uvfobar: {uvfobar}")

        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        safe_log(logger, "debug", f"Calculated uvoobar: {uvoobar}")

        msve = uvffbar - 2.0 * uvfobar + uvoobar
        safe_log(logger, "debug", f"Calculated mean squared vector error (MSVE): {msve}")

        if msve < 0:
            safe_log(logger, "warning", f"MSVE is negative, setting result to None.")
            result = None
        else:
            result = round_half_up(msve, PRECISION)
            safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_MSVE.")
    return result


def calculate_vl1l2_rmsve(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VL1L2_RMSVE -
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed
        Returns:
            calculated VL1L2_RMSVE as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of VL1L2_RMSVE.")
    warnings.filterwarnings('error')
    try:
        msve = calculate_vl1l2_msve(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated mean squared vector error (MSVE): {msve}")

        if msve is None:
            safe_log(logger, "warning", "MSVE calculation returned None, RMSVE cannot be calculated.")
            result = None
        else:
            rmsve = np.sqrt(msve)
            safe_log(logger, "debug", f"Calculated root mean squared vector error (RMSVE): {rmsve}")

            result = round_half_up(rmsve, PRECISION)
            safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of VL1L2_RMSVE.")
    return result


def calculate_vl1l2_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for
        Vector Partial Sums
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some of the data values are missing or invalid
    """
    safe_log(logger, "info", "Starting calculation of total number of matched pairs (VL1L2_TOTAL).")
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Calculated total number of matched pairs: {total}")

        result = round_half_up(total, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None

    safe_log(logger, "info", "Finished calculation of VL1L2_TOTAL.")
    return result


def calculate_vl1l2_dir_me(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIR_ME, which was added in MET v12.0
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dir_me
    """
    safe_log(logger, "info", "Starting calculation of DIR_ME.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_me') / total
        safe_log(logger, "debug", f"Raw DIR_ME calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of DIR_ME.")
    return result


def calculate_vl1l2_dir_mae(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIR_MAE
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dir_mae statistic
    """
    safe_log(logger, "info", "Starting calculation of DIR_MAE.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_mae') / total
        safe_log(logger, "debug", f"Raw DIR_MAE calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of DIR_MAE.")
    return result


def calculate_vl1l2_dir_mse(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIR_MSE
     Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dir_mse statistic
    """
    safe_log(logger, "info", "Starting calculation of DIR_MSE.")
    warnings.filterwarnings('error')
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values calculated: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dir_mse') / total
        safe_log(logger, "debug", f"Raw DIR_MSE calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Result rounded to precision {PRECISION}: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    safe_log(logger, "info", "Finished calculation of DIR_MSE.")
    return result