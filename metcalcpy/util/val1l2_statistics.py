# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: val1l2_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values, get_met_version, \
    get_total_dir_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_val1l2_anom_corr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of VAL1L2_ANOM_CORR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VAL1L2_ANOM_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting VAL1L2_ANOM_CORR calculation.")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")

        ufabar = sum_column_data_by_name(input_data, columns_names, 'ufabar') / total
        vfabar = sum_column_data_by_name(input_data, columns_names, 'vfabar') / total
        uoabar = sum_column_data_by_name(input_data, columns_names, 'uoabar') / total
        voabar = sum_column_data_by_name(input_data, columns_names, 'voabar') / total
        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar') / total
        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar') / total
        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar') / total
        safe_log(logger, "debug", f"Summed values: ufabar={ufabar}, vfabar={vfabar}, uoabar={uoabar}, "
                                  f"voabar={voabar}, uvfoabar={uvfoabar}, uvffabar={uvffabar}, uvooabar={uvooabar}")
        result = calc_wind_corr(ufabar, vfabar, uoabar, voabar, uvfoabar, uvffabar, uvooabar, logger=logger)
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final VAL1L2_ANOM_CORR result: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during VAL1L2_ANOM_CORR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calc_wind_corr(uf, vf, uo, vo, uvfo, uvff, uvoo, logger=None):
    """Calculates  wind correlation
        Args:
            uf - Mean(uf-uc)
            vf - Mean(vf-vc)
            uo - Mean(uo-uc)
            vo - Mean(vo-vc)
            uvfo - Mean((uf-uc)*(uo-uc)+(vf-vc)*(vo-vc))
            uvff - Mean((uf-uc)^2+(vf-vc)^2)
            uvoo - Mean((uo-uc)^2+(vo-vc)^2)

        Returns:
                calculated wind correlation as float
                or None if some of the data values are None
        """
    try:
        corr = (uvfo - uf * uo - vf * vo) / (np.sqrt(uvff - uf * uf - vf * vf)
                                             * np.sqrt(uvoo - uo * uo - vo * vo))
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during wind correlation calculation: {str(e)}"):
        corr = None
    return corr


def calculate_val1l2_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for
        Vector Anomaly Partial Sums
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Starting calculation of VAL1L2 total number of matched pairs.")

        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total value before rounding: {total}")

        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Rounded total number of matched pairs: {result}")

        return result

    except Exception as e:
        safe_log(logger, "error", f"Error during calculation of VAL1L2 total number of matched pairs: {str(e)}")
        return None


def calculate_val1l2_total_dir(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for
       well-defined forecast and observation wind directions (TOTAL_DIR column)
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Starting calculation of VAL1L2 total number of matched pairs for wind directions.")

        total = sum_column_data_by_name(input_data, columns_names, 'total_dir')
        safe_log(logger, "debug", f"Total_DIR value before rounding: {total}")

        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Rounded total number of matched pairs for wind directions: {result}")

        return result

    except Exception as e:
        safe_log(logger, "error", f"Error during calculation of VAL1L2 total number of matched pairs for wind directions: {str(e)}")
        return None



def calculate_val1l2_dira_me(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIRA_ME
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dira_me
    """
    try:
        safe_log(logger, "debug", "Starting calculation of DIRA_ME.")

        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values for DIRA_ME calculation: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dira_me') / total

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded DIRA_ME value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of DIRA_ME: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_val1l2_dira_mae(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIRA_MAE
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dira_mae statistic
    """
    try:
        safe_log(logger, "debug", "Starting calculation of DIRA_MAE.")

        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        safe_log(logger, "debug", f"Total direction values for DIRA_MAE calculation: {total}")

        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dira_mae') / total

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Rounded DIRA_MAE value: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation of DIRA_MAE: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result

def calculate_val1l2_dira_mse(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of DIRA_MSE
     Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
        Returns:
            dira_mse statistic
    """
    try:
        total = get_total_dir_values(input_data, np.array(columns_names), aggregation)
        result = sum_column_data_by_name(input_data, np.array(columns_names), 'dira_mse') / total

        result = round_half_up(result, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError, logger=None):
        result = None
    warnings.filterwarnings('ignore')
    return result

