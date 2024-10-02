# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: sal1l2_statistics.py
"""
import warnings
import numpy as np
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_anom_corr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ANOM_CORR - The Anomaly Correlation
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ANOM_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting calculation of ANOM_CORR")

        total = get_total_values(input_data, columns_names, aggregation)
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffabar') / total
        fbar = sum_column_data_by_name(input_data, columns_names, 'fabar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'ooabar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'oabar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'foabar') / total

        safe_log(logger, "debug", "Intermediate values calculated successfully")

        v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) \
            * (total ** 2 * oobar - total ** 2 * obar ** 2)

        if v <= 0:
            safe_log(logger, "warning", "Variance calculation resulted in a non-positive value")
            return None

        anom_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)

        if anom_corr > 1:
            safe_log(logger, "warning", f"Anomaly correlation exceeded 1: {anom_corr}")
            anom_corr = None
        else:
            anom_corr = round_half_up(anom_corr, PRECISION)

        safe_log(logger, "debug", f"ANOM_CORR calculated successfully: {anom_corr}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during ANOM_CORR calculation: {str(e)}")
        anom_corr = None
    warnings.filterwarnings('ignore')
    return anom_corr


def calculate_anom_corr_raw(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ANOM_CORR_RAW - The Uncentered Anomaly Correlation
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ANOM_CORR_RAW as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting calculation of ANOM_CORR_RAW")

        total = get_total_values(input_data, columns_names, aggregation)
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffabar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'ooabar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'foabar') / total

        safe_log(logger, "debug", "Intermediate values calculated successfully")

        v = ffbar * oobar

        if v < 0:
            safe_log(logger, "warning", "Variance calculation resulted in a negative value")
            return None

        anom_corr_raw = fobar / np.sqrt(v)

        if anom_corr_raw > 1:
            safe_log(logger, "warning", f"Anomaly correlation raw exceeded 1: {anom_corr_raw}")
            anom_corr_raw = 1
        elif anom_corr_raw < -1:
            safe_log(logger, "warning", f"Anomaly correlation raw below -1: {anom_corr_raw}")
            anom_corr_raw = -1

        anom_corr_raw = round_half_up(anom_corr_raw, PRECISION)

        safe_log(logger, "debug", f"ANOM_CORR_RAW calculated successfully: {anom_corr_raw}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during ANOM_CORR_RAW calculation: {str(e)}")
        anom_corr_raw = None
    warnings.filterwarnings('ignore')
    return anom_corr_raw


def calculate_rmsfa(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of RMSFA - Root mean squared forecast anomaly (f-c)
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated RMSFA as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting calculation of RMSFA")

        total = get_total_values(input_data, columns_names, aggregation)
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffabar') / total

        if ffbar is None:
            safe_log(logger, "warning", "FFBAR is None, cannot calculate RMSFA")
            result = None
        elif ffbar < 0:
            safe_log(logger, "warning", f"FFBAR is negative: {ffbar}, cannot calculate RMSFA")
            result = None
        else:
            result = np.sqrt(ffbar)
            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"RMSFA calculated successfully: {result}")

    except (TypeError, Warning) as e:
        safe_log(logger, "warning", f"Exception occurred during RMSFA calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rmsoa(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of RMSOA - Root mean squared observation anomaly (o-c)
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated RMSOA as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting calculation of RMSOA")

        total = get_total_values(input_data, columns_names, aggregation)
        oobar = sum_column_data_by_name(input_data, columns_names, 'ooabar') / total

        if oobar is None:
            safe_log(logger, "warning", "OOBAR is None, cannot calculate RMSOA")
            result = None
        elif oobar < 0:
            safe_log(logger, "warning", f"OOBAR is negative: {oobar}, cannot calculate RMSOA")
            result = None
        else:
            result = np.sqrt(oobar)
            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"RMSOA calculated successfully: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RMSOA calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result