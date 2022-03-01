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
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_vl1l2_bias(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        bias = np.sqrt(uvffbar) - np.sqrt(uvoobar)
        result = round_half_up(bias, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_fvar(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        f_speed_bar = sum_column_data_by_name(input_data, columns_names, 'f_speed_bar') / total
        result = uvffbar - f_speed_bar * f_speed_bar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_ovar(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        o_speed_bar = sum_column_data_by_name(input_data, columns_names, 'o_speed_bar') / total
        result = uvoobar - o_speed_bar * o_speed_bar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_fspd(input_data, columns_names, aggregation=False):
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


def calculate_vl1l2_ospd(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uobar = sum_column_data_by_name(input_data, columns_names, 'uobar') / total
        vobar = sum_column_data_by_name(input_data, columns_names, 'vobar') / total
        ospd = calc_speed(uobar, vobar)
        result = round_half_up(ospd, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_speed_err(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        speed_bias = calculate_vl1l2_fspd(input_data, columns_names, aggregation) \
                     - calculate_vl1l2_ospd(input_data, columns_names, aggregation)
        result = round_half_up(speed_bias, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_msve(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        uvffbar = sum_column_data_by_name(input_data, columns_names, 'uvffbar') / total
        uvfobar = sum_column_data_by_name(input_data, columns_names, 'uvfobar') / total
        uvoobar = sum_column_data_by_name(input_data, columns_names, 'uvoobar') / total
        msve = uvffbar - 2.0 * uvfobar + uvoobar
        if msve < 0:
            result = None
        else:
            result = round_half_up(msve, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_rmsve(input_data, columns_names, aggregation=False):
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
    warnings.filterwarnings('error')
    try:
        rmsve = np.sqrt(calculate_vl1l2_msve(input_data, columns_names, aggregation))
        result = round_half_up(rmsve, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_vl1l2_total(input_data, columns_names):
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
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return round_half_up(total, PRECISION)
