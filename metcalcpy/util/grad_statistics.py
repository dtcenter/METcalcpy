# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: grad_statistics.py
"""
import warnings
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_fgbar(input_data, columns_names, aggregation=False):
    """Performs calculation of FGBAR - Mean of absolute value of forecast gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated FGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ogbar(input_data, columns_names, aggregation=False):
    """Performs calculation of OGBAR - Mean of absolute value of observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated OGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mgbar(input_data, columns_names, aggregation=False):
    """Performs calculation of MGBAR - Mean of maximum of absolute values
        of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated MGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_egbar(input_data, columns_names, aggregation=False):
    """Performs calculation of EGBAR - Mean of absolute value of forecast minus observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated EGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1(input_data, columns_names, aggregation=False):
    """Performs calculation of S1 - S1 score

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated S1 as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        mgbar = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = 100 * egbar / mgbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1_og(input_data, columns_names, aggregation=False):
    """Performs calculation of S1_OG - S1 score with respect to observed gradient

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated S1_OG as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * egbar / ogbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fgog_ratio(input_data, columns_names, aggregation=False):
    """Performs calculation of FGOG_RATIO - Ratio of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated FGOG_RATIO as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        fgbar = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * fgbar / ogbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_grad_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for Gradient partial sums
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
