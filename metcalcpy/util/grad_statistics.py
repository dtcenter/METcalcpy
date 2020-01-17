"""
Program Name: grad_statistics.py
"""
import warnings
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


def calculate_fgbar(input_data, columns_names):
    """Performs calculation of FGBAR - Mean of absolute value of forecast gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated FGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ogbar(input_data, columns_names):
    """Performs calculation of OGBAR - Mean of absolute value of observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated OGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mgbar(input_data, columns_names):
    """Performs calculation of MGBAR - Mean of maximum of absolute values
        of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated MGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_egbar(input_data, columns_names):
    """Performs calculation of EGBAR - Mean of absolute value of forecast minus observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated EGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1(input_data, columns_names):
    """Performs calculation of S1 - S1 score

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated S1 as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        mgbar = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = 100 * egbar / mgbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1_og(input_data, columns_names):
    """Performs calculation of S1_OG - S1 score with respect to observed gradient

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated S1_OG as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * egbar / ogbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fgog_ratio(input_data, columns_names):
    """Performs calculation of FGOG_RATIO - Ratio of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated FGOG_RATIO as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fgbar = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * fgbar / ogbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result
