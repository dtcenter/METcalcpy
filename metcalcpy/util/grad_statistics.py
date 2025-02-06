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
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_fgbar(input_data, columns_names, aggregation=False, logger=None):
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

    safe_log(logger, "debug", "Starting calculation of FGBAR.")

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated FGBAR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating FGBAR: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_ogbar(input_data, columns_names, aggregation=False, logger=None):
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

    safe_log(logger, "debug", "Starting calculation of OGBAR.")

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated OGBAR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating OGBAR: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_mgbar(input_data, columns_names, aggregation=False, logger=None):
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

    safe_log(logger, "debug", "Starting calculation of MGBAR.")

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated MGBAR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating MGBAR: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_egbar(input_data, columns_names, aggregation=False, logger=None):
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

    safe_log(logger, "debug", "Starting calculation of EGBAR.")

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated EGBAR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating EGBAR: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_s1(input_data, columns_names, aggregation=False, logger=None):
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

    safe_log(logger, "debug", "Starting calculation of S1 score.")

    try:
        total = get_total_values(input_data, columns_names, aggregation)
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        mgbar = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = 100 * egbar / mgbar
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated S1 score: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating S1 score: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_s1_og(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", f"Total calculated: {total}")
        
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        safe_log(logger, "debug", f"EG Bar calculated: {egbar}")
        
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        safe_log(logger, "debug", f"OG Bar calculated: {ogbar}")
        
        result = 100 * egbar / ogbar
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Result calculated: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error encountered: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fgog_ratio(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", f"Total calculated: {total}")

        fgbar = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        safe_log(logger, "debug", f"FG Bar calculated: {fgbar}")

        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        safe_log(logger, "debug", f"OG Bar calculated: {ogbar}")

        result = 100 * fgbar / ogbar
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"FGOG Ratio calculated: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error encountered: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_grad_total(input_data, columns_names, logger=None):
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
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total number of matched pairs calculated: {total}")
        result = round_half_up(total, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error encountered during calculation: {str(e)}")
        result = None
    return result