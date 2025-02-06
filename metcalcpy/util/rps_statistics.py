# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: rps_statistics.py
"""
import warnings

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_rps(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of RPS - Ranked Probability Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated RPS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting RPS calculation")
        
        # Retrieve the total number of values, considering aggregation if specified
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")
        
        # Calculate the Ranked Probability Score
        rps = sum_column_data_by_name(input_data, columns_names, 'rps') / total
        result = round_half_up(rps, PRECISION)
        
        safe_log(logger, "debug", f"RPS calculation completed successfully. Result: {result}")
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RPS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rps_comp(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of RPS_COMP - Complement of the Ranked Probability Score
       It is computed simply as RPS_COMP = 1 - RPS

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated RPS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting RPS_COMP calculation")
        
        # Retrieve the total number of values, considering aggregation if specified
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")
        
        # Calculate the Complement of the Ranked Probability Score (RPS_COMP)
        rps_comp = sum_column_data_by_name(input_data, columns_names, 'rps_comp') / total
        result = round_half_up(rps_comp, PRECISION)
        
        safe_log(logger, "debug", f"RPS_COMP calculation completed successfully. Result: {result}")
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RPS_COMP calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rpss(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of RPSS -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated RPS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Starting RPSS calculation")
        
        # Retrieve the total number of values, considering aggregation if specified
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")
        
        # Calculate the Ranked Probability Score (RPS)
        rps = sum_column_data_by_name(input_data, columns_names, 'rps') / total
        safe_log(logger, "debug", f"RPS calculated: {rps}")
        
        # Calculate the climatological Ranked Probability Score (RPS_CLIMO)
        rps_climo = sum_column_data_by_name(input_data, columns_names, 'rps_climo') / total
        safe_log(logger, "debug", f"RPS_CLIMO calculated: {rps_climo}")
        
        # Calculate the Ranked Probability Skill Score (RPSS)
        rpss = 1 - rps / rps_climo
        result = round_half_up(rpss, PRECISION)
        
        safe_log(logger, "debug", f"RPSS calculation completed successfully. Result: {result}")
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RPSS calculation: {str(e)}")
        result = None
    return result


def calculate_rps_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for
        Ranked Probability Score Statistics
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
        safe_log(logger, "debug", "Starting calculation of total matched pairs for RPS statistics")

        # Calculate the total number of matched pairs
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)

        safe_log(logger, "debug", f"Total matched pairs calculated successfully. Result: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RPS total calculation: {str(e)}")
        result = None

    return result