# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mode_2d_arearat_statistics.py
"""
from metcalcpy.util.mode_arearat_statistics import *
from metcalcpy.util.utils import column_data_by_name_value, TWO_D_DATA_FILTER
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_2d_arearat_fsa_asa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted statistic using filtered data.")
        result = calculate_arearat_fsa_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted statistic: {str(e)}.")
        return None


def calculate_2d_arearat_osa_asa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted statistic using filtered data.")
        result = calculate_arearat_osa_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted statistic: {str(e)}.")
        return None


def calculate_2d_arearat_asm_asa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of matched objects using filtered data.")
        result = calculate_arearat_asm_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of matched objects: {str(e)}.")
        return None


def calculate_2d_arearat_asu_asa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of unmatched objects using filtered data.")
        result = calculate_arearat_asu_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of unmatched objects: {str(e)}.")
        return None


def calculate_2d_arearat_fsm_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of matched forecast objects using filtered data.")
        result = calculate_arearat_fsm_fsa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of matched forecast objects: {str(e)}.")
        return None


def calculate_2d_arearat_fsu_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of unmatched forecast objects using filtered data.")
        result = calculate_arearat_fsu_fsa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of unmatched forecast objects: {str(e)}.")
        return None


def calculate_2d_arearat_osm_osa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of matched observation objects using filtered data.")
        result = calculate_arearat_osm_osa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of matched observation objects: {str(e)}.")
        return None


def calculate_2d_arearat_osu_osa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of unmatched observation objects using filtered data.")
        result = calculate_arearat_osu_osa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of unmatched observation objects: {str(e)}.")
        return None


def calculate_2d_arearat_fsm_asm(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of matched forecast objects using filtered data.")
        result = calculate_arearat_fsm_asm(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of matched forecast objects: {str(e)}.")
        return None


def calculate_2d_arearat_osm_asm(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of matched observation objects using filtered data.")
        result = calculate_arearat_osm_asm(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of matched observation objects: {str(e)}.")
        return None


def calculate_2d_arearat_osu_asu(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of s2d imple unmatched objects that are observation
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of unmatched observation objects using filtered data.")
        result = calculate_arearat_osu_asu(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of unmatched observation objects: {str(e)}.")
        return None


def calculate_2d_arearat_fsa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of forecast objects against all area using filtered data.")
        result = calculate_arearat_fsa_aaa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of forecast objects against all area: {str(e)}.")
        return None


def calculate_2d_arearat_osa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of observation objects against all area using filtered data.")
        result = calculate_arearat_osa_aaa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of observation objects against all area: {str(e)}.")
        return None


def calculate_2d_arearat_fsa_faa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of all 2d forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of all 2d forecast objects that are simple.")
        result = calculate_arearat_fsa_faa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of all 2d forecast objects that are simple: {str(e)}.")
        return None


def calculate_2d_arearat_fca_faa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of all 2d forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of all 2d forecast objects that are cluster.")
        result = calculate_arearat_fca_faa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of all 2d forecast objects that are cluster: {str(e)}.")
        return None


def calculate_2d_arearat_osa_oaa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of all 2d observation objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of all 2d observation objects that are simple.")
        result = calculate_arearat_osa_oaa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of all 2d observation objects that are simple: {str(e)}.")
        return None


def calculate_2d_arearat_oca_oaa(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of all 2d observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of all 2d observation objects that are cluster.")
        result = calculate_arearat_oca_oaa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of all 2d observation objects that are cluster: {str(e)}.")
        return None


def calculate_2d_arearat_fca_aca(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of 2d cluster objects that are forecast.")
        result = calculate_arearat_fca_aca(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of 2d cluster objects that are forecast: {str(e)}.")
        return None


def calculate_2d_arearat_oca_aca(input_data, columns_names, logger=None):
    """Performs calculation of Area-weighted % of 2d cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating area-weighted % of 2d cluster objects that are observation.")
        result = calculate_arearat_oca_aca(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2D area-weighted % of 2d cluster objects that are observation: {str(e)}.")
        return None


def calculate_2d_arearat_fsa_osa(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d simple forecasts to
        2d simple observations [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
     try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d simple forecasts to 2d simple observations.")
        result = calculate_arearat_fsa_osa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d simple forecasts to 2d simple observations: {str(e)}.")
        return None


def calculate_2d_arearat_osa_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d simple observations to
        2d simple forecasts [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d simple observations to 2d simple forecasts.")
        result = calculate_arearat_osa_fsa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d simple observations to 2d simple forecasts: {str(e)}.")
        return None


def calculate_2d_arearat_aca_asa(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d cluster objects to 2d simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d cluster objects to 2d simple objects.")
        result = calculate_arearat_aca_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d cluster objects to 2d simple objects: {str(e)}.")
        return None


def calculate_2d_arearat_asa_aca(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d simple objects to 2d cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d simple objects to 2d cluster objects.")
        result = calculate_arearat_asa_aca(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d simple objects to 2d cluster objects: {str(e)}.")
        return None

def calculate_2d_arearat_fca_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d cluster forecast objects to
        2d simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d cluster forecast objects to 2d simple forecast objects.")
        result = calculate_arearat_fca_fsa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d cluster forecast objects to 2d simple forecast objects: {str(e)}.")
        return None


def calculate_2d_arearat_fsa_fca(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d simple forecast objects to
        2d cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d simple forecast objects to 2d cluster forecast objects.")
        result = calculate_arearat_fsa_fca(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d simple forecast objects to 2d cluster forecast objects: {str(e)}.")
        return None



def calculate_2d_arearat_oca_osa(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d cluster observation objects to
        2d simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d cluster observation objects to 2d simple observation objects.")
        result = calculate_arearat_oca_osa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d cluster observation objects to 2d simple observation objects: {str(e)}.")
        return None

def calculate_2d_arearat_osa_oca(input_data, columns_names, logger=None):
    """Performs calculation of Area Ratio of 2d simple observation objects to
        2d cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area Ratio of 2d simple observation objects to 2d cluster observation objects.")
        result = calculate_arearat_osa_oca(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area Ratio of 2d simple observation objects to 2d cluster observation objects: {str(e)}.")
        return None


def calculate_2d_objahits(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d Hits.")
        result = calculate_objahits(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d Hits: {str(e)}.")
        return None


def calculate_2d_objamisses(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d Misses (OSU).")
        result = calculate_objamisses(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d Misses (OSU): {str(e)}.")
        return None


def calculate_2d_objafas(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d False Alarms (FSU).")
        result = calculate_objafas(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d False Alarms (FSU): {str(e)}.")
        return None


def calculate_2d_objacsi(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d critical success index CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d Critical Success Index (CSI).")
        result = calculate_objacsi(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d Critical Success Index (CSI): {str(e)}.")
        return None


def calculate_2d_objapody(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d prob of detecting yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d Probability of Detecting Yes (PODY).")
        result = calculate_objapody(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d Probability of Detecting Yes (PODY): {str(e)}.")
        return None

def calculate_2d_objafar(input_data, columns_names, logger=None):
    """Performs calculation of Area 2d FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating Area 2d False Alarm Ratio (FAR).")
        result = calculate_objafar(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate Area 2d False Alarm Ratio (FAR): {str(e)}.")
        return None
