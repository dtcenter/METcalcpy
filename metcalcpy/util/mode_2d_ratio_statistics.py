# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mode_2d_ratio_statistics.py
"""
from metcalcpy.util.mode_ratio_statistics import *
from metcalcpy.util.utils import column_data_by_name_value, TWO_D_DATA_FILTER
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_2d_ratio_fsa_asa(input_data, columns_names, logger=None):
    """Performs calculation  % of 2d simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating percentage of 2D simple objects that are forecast.")
        result = calculate_ratio_fsa_asa(filtered_data, columns_names, logger=logger)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple objects that are forecast: {str(e)}.")
        return None


def calculate_2d_ratio_osa_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating percentage of 2D simple objects that are observation.")
        result = calculate_ratio_osa_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple objects that are observation: {str(e)}.")
        return None


def calculate_2d_ratio_asm_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating percentage of 2D simple objects that are matched.")
        result = calculate_ratio_asm_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple objects that are matched: {str(e)}.")
        return None


def calculate_2d_ratio_asu_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
        
        safe_log(logger, "debug", "Calculating percentage of 2D simple objects that are unmatched.")
        result = calculate_ratio_asu_asa(filtered_data, columns_names)
        
        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple objects that are unmatched: {str(e)}.")
        return None

def calculate_2d_ratio_fsm_fsa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple forecast objects that are matched.")
        result = calculate_ratio_fsm_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple forecast objects that are matched: {str(e)}.")
        return None


def calculate_2d_ratio_fsu_fsa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple forecast objects that are unmatched.")
        result = calculate_ratio_fsu_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple forecast objects that are unmatched: {str(e)}.")
        return None


def calculate_2d_ratio_osm_osa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple observation objects that are matched.")
        result = calculate_ratio_osm_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple observation objects that are matched: {str(e)}.")
        return None

def calculate_2d_ratio_osu_osa(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple observation objects that are unmatched.")
        result = calculate_ratio_osu_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple observation objects that are unmatched: {str(e)}.")
        return None


def calculate_2d_ratio_fsm_asm(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple matched objects that are forecasts.")
        result = calculate_ratio_fsm_asm(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple matched objects that are forecasts: {str(e)}.")
        return None



def calculate_2d_ratio_osm_asm(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple matched objects that are observations.")
        result = calculate_ratio_osm_asm(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple matched objects that are observations: {str(e)}.")
        return None


def calculate_2d_ratio_fsu_asu(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple unmatched objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple unmatched objects that are forecasts.")
        result = calculate_ratio_fsu_asu(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple unmatched objects that are forecasts: {str(e)}.")
        return None


def calculate_2d_ratio_osu_asu(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating percentage of 2D simple unmatched objects that are observations.")
        result = calculate_ratio_osu_asu(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate percentage of 2D simple unmatched objects that are observations: {str(e)}.")
        return None


def calculate_2d_ratio_fsa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the 2D ratio for FSA against AAA.")
        result = calculate_ratio_fsa_aaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the 2D ratio for FSA against AAA: {str(e)}.")
        return None


def calculate_2d_ratio_osa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of osa to aaa.")
        result = calculate_ratio_osa_aaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of osa to aaa: {str(e)}.")
        return None


def calculate_2d_ratio_fsa_faa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 2d forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
   
        safe_log(logger, "debug", "Calculating the ratio of FSA (Forecast Simple Area) to FAA (All Forecast Area).")
        result = calculate_ratio_fsa_faa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
    
        safe_log(logger, "warning", f"Failed to calculate the ratio of FSA to FAA: {str(e)}.")
        return None


def calculate_2d_ratio_fca_faa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 2d forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of FCA (Forecast Cluster Area) to FAA (All Forecast Area).")
        result = calculate_ratio_fca_faa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of FCA to FAA: {str(e)}.")
        return None


def calculate_2d_ratio_osa_oaa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 2d observation objects that are simple'

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of OSA to OAA.")
        result = calculate_ratio_osa_oaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of OSA to OAA: {str(e)}.")
        return None
    finally:
        warnings.filterwarnings('ignore')


def calculate_2d_ratio_oca_oaa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 2d observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of OSA (Observation Simple Area) to OAA (All Observation Area).")
        result = calculate_ratio_osa_oaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of OSA to OAA: {str(e)}.")
        return None


def calculate_2d_ratio_fca_aca(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fca_aca(filtered_data, columns_names)


def calculate_2d_ratio_oca_aca(input_data, columns_names, logger=None):
    """Performs calculation of % of 2d cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of FCA to ACA.")
        result = calculate_ratio_fca_aca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of FCA to ACA: {str(e)}.")
        return None


def calculate_2d_ratio_fsa_osa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d simple forecasts to 2d simple observations
        [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of FSA to OSA (frequency bias).")
        result = calculate_ratio_fsa_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of FSA to OSA: {str(e)}.")
        return None


def calculate_2d_ratio_osa_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d simple observations to 2d simple forecasts
        [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of OSA to FSA (1 / frequency bias).")
        result = calculate_ratio_osa_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of OSA to FSA: {str(e)}.")
        return None
    finally:
        warnings.filterwarnings('ignore')


def calculate_2d_ratio_aca_asa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of  2d cluster objects to 2d  simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of ACA to ASA.")
        result = calculate_ratio_aca_asa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of ACA to ASA: {str(e)}.")
        return None


def calculate_2d_ratio_asa_aca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d simple objects to 2d cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of ASA to ACA.")
        result = calculate_ratio_asa_aca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of ASA to ACA: {str(e)}.")
        return None


def calculate_2d_ratio_fca_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d cluster forecast objects to 2d simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of FCA to FSA.")
        result = calculate_ratio_fca_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of FCA to FSA: {str(e)}.")
        return None


def calculate_2d_ratio_fsa_fca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d simple forecast objects to 2d cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of FSA to FCA.")
        result = calculate_ratio_fsa_fca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of FSA to FCA: {str(e)}.")
        return None


def calculate_2d_ratio_oca_osa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d cluster observation objects
        to 2d simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of OCA to OSA.")
        result = calculate_ratio_oca_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of OCA to OSA: {str(e)}.")
        return None


def calculate_2d_ratio_osa_oca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 2d simple observation objects to
        2d cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating the ratio of OSA to OCA.")
        result = calculate_ratio_osa_oca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate the ratio of OSA to OCA: {str(e)}.")
        return None


def calculate_2d_objhits(input_data, columns_names, logger=None):
    """Performs calculation of 2d Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
        
    try:    
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object hits.")
        result = calculate_objhits(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object hits: {str(e)}.")
        return None


def calculate_2d_objmisses(input_data, columns_names, logger=None):
    """Performs calculation of 2d Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object misses.")
        result = calculate_objmisses(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object misses: {str(e)}.")
        return None


def calculate_2d_objfas(input_data, columns_names, logger=None):
    """Performs calculation of 2d False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object false alarms.")
        result = calculate_objfas(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object false alarms: {str(e)}.")
        return None


def calculate_2d_objcsi(input_data, columns_names, logger=None):
    """Performs calculation of 2d CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object CSI.")
        result = calculate_objcsi(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object CSI: {str(e)}.")
        return None


def calculate_2d_objpody(input_data, columns_names, logger=None):
    """Performs calculation of 2d Probability of Detecting Yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object PODY.")
        result = calculate_objpody(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object PODY: {str(e)}.")
        return None


def calculate_2d_objfar(input_data, columns_names, logger=None):
    """Performs calculation of False alarm ratio FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on TWO_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 2d object FAR.")
        result = calculate_objfar(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 2d object FAR: {str(e)}.")
        return None