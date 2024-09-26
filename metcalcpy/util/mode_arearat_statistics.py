# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mode_arearat_statistics.py
"""
import warnings
from metcalcpy.util.utils import round_half_up, PRECISION, \
    column_data_by_name_value, sum_column_data_by_name
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_arearat_fsa_asa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple objects that are forecast.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_osa_asa(input_data, columns_names, logger):
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
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple objects that are observation.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_asm_asa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple objects that are matched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_asu_asa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple objects that are unmatched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_fsm_fsa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple forecast objects that are matched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_fsu_fsa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple forecast objects that are unmatched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    return result


def calculate_arearat_osm_osa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple observation objects that are matched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osu_osa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple simple observation objects
        that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple observation objects that are unmatched.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fsm_asm(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple matched objects that are forecasts.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osm_asm(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'simple_flag': 1, 'matched_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple matched objects that are observations.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osu_asu(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'simple_flag': 1, 'matched_flag': 0}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of simple unmatched objects that are observation.")
        
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")
        
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")
        
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")
        
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")
        
        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fsa_aaa(input_data, columns_names, logger):
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
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted ? (FSA/AAA).")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator = sum_column_data_by_name(input_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osa_aaa(input_data, columns_names, logger):
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
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted ? (OSA/AAA).")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator = sum_column_data_by_name(input_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fsa_faa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of all forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of all forecast objects that are simple.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fca_faa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of all forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 1}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of all forecast objects that are cluster.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osa_oaa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of all observation objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of all observation objects that are simple.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_oca_oaa(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of all observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 0}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of all observation objects that are cluster.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fca_aca(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'simple_flag': 0}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of cluster objects that are forecast.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_oca_aca(input_data, columns_names, logger):
    """Performs calculation of Area-weighted % of cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'simple_flag': 0}

    try:
        safe_log(logger, "debug", "Starting calculation for Area-weighted % of cluster observation objects.")

        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        safe_log(logger, "debug", f"Nominator data filtered with {nominator_filter}.")

        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Nominator area sum: {nominator}.")

        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        safe_log(logger, "debug", f"Denominator data filtered with {denominator_filter}.")

        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        safe_log(logger, "debug", f"Denominator area sum: {denominator}.")

        result = round_half_up(nominator / denominator, PRECISION)
        safe_log(logger, "debug", f"Calculation result: {result}.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error during calculation: {str(e)}.")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fsa_osa(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of simple forecasts
        to simple observations [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} and {denominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        
        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        
        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        
        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)
        
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osa_fsa(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of simple observations
        to simple forecasts [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} and {denominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)
        
        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')
        
        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')
        
        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)
        
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_aca_asa(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of cluster objects to simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 0}
    denominator_filter = {'simple_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} and {denominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_asa_aca(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of simple objects to cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1}
    denominator_filter = {'simple_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} for nominator and {denominator_filter} for denominator")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fca_fsa(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of cluster forecast objects to simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} for nominator and {denominator_filter} for denominator")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_fsa_fca(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of simple forecast objects to cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} for nominator and {denominator_filter} for denominator")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_oca_osa(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of cluster observation objects to
        simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} for nominator and {denominator_filter} for denominator")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_arearat_osa_oca(input_data, columns_names, logger):
    """Performs calculation of Area Ratio of simple observation objects to
        cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filters to input data: {nominator_filter} for nominator and {denominator_filter} for denominator")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_data = column_data_by_name_value(input_data, columns_names, denominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating sum of denominator data for 'area'")
        denominator = sum_column_data_by_name(denominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating final result")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objahits(input_data, columns_names, logger):
    """Performs calculation of Area Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filter to input data: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        denominator = 2
        safe_log(logger, "debug", f"Calculating final result with denominator = {denominator}")
        result = round_half_up(nominator / denominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objamisses(input_data, columns_names, logger):
    """Performs calculation of Area Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filter to input data: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Rounding the result")
        result = round_half_up(nominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objafas(input_data, columns_names, logger):
    """Performs calculation of Area False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filter to input data: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", "Rounding the result")
        result = round_half_up(nominator, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objacsi(input_data, columns_names, logger):
    """Performs calculation of Area critical success index CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_1 = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_2 = {'simple_flag': 1, 'matched_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filter to input data for nominator: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area' and dividing by 2")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area') / 2

        safe_log(logger, "debug", f"Applying filter to input data for denominator_1: {denominator_filter_1}")
        denominator_1_data = column_data_by_name_value(input_data, columns_names, denominator_filter_1)

        safe_log(logger, "debug", "Calculating sum of denominator_1 data for 'area' and dividing by 2")
        denominator_1 = sum_column_data_by_name(denominator_1_data, columns_names, 'area') / 2

        safe_log(logger, "debug", f"Applying filter to input data for denominator_2: {denominator_filter_2}")
        denominator_2_data = column_data_by_name_value(input_data, columns_names, denominator_filter_2)

        safe_log(logger, "debug", "Calculating sum of denominator_2 data for 'area'")
        denominator_2 = sum_column_data_by_name(denominator_2_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating CSI by dividing nominator by the sum of denominators")
        result = round_half_up(nominator / (denominator_1 + denominator_2), PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objapody(input_data, columns_names, logger):
    """Performs calculation of Area prob of detecting yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_1 = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_2 = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}

    try:
        safe_log(logger, "debug", f"Applying filter to input data for nominator: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", f"Applying filter to input data for denominator_1: {denominator_filter_1}")
        denominator_1_data = column_data_by_name_value(input_data, columns_names, denominator_filter_1)

        safe_log(logger, "debug", "Calculating sum of denominator_1 data for 'area'")
        denominator_1 = sum_column_data_by_name(denominator_1_data, columns_names, 'area')

        safe_log(logger, "debug", f"Applying filter to input data for denominator_2: {denominator_filter_2}")
        denominator_2_data = column_data_by_name_value(input_data, columns_names, denominator_filter_2)

        safe_log(logger, "debug", "Calculating sum of denominator_2 data for 'area'")
        denominator_2 = sum_column_data_by_name(denominator_2_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating PODY by dividing nominator by the sum of denominators")
        result = round_half_up(nominator / (denominator_1 + 2 * denominator_2), PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objafar(input_data, columns_names, logger):
    """Performs calculation of Area FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter_1 = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter_2 = {'simple_flag': 1, 'matched_flag': 1}

    try:
        safe_log(logger, "debug", f"Applying filter to input data for nominator: {nominator_filter}")
        nominator_data = column_data_by_name_value(input_data, columns_names, nominator_filter)

        safe_log(logger, "debug", "Calculating sum of nominator data for 'area'")
        nominator = sum_column_data_by_name(nominator_data, columns_names, 'area')

        safe_log(logger, "debug", f"Applying filter to input data for denominator_1: {denominator_filter_1}")
        denominator_1_data = column_data_by_name_value(input_data, columns_names, denominator_filter_1)

        safe_log(logger, "debug", "Calculating sum of denominator_1 data for 'area'")
        denominator_1 = sum_column_data_by_name(denominator_1_data, columns_names, 'area')

        safe_log(logger, "debug", f"Applying filter to input data for denominator_2: {denominator_filter_2}")
        denominator_2_data = column_data_by_name_value(input_data, columns_names, denominator_filter_2)

        safe_log(logger, "debug", "Calculating sum of denominator_2 data for 'area'")
        denominator_2 = sum_column_data_by_name(denominator_2_data, columns_names, 'area')

        safe_log(logger, "debug", "Calculating FAR by dividing nominator by the sum of denominators")
        result = round_half_up(nominator / (denominator_1 + 2 * denominator_2), PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result
