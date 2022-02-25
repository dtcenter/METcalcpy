# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mode_ratio_statistics.py
"""
import warnings
from metcalcpy.util.utils import round_half_up, PRECISION, nrow_column_data_by_name_value

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_ratio_asm_asa(input_data, columns_names):
    """Performs calculation of % of simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsa_asa(input_data, columns_names):
    """Performs calculation of % of simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osa_asa(input_data, columns_names):
    """Performs calculation of % of simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_asu_asa(input_data, columns_names):
    """Performs calculation of % of simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsm_fsa(input_data, columns_names):
    """Performs calculation of % of simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'matched_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsu_fsa(input_data, columns_names):
    """Performs calculation of % of simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osm_osa(input_data, columns_names):
    """Performs calculation of % of simple simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osu_osa(input_data, columns_names):
    """Performs calculation of % of simple simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsm_asm(input_data, columns_names):
    """Performs calculation of % of simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'matched_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osm_asm(input_data, columns_names):
    """Performs calculation of % of simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'matched_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsu_asu(input_data, columns_names):
    """Performs calculation of % of simple unmatched objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'matched_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osu_asu(input_data, columns_names):
    """Performs calculation of % of simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter = {'matched_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsa_aaa(input_data, columns_names):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    ##!!!!!!!! This is the division by the count of all object_id
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osa_aaa(input_data, columns_names):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    ##!!!!!!!! This is the division by the count of all object_id
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsa_faa(input_data, columns_names):
    """Performs calculation of % of all forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fca_faa(input_data, columns_names):
    """Performs calculation of % of all forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osa_oaa(input_data, columns_names):
    """Performs calculation of % of all observation objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_oca_oaa(input_data, columns_names):
    """Performs calculation of % of all observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fca_aca(input_data, columns_names):
    """Performs calculation of % of cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'simple_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_oca_aca(input_data, columns_names):
    """Performs calculation of % of cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'simple_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsa_osa(input_data, columns_names):
    """Performs calculation of Ratio of simple forecasts to simple observations [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osa_fsa(input_data, columns_names):
    """Performs calculation of Ratio of simple observations to simple forecasts [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_aca_asa(input_data, columns_names):
    """Performs calculation of Ratio of cluster objects to simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 0}
    denominator_filter = {'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_asa_aca(input_data, columns_names):
    """Performs calculation of Ratio of simple objects to cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1}
    denominator_filter = {'simple_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fca_fsa(input_data, columns_names):
    """Performs calculation of Ratio of cluster forecast objects to simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_fsa_fca(input_data, columns_names):
    """Performs calculation of Ratio of simple forecast objects to cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 1, 'simple_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_oca_osa(input_data, columns_names):
    """Performs calculation of Ratio of cluster observation objects to simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ratio_osa_oca(input_data, columns_names):
    """Performs calculation of Ratio of simple observation objects to cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1}
    denominator_filter = {'fcst_flag': 0, 'simple_flag': 0}
    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = nrow_column_data_by_name_value(input_data, columns_names, denominator_filter)
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objhits(input_data, columns_names):
    """Performs calculation of Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator = 2
        result = round_half_up(nominator / denominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objmisses(input_data, columns_names):
    """Performs calculation of Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 0, 'simple_flag': 1, 'matched_flag': 0}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        result = round_half_up(nominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objfas(input_data, columns_names):
    """Performs calculation of False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        result = round_half_up(nominator, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objcsi(input_data, columns_names):
    """Performs calculation of CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'matched_flag': 0}
    denominator_filter_1 = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_2 = {'simple_flag': 1, 'matched_flag': 0}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_1 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_1)
        denominator_2 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_2)
        result = round_half_up(nominator / (denominator_1 + 2 * denominator_2), PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objpody(input_data, columns_names):
    """Performs calculation of Probability of Detecting Yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_1 = {'simple_flag': 1, 'matched_flag': 1}
    denominator_filter_2 = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_1 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_1)
        denominator_2 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_2)
        result = round_half_up(nominator / (denominator_1 + 2 * denominator_2), PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_objfar(input_data, columns_names):
    """Performs calculation of False alarm ratio FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    nominator_filter = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter_1 = {'fcst_flag': 1, 'simple_flag': 1, 'matched_flag': 0}
    denominator_filter_2 = {'simple_flag': 1, 'matched_flag': 1}

    try:
        nominator = nrow_column_data_by_name_value(input_data, columns_names, nominator_filter)
        denominator_1 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_1)
        denominator_2 = \
            nrow_column_data_by_name_value(input_data, columns_names, denominator_filter_2)
        result = round_half_up(nominator / (denominator_1 + denominator_2 / 2), PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result
