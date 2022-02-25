# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: nbrcnt_statistics.py
"""
import warnings

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_nbr_fbs(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_FBS - Fractions Brier Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_FBS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        fbs = sum_column_data_by_name(input_data, columns_names, 'fbs') / total
        result = round_half_up(fbs, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_fss(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_FSS - Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_FSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        fss_den = sum_column_data_by_name(input_data, columns_names, 'fss') / total
        fbs = sum_column_data_by_name(input_data, columns_names, 'fbs') / total
        fss = 1.0 - fbs / fss_den
        result = round_half_up(fss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_afss(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_AFSS - Asymptotic Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_AFSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        f_rate = sum_column_data_by_name(input_data, columns_names, 'f_rate') / total
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total

        afss_num = 2.0 * f_rate * o_rate
        afss_den = f_rate * f_rate + o_rate * o_rate
        afss = afss_num / afss_den
        result = round_half_up(afss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_ufss(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_UFSS - Uniform Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_UFSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total
        ufss = 0.5 + o_rate / 2.0
        result = round_half_up(ufss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_f_rate(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_F_RATE - Forecast event frequency

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_F_RATE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        f_rate = sum_column_data_by_name(input_data, columns_names, 'f_rate') / total
        result = round_half_up(f_rate, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_o_rate(input_data, columns_names, aggregation=False):
    """Performs calculation of NBR_O_RATE - Observed event frequency

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_O_RATE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total
        result = round_half_up(o_rate, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_cnt_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for Neighborhood Continuous Statistics
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
