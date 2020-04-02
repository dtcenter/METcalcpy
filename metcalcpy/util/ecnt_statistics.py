"""
Program Name: ecnt_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


def calculate_ecnt_crps(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPS - The Continuous Ranked Probability Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crps = sum_column_data_by_name(input_data, columns_names, 'crps') / total
        result = round_half_up(crps, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_crpss(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPSS - The Continuous Ranked Probability Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crps_climo = sum_column_data_by_name(input_data, columns_names, 'crps_climo') / total
        crps = sum_column_data_by_name(input_data, columns_names, 'crps') / total
        crpss = (crps_climo - crps) / crps_climo
        result = round_half_up(crpss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_ign(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_IGN - The Ignorance Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_IGN as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ign = sum_column_data_by_name(input_data, columns_names, 'ign') / total
        result = round_half_up(ign, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_me(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_ME - The Mean Error of the ensemble mean
    (unperturbed or supplied)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_ME as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        me = sum_column_data_by_name(input_data, columns_names, 'me') / total
        result = round_half_up(me, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_rmse(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_RMSE - The Root Mean Square Error of the ensemble mean
    (unperturbed or supplied)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_RMSE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        mse = sum_column_data_by_name(input_data, columns_names, 'mse') / total
        result = round_half_up(mse, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_spread(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_SPREAD - The mean of the spread (standard deviation)
        of the unperturbed ensemble member values at each observation location

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_SPREAD as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        spread = sum_column_data_by_name(input_data, columns_names, 'spread') / total
        result = round_half_up(spread, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_me_oerr(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_ME_OERR - The Mean Error of the PERTURBED ensemble mean
        (e.g. with Observation Error)
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_ME_OERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        me_oerr = sum_column_data_by_name(input_data, columns_names, 'me_oerr') / total
        result = round_half_up(me_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_rmse_oerr(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_RMSE_OERR - TheRoot Mean Square Error of the PERTURBED
        ensemble mean (e.g.with Observation Error)
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_RMSE_OERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        mse_oerr = sum_column_data_by_name(input_data, columns_names, 'mse_oerr') / total
        rmse_oerr = np.sqrt(mse_oerr)
        result = round_half_up(rmse_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_spread_oerr(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_SPREAD_OERR - The mean of the spread (standard deviation)
        of the PERTURBED ensemble member values (e.g. with Observation Error )
        at each observation location

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_SPREAD_OERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        spread_oerr = sum_column_data_by_name(input_data, columns_names, 'spread_oerr') / total
        result = round_half_up(spread_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_spread_plus_oerr(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_SPREAD_PLUS_OERR - The square root of the sum of
        unperturbed ensemble variance and the observation error variance
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_SPREAD_PLUS_OERR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        spread_plus_oerr = sum_column_data_by_name(input_data, columns_names, 'spread_plus_oerr') \
                           / total
        result = round_half_up(spread_plus_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for Ensemble Continuous Statistics
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
