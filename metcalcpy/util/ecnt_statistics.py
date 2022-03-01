# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: ecnt_statistics.py
"""
import warnings
import numpy as np
import math

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_crpscl(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPSCL - Climatological Continuous Ranked Probability Score
        (normal distribution)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPSCL as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crpscl = sum_column_data_by_name(input_data, columns_names, 'crpscl') / total
        result = round_half_up(crpscl, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ecnt_crpss(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPSS - The Continuous Ranked Probability Skill Score
        (normal distribution)

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
        crpscl = sum_column_data_by_name(input_data, columns_names, 'crpscl') / total
        crps = sum_column_data_by_name(input_data, columns_names, 'crps') / total
        crpss = 1 - crps / crpscl
        result = round_half_up(crpss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result

def calculate_ecnt_crps_emp(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPS_EMP - The Continuous Ranked Probability Score
        (empirical distribution)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPS_EMP as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crps_emp = sum_column_data_by_name(input_data, columns_names, 'crps_emp') / total
        result = round_half_up(crps_emp, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result

def calculate_ecnt_crpscl_emp(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPSCL_EMP - Climatological Continuous Ranked Probability Score
        (empirical distribution)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPSCL_EMP as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crpscl_emp = sum_column_data_by_name(input_data, columns_names, 'crpscl_emp') / total
        result = round_half_up(crpscl_emp, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result

def calculate_ecnt_crpss_emp(input_data, columns_names, aggregation=False):
    """Performs calculation of ECNT_CRPSS_EMP - The Continuous Ranked Probability Skill Score
        (empirical distribution)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPSS_EMP as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crps_emp = sum_column_data_by_name(input_data, columns_names, 'crps_emp') / total
        crpscl_emp = sum_column_data_by_name(input_data, columns_names, 'crpscl_emp') / total
        crpss_emp =  1 - crps_emp/crpscl_emp
        result = round_half_up(crpss_emp, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
        rmse = math.sqrt(sum_column_data_by_name(input_data, columns_names, 'mse') / total)
        result = round_half_up(rmse, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
        spread = math.sqrt(sum_column_data_by_name(input_data, columns_names, 'variance') / total)
        result = round_half_up(spread, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
        spread_oerr = math.sqrt(sum_column_data_by_name(input_data, columns_names, 'variance_oerr') / total)
        result = round_half_up(spread_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
        spread_plus_oerr = math.sqrt(sum_column_data_by_name(input_data, columns_names, 'variance_plus_oerr') / total)
        result = round_half_up(spread_plus_oerr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
