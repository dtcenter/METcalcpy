# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: ssvar_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.sal1l2_statistics import calculate_anom_corr
from metcalcpy.util.sl1l2_statistics import calculate_fbar, calculate_fstdev, \
    calculate_obar, calculate_ostdev, \
    calculate_pr_corr, calculate_me, calculate_estdev, calculate_mse, \
    calculate_bcmse, calculate_bcrmse, calculate_rmse, \
    calculate_me2, calculate_msess
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_ssvar_fbar(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_FBAR - Average forecast value

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_FBAR as float
            or None if some of the data values are missing or invalid
    """
    return calculate_fbar(input_data, columns_names, aggregation)


def calculate_ssvar_fstdev(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_FSTDEV - Standard deviation of the error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_FSTDEV as float
            or None if some of the data values are missing or invalid
    """
    return calculate_fstdev(input_data, columns_names, aggregation)


def calculate_ssvar_obar(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_OBAR - Average observed value

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_OBAR as float
            or None if some of the data values are missing or invalid
    """
    return calculate_obar(input_data, columns_names, aggregation)


def calculate_ssvar_ostdev(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_OSTDEV - Standard deviation of the error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_OSTDEV as float
            or None if some of the data values are missing or invalid
    """
    return calculate_ostdev(input_data, columns_names, aggregation)


def calculate_ssvar_pr_corr(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_PR_CORR - Pearson correlation coefficient

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_PR_CORR as float
            or None if some of the data values are missing or invalid
    """
    return calculate_pr_corr(input_data, columns_names, aggregation)


def calculate_ssvar_me(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_ME - Mean error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_ME as float
            or None if some of the data values are missing or invalid
    """
    return calculate_me(input_data, columns_names, aggregation)


def calculate_ssvar_estdev(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_ESTDEV - Standard deviation of the error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_ESTDEV as float
            or None if some of the data values are missing or invalid
    """
    return calculate_estdev(input_data, columns_names, aggregation)


def calculate_ssvar_mse(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_MSE - Mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated SSVAR_MSE as float
            or None if some of the data values are missing or invalid
    """
    return calculate_mse(input_data, columns_names, aggregation)


def calculate_ssvar_bcmse(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_BCMSE - Bias corrected root mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_BCMSE as float
            or None if some of the data values are missing or invalid
    """
    return calculate_bcmse(input_data, columns_names, aggregation)


def calculate_ssvar_bcrmse(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_BCRMSE - Bias corrected root mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_BCRMSE as float
            or None if some of the data values are missing or invalid
    """
    return calculate_bcrmse(input_data, columns_names, aggregation)


def calculate_ssvar_rmse(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_RMSE - Root mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_RMSE as float
            or None if some of the data values are missing or invalid
    """
    return calculate_rmse(input_data, columns_names, aggregation)


def calculate_ssvar_anom_corr(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_ANOM_CORR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_ANOM_CORR as float
            or None if some of the data values are missing or invalid
    """
    # change the names to comply with sal1l2 names
    sal1l2_columns_names = np.copy(columns_names)
    sal1l2_columns_names[sal1l2_columns_names  == 'ffbar'] = 'ffabar'
    sal1l2_columns_names[sal1l2_columns_names  == 'fbar'] = 'fabar'
    sal1l2_columns_names[sal1l2_columns_names  == 'oobar'] = 'ooabar'
    sal1l2_columns_names[sal1l2_columns_names  == 'obar'] = 'oabar'
    sal1l2_columns_names[sal1l2_columns_names  == 'fobar'] = 'foabar'
    return calculate_anom_corr(input_data, sal1l2_columns_names, aggregation)


def calculate_ssvar_me2(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_ME2 -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_ME2 as float
            or None if some of the data values are missing or invalid
    """
    return calculate_me2(input_data, columns_names, aggregation)


def calculate_ssvar_msess(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_MSESS -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_MSESS as float
            or None if some of the data values are missing or invalid
    """

    return calculate_msess(input_data, columns_names, aggregation)


def calculate_ssvar_spread(input_data, columns_names, aggregation=False):
    """Performs calculation of SSVAR_SPREAD -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SSVAR_SPREAD as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        var_mean = sum_column_data_by_name(input_data, columns_names, 'var_mean') / total
        result = np.sqrt(var_mean)
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ssvar_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for
        Spread/Skill Variance
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some of the data values are missing or invalid
    """
    total = sum_column_data_by_name(input_data, columns_names, 'total_orig')
    return round_half_up(total, PRECISION)
