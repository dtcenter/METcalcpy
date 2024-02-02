# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: sl1l2_statistics.py
"""
import warnings
import numpy as np
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_fbar(input_data, columns_names, aggregation=False):
    """Performs calculation of FBAR - Forecast mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated FBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_obar(input_data, columns_names, aggregation=False):
    """Performs calculation of OBAR - Observation Mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated OBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fstdev(input_data, columns_names, aggregation=False):
    """Performs calculation of FSTDEV - Forecast standard deviation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated FSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if aggregation:
            total1 = total
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total1
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total1
        result = calculate_stddev(fbar * total, ffbar * total, total)
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ostdev(input_data, columns_names, aggregation=False):
    """Performs calculation of OSTDEV - Observation Standard Deviation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated OSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if aggregation:
            total1 = total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total1
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total1
        result = calculate_stddev(obar * total, oobar * total, total)
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fobar(input_data, columns_names, aggregation=False):
    """Performs calculation of FOBAR - Average product of forecast and observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated FOBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ffbar(input_data, columns_names, aggregation=False):
    """Performs calculation of FFBAR - Average of forecast squared

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated FFBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_oobar(input_data, columns_names, aggregation=False):
    """Performs calculation of OOBAR - Average of observation squared

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated OOBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mae(input_data, columns_names, aggregation=False):
    """Performs calculation of MAE - Mean absolute error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated MAE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'mae') / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mbias(input_data, columns_names, aggregation=False):
    """Performs calculation of MBIAS - Multiplicative Bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated MBIAS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        if obar == 0:
            result = None
        else:
            fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
            result = fbar / obar
            result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pr_corr(input_data, columns_names, aggregation=False):
    """Performs calculation of PR_CORR - Pearson correlation coefficient
        including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated PR_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if aggregation:
            total1 = total
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total1
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total1
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total1
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total1
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total1
        v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) \
            * (total ** 2 * oobar - total ** 2 * obar ** 2)
        pr_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
        if v <= 0 or pr_corr > 1:
            pr_corr = None
        else:
            pr_corr = round_half_up(pr_corr, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        pr_corr = None
    warnings.filterwarnings('ignore')
    return pr_corr


def calculate_fe(input_data, columns_names, aggregation=False):
    """Performs calculation of FE - Fractional error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated FE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        result = (fbar - obar) / fbar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me(input_data, columns_names, aggregation=False):
    """Performs calculation of ME - Mean error, aka Additive bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ME as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        result = fbar - obar
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me2(input_data, columns_names, aggregation=False):
    """Performs calculation of ME2 - The square of the mean error (bias)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ME2 as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        me = calculate_me(input_data, columns_names, aggregation)
        result = me ** 2
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mse(input_data, columns_names, aggregation=False):
    """Performs calculation of MSE - Mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated MSE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        result = ffbar + oobar - 2 * fobar
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_msess(input_data, columns_names, aggregation=False):
    """Performs calculation of MSESS - The mean squared error skill score

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated MSESS as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        ostdev = calculate_ostdev(input_data, columns_names, aggregation)
        mse = calculate_mse(input_data, columns_names, aggregation)
        result = 1.0 - mse / ostdev ** 2
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rmse(input_data, columns_names, aggregation=False):
    """Performs calculation of RMSE - Root-mean squared error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated RMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_mse(input_data, columns_names, aggregation))
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_si(input_data, columns_names, aggregation=False):
    """Performs calculation of SI - Scatter Index

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated SI as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        rmse = calculate_rmse(input_data, columns_names, aggregation)
        obar = calculate_obar(input_data, columns_names, aggregation)
        result = rmse / obar
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_estdev(input_data, columns_names, aggregation=False):
    """Performs calculation of ESTDEV - Standard deviation of the error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ESTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        me = calculate_me(input_data, columns_names)
        mse = calculate_mse(input_data, columns_names)
        result = calculate_stddev(me * total, mse * total, total)
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcmse(input_data, columns_names, aggregation=False):
    """Performs calculation of BCMSE - Bias-corrected mean squared error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated BCMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        mse = calculate_mse(input_data, columns_names, aggregation)
        me = calculate_me(input_data, columns_names, aggregation)
        result = mse - me ** 2
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcrmse(input_data, columns_names, aggregation=False):
    """Performs calculation of BCRMSE - Bias-corrected root mean square error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array
                aggregation: if the aggregation on fields was performed

            Returns:
                calculated BCRMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_bcmse(input_data, columns_names, aggregation))
        result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_stddev(sum_total, sum_sq, n):
    """Performs calculation of STDDEV - Standard deviation

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated STDDEV as float
                or None if some of the data values are missing or invalid
    """
    if n < 1:
        return None
    v = (sum_sq - sum_total * sum_total / n) / (n - 1)
    if v < 0:
        return None

    return np.sqrt(v)


def calculate_sl1l2_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for
        Scalar Partial Sums
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


def calculate_sal1l2_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for
        Scalar Anomaly Partial Sums
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
