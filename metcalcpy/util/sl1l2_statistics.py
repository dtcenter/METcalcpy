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
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_fbar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of FBAR")

        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'fbar') / total

        if result is None:
            safe_log(logger, "warning", "FBAR calculation resulted in None")
        else:
            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"FBAR calculated successfully: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during FBAR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_obar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of OBAR")

        total = get_total_values(input_data, columns_names, aggregation)
        result = sum_column_data_by_name(input_data, columns_names, 'obar') / total

        if result is None:
            safe_log(logger, "warning", "OBAR calculation resulted in None")
        else:
            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"OBAR calculated successfully: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during OBAR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fstdev(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of FSTDEV")

        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if aggregation:
            total1 = total
        
        safe_log(logger, "debug", f"Total: {total}, Total1: {total1}, Aggregation: {aggregation}")
        
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total1
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total1

        safe_log(logger, "debug", f"Fbar: {fbar}, FFbar: {ffbar}")

        result = calculate_stddev(fbar * total, ffbar * total, total, logger=logger)
        result = round_half_up(result, PRECISION)

        safe_log(logger, "debug", f"FSTDEV calculated successfully: {result}")

    except (TypeError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during FSTDEV calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ostdev(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of OSTDEV")

        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        if aggregation:
            total1 = total
        safe_log(logger, "debug", f"Total1 value for aggregation: {total1}")

        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total1
        safe_log(logger, "debug", f"Obar calculated: {obar}")

        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total1
        safe_log(logger, "debug", f"Oobar calculated: {oobar}")

        result = calculate_stddev(obar * total, oobar * total, total, logger=logger)
        safe_log(logger, "debug", f"Standard deviation calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final rounded OSTDEV result: {result}")
        
    except (TypeError, Warning, ZeroDivisionError, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during OSTDEV calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fobar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of FOBAR")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        safe_log(logger, "debug", f"Intermediate FOBAR result: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final rounded FOBAR result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during FOBAR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ffbar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of FFBAR")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        safe_log(logger, "debug", f"Intermediate FFBAR result: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final rounded FFBAR result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during FFBAR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_oobar(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of OOBAR")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        safe_log(logger, "debug", f"Intermediate OOBAR result: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final rounded OOBAR result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during OOBAR calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mae(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of MAE")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        result = sum_column_data_by_name(input_data, columns_names, 'mae') / total
        safe_log(logger, "debug", f"Intermediate MAE result: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final rounded MAE result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during MAE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mbias(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of MBIAS")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        safe_log(logger, "debug", f"Calculated OBAR (Observation Mean): {obar}")

        if obar == 0:
            safe_log(logger, "warning", "OBAR is zero, returning None for MBIAS")
            result = None
        else:
            fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
            safe_log(logger, "debug", f"Calculated FBAR (Forecast Mean): {fbar}")

            result = fbar / obar
            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"Final rounded MBIAS result: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during MBIAS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pr_corr(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting calculation of PR_CORR")

        total1 = 1
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total sum calculated: {total}")

        if aggregation:
            total1 = total
            safe_log(logger, "debug", "Aggregation is enabled, setting total1 to total")

        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total1
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total1
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total1
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total1
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total1

        safe_log(logger, "debug", f"Calculated values - FFBar: {ffbar}, FBar: {fbar}, OOBAR: {oobar}, OBAR: {obar}, FOBAR: {fobar}")

        v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) \
            * (total ** 2 * oobar - total ** 2 * obar ** 2)

        if v <= 0:
            safe_log(logger, "warning", "Calculation of variance 'v' resulted in a non-positive value, returning None for PR_CORR")
            pr_corr = None
        else:
            pr_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
            safe_log(logger, "debug", f"Calculated PR_CORR before rounding: {pr_corr}")

            if pr_corr > 1:
                safe_log(logger, "warning", "PR_CORR value exceeds 1, setting PR_CORR to None")
                pr_corr = None
            else:
                pr_corr = round_half_up(pr_corr, PRECISION)
                safe_log(logger, "debug", f"Final rounded PR_CORR result: {pr_corr}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error occurred during PR_CORR calculation: {str(e)}")
        pr_corr = None
    warnings.filterwarnings('ignore')
    return pr_corr


def calculate_fe(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting FE calculation")
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")
        
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        safe_log(logger, "debug", f"fbar calculated: {fbar}")
        
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        safe_log(logger, "debug", f"obar calculated: {obar}")
        
        result = (fbar - obar) / fbar
        safe_log(logger, "debug", f"Fractional error calculated: {result}")
        
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final result after rounding: {result}")
        
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during FE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting ME calculation")
        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")

        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        safe_log(logger, "debug", f"fbar calculated: {fbar}")

        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        safe_log(logger, "debug", f"obar calculated: {obar}")

        result = fbar - obar
        safe_log(logger, "debug", f"Mean error (ME) calculated: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final result after rounding: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during ME calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me2(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting ME2 calculation")
        me = calculate_me(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"ME calculated: {me}")

        if me is not None:
            result = me ** 2
            safe_log(logger, "debug", f"ME squared (ME2) calculated: {result}")

            result = round_half_up(result, PRECISION)
            safe_log(logger, "debug", f"Final ME2 result after rounding: {result}")
        else:
            safe_log(logger, "warning", "ME calculation returned None, so ME2 cannot be calculated")
            result = None
    except (TypeError, Warning) as e:
        safe_log(logger, "warning", f"Exception occurred during ME2 calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mse(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting MSE calculation")

        total = get_total_values(input_data, columns_names, aggregation)
        safe_log(logger, "debug", f"Total values calculated: {total}")

        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total

        safe_log(logger, "debug", f"FFBAR: {ffbar}, OOBAR: {oobar}, FOBAR: {fobar}")

        result = ffbar + oobar - 2 * fobar
        safe_log(logger, "debug", f"Calculated MSE before rounding: {result}")

        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final MSE result after rounding: {result}")

    except (TypeError, Warning) as e:
        safe_log(logger, "warning", f"Exception occurred during MSE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_msess(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting MSESS calculation")

        ostdev = calculate_ostdev(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated OSTDEV: {ostdev}")

        mse = calculate_mse(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated MSE: {mse}")
        result = 1.0 - mse / ostdev ** 2
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final MSESS result after rounding: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        safe_log(logger, "warning", f"Exception occurred during MSESS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rmse(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting RMSE calculation")

        mse = calculate_mse(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated MSE: {mse}")
        result = np.sqrt(calculate_mse(input_data, columns_names, aggregation, logger=logger))
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final RMSE result after rounding: {result}")
    except (TypeError, Warning) as e:
        safe_log(logger, "warning", f"Exception occurred during RMSE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_si(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting SI calculation")

        rmse = calculate_rmse(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated RMSE: {rmse}")

        obar = calculate_obar(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated OBAR: {obar}")
        result = rmse / obar
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final SI result after rounding: {result}")
    except (TypeError, Warning, ZeroDivisionError) as e:
        safe_log(logger, "warning", f"Exception occurred during SI calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_estdev(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting ESTDEV calculation")

        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Calculated total: {total}")

        me = calculate_me(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated ME: {me}")

        mse = calculate_mse(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated MSE: {mse}")
        result = calculate_stddev(me * total, mse * total, total, logger=logger)
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final ESTDEV result after rounding: {result}")
    except (TypeError, Warning, ZeroDivisionError) as e:
        safe_log(logger, "warning", f"Exception occurred during ESTDEV calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcmse(input_data, columns_names, aggregation=False, logger=None):
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
        safe_log(logger, "debug", "Starting BCMSE calculation")

        mse = calculate_mse(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated MSE: {mse}")

        me = calculate_me(input_data, columns_names, aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated ME: {me}")
        result = mse - me ** 2
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final BCMSE result after rounding: {result}")
        if result < 0:
            safe_log(logger, "debug", "BCMSE result is negative, setting to 0.")
            return 0.
    except (TypeError, Warning, ZeroDivisionError) as e:
        safe_log(logger, "warning", f"Exception occurred during BCMSE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcrmse(input_data, columns_names, aggregation=False, logger=None):
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
        result = np.sqrt(calculate_bcmse(input_data, columns_names, aggregation, logger=logger))
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Final BCRMSE result after rounding: {result}")
    except (TypeError, Warning, ZeroDivisionError) as e:
        safe_log(logger, "warning", f"Exception occurred during BCRMSE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_stddev(sum_total, sum_sq, n, logger=None):
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
        safe_log(logger, "warning", f"Invalid number of observations: {n}")
        return None

    safe_log(logger, "debug", f"Calculating variance with sum_total: {sum_total}, sum_sq: {sum_sq}, n: {n}")
    v = (sum_sq - sum_total * sum_total / n) / (n - 1)

    if v < 0:
        safe_log(logger, "warning", f"Calculated variance is negative: {v}")
        return None

    stddev = np.sqrt(v)
    safe_log(logger, "debug", f"Calculated standard deviation: {stddev}")

    return stddev


def calculate_sl1l2_total(input_data, columns_names, logger=None):
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
    try:
        safe_log(logger, "debug", "Calculating total number of matched pairs for Scalar Partial Sums")
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Total number of matched pairs calculated: {result}")
        return result
    except Exception as e:
        safe_log(logger, "warning", f"Exception occurred while calculating total number of matched pairs: {str(e)}")
        return None


def calculate_sal1l2_total(input_data, columns_names, logger=None):
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
    try:
        safe_log(logger, "debug", "Calculating total number of matched pairs for Scalar Anomaly Partial Sums")
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Total number of matched pairs calculated: {result}")
        return result
    except Exception as e:
        safe_log(logger, "warning", f"Exception occurred while calculating total number of matched pairs: {str(e)}")
        return None