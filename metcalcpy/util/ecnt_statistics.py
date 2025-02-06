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
from metcalcpy.util.safe_log import safe_log

def calculate_ecnt_crps(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ECNT_CRPS - The Continuous Ranked Probability Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPS as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting ECNT_CRPS calculation.")
    try:
        result = weighted_average(input_data, columns_names, 'crps', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_CRPS calculation completed successfully.")
    except Exception as e:
        safe_log(logger, "error", f"ECNT_CRPS calculation failed due to an error: {e}")
        result = None
    return result


def calculate_ecnt_crpscl(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting ECNT_CRPSCL calculation.")
    try:
        result = weighted_average(input_data, columns_names, 'crpscl', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_CRPSCL calculation completed successfully.")
    except Exception as e:
        safe_log(logger, "error", f"ECNT_CRPSCL calculation failed due to an error: {e}")
        result = None
    return result


def calculate_ecnt_crpss(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting ECNT_CRPSS calculation.")
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crpscl = sum_column_data_by_name(input_data, columns_names, 'crpscl') / total
        crps = sum_column_data_by_name(input_data, columns_names, 'crps') / total
        crpss = 1 - crps / crpscl
        result = round_half_up(crpss, PRECISION)
        safe_log(logger, "info", "ECNT_CRPSS calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_CRPSS calculation failed due to an error: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result

def calculate_ecnt_crps_emp(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_CRPS_EMP calculation.")
    try:
        result = weighted_average(input_data, columns_names, 'crps_emp', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_CRPS_EMP calculation completed successfully.")
    except Exception as e:
        safe_log(logger, "error", f"ECNT_CRPS_EMP calculation failed due to an error: {e}")
        result = None
    return result

def calculate_ecnt_crps_emp_fair(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ECNT_CRPS_EMP_FAIR - The Continuous Ranked Probability Score
        (empirical distribution) adjusted by the mean absolute difference of the ensemble members 

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_CRPS_EMP_FAIR as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_CRPS_EMP_FAIR calculation.")
    try:
        result = weighted_average(input_data, columns_names, 'crps_emp_fair', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_CRPS_EMP_FAIR calculation completed successfully.")
    except Exception as e:
        safe_log(logger, "error", f"ECNT_CRPS_EMP_FAIR calculation failed due to an error: {e}")
        result = None
    return result


def calculate_ecnt_spread_md(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of SPREAD_MD - The pairwise Mean Absolute Difference
        of the unperturbed ensemble members

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SPREAD_MD as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting SPREAD_MD calculation.")
    try:
        result = weighted_average(input_data, columns_names, 'spread_md', aggregation, logger=logger)
        safe_log(logger, "info", "SPREAD_MD calculation completed successfully.")
    except Exception as e:
        safe_log(logger, "error", f"SPREAD_MD calculation failed due to an error: {e}")
        result = None
    return result


def weighted_average(input_data, columns_names, column_name, aggregation=False, logger=None):
    """ Performs aggregation over multiple cases using a weighted average approach,
     where the weight is defined by the number of matched pairs in the TOTAL column

    :param input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
    :param columns_names: names of the columns for the 2nd dimension as Numpy array
    :param column_name: name of the column to be aggregated
    :param aggregation: if the aggregation on fields was performed
    :return: aggregated column values or None if some data values are missing or invalid
    """
    
    safe_log(logger, "debug", f"Starting weighted average calculation for column: {column_name}.")
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        statistic = sum_column_data_by_name(input_data, columns_names, column_name) / total
        result = round_half_up(statistic, PRECISION)
        safe_log(logger, "info", f"Weighted average calculation for column {column_name} completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Weighted average calculation failed for column {column_name} due to error: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result

def calculate_ecnt_crpscl_emp(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_CRPSCL_EMP calculation.")
    
    result = weighted_average(input_data, columns_names, 'crpscl_emp', aggregation, logger=logger)
    
    if result is not None:
        safe_log(logger, "info", "ECNT_CRPSCL_EMP calculation completed successfully.")
    else:
        safe_log(logger, "error", "ECNT_CRPSCL_EMP calculation failed.")

    return result

def calculate_ecnt_crpss_emp(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_CRPSS_EMP calculation.")
    
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        crps_emp = sum_column_data_by_name(input_data, columns_names, 'crps_emp') / total
        crpscl_emp = sum_column_data_by_name(input_data, columns_names, 'crpscl_emp') / total
        crpss_emp =  1 - crps_emp/crpscl_emp
        result = round_half_up(crpss_emp, PRECISION)
        safe_log(logger, "info", "ECNT_CRPSS_EMP calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_CRPSS_EMP calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result

def calculate_ecnt_ign(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ECNT_IGN - The Ignorance Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_IGN as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_IGN calculation.")

    try:
        result = weighted_average(input_data, columns_names, 'ign', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_IGN calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_IGN calculation failed: {str(e)}")
        result = None

    return result


def calculate_ecnt_me(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_ME calculation.")

    try:
        result = weighted_average(input_data, columns_names, 'me', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_ME calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_ME calculation failed: {str(e)}")
        result = None

    return result

def calculate_ecnt_mae(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ECNT_MAE - The Mean Absolute Error of the ensemble mean
        (unperturbed or supplied)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_MAE as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_MAE calculation.")

    try:
        result = weighted_average(input_data, columns_names, 'mae', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_MAE calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_MAE calculation failed: {str(e)}")
        result = None

    return result

def calculate_ecnt_mae_oerr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of MAE_OERR - The Mean Absolute Error of the PERTURBED
        ensemble mean (e.g. with Observation Error)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated MAE_OERR as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting MAE_OERR calculation.")

    try:
        result = weighted_average(input_data, columns_names, 'mae_oerr', aggregation, logger=logger)
        safe_log(logger, "info", "MAE_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"MAE_OERR calculation failed: {str(e)}")
        result = None

    return result


def calculate_ecnt_rmse(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting RMSE calculation.")

    warnings.filterwarnings('error')
    try:
        wa = weighted_average(input_data, columns_names, 'mse', aggregation, logger=logger)
        rmse = math.sqrt(wa)
        result = round_half_up(rmse, PRECISION)
        safe_log(logger, "info", "RMSE calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"RMSE calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_spread(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_SPREAD calculation.")

    warnings.filterwarnings('error')
    try:
        wa = weighted_average(input_data, columns_names, 'variance', aggregation, logger=logger)
        spread = math.sqrt(wa)
        result = round_half_up(spread, PRECISION)
        safe_log(logger, "info", "ECNT_SPREAD calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_SPREAD calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_me_oerr(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_ME_OERR calculation.")

    warnings.filterwarnings('error')
    try:
        result = weighted_average(input_data, columns_names, 'me_oerr', aggregation, logger=logger)
        safe_log(logger, "info", "ECNT_ME_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_ME_OERR calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_rmse_oerr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ECNT_RMSE_OERR - The Root Mean Square Error of the PERTURBED
        ensemble mean (e.g.with Observation Error)
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ECNT_RMSE_OERR as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting ECNT_RMSE_OERR calculation.")

    warnings.filterwarnings('error')
    try:
        wa = weighted_average(input_data, columns_names, 'mse_oerr', aggregation, logger=logger)
        mse_oerr = math.sqrt(wa)
        result = round_half_up(mse_oerr, PRECISION)
        safe_log(logger, "info", "ECNT_RMSE_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_RMSE_OERR calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_spread_oerr(input_data, columns_names, aggregation=False, logger=None):
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
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting ECNT_SPREAD_OERR calculation.")

    warnings.filterwarnings('error')
    try:
        wa = weighted_average(input_data, columns_names, 'variance_oerr', aggregation, logger=logger)
        spread_oerr = math.sqrt(wa)
        result = round_half_up(spread_oerr, PRECISION)
        safe_log(logger, "info", "ECNT_SPREAD_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ECNT_SPREAD_OERR calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_spread_plus_oerr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of SPREAD_PLUS_OERR - The square root of the sum of
        unperturbed ensemble variance and the observation error variance
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated SPREAD_PLUS_OERR as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting SPREAD_PLUS_OERR calculation.")

    warnings.filterwarnings('error')
    try:
        wa = weighted_average(input_data, columns_names, 'variance_plus_oerr', aggregation, logger=logger)
        spread_plus_oerr = math.sqrt(wa)
        result = round_half_up(spread_plus_oerr, PRECISION)
        safe_log(logger, "info", "SPREAD_PLUS_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"SPREAD_PLUS_OERR calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result

def calculate_ecnt_n_ge_obs(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of N_GE_OBS - The number of ensemble values greater
        than or equal to their observations
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated N_GE_OBS as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting N_GE_OBS calculation.")

    warnings.filterwarnings('error')
    try:
        n_ge_obs = sum_column_data_by_name(input_data, columns_names, 'n_ge_obs')
        result = round_half_up(n_ge_obs, PRECISION)
        safe_log(logger, "info", "N_GE_OBS calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"N_GE_OBS calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result

def calculate_ecnt_n_lt_obs(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of N_LT_OBS - The number of ensemble values less
        than their observations
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated N_LT_OBS as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting N_LT_OBS calculation.")

    warnings.filterwarnings('error')
    try:
        n_lt_obs = sum_column_data_by_name(input_data, columns_names, 'n_lt_obs')
        result = round_half_up(n_lt_obs, PRECISION)
        safe_log(logger, "info", "N_LT_OBS calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"N_LT_OBS calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result

def calculate_ecnt_me_ge_obs(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ME_GE_OBS - The Mean Error of the ensemble values
        greater than or equal to their observations
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ME_GE_OBS as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting ME_GE_OBS calculation.")

    warnings.filterwarnings('error')
    try:
        n_ge_obs = sum_column_data_by_name(input_data, columns_names, 'n_ge_obs')
        me_ge_obs = sum_column_data_by_name(input_data, columns_names, 'me_ge_obs') / n_ge_obs
        result = round_half_up(me_ge_obs, PRECISION)
        safe_log(logger, "info", "ME_GE_OBS calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ME_GE_OBS calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result

def calculate_ecnt_me_lt_obs(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of ME_GE_OBS - The Mean Error of the ensemble values
        greater than or equal to their observations
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated ME_GE_OBS as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting ME_LT_OBS calculation.")

    warnings.filterwarnings('error')
    try:
        n_lt_obs = sum_column_data_by_name(input_data, columns_names, 'n_lt_obs')
        me_lt_obs = sum_column_data_by_name(input_data, columns_names, 'me_lt_obs') / n_lt_obs
        result = round_half_up(me_lt_obs, PRECISION)
        safe_log(logger, "info", "ME_LT_OBS calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"ME_LT_OBS calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

def calculate_ecnt_bias_ratio(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of BIAS_RATIO - The Bias Ratio
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated BIAS_RATIO as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting BIAS_RATIO calculation.")

    warnings.filterwarnings('error')
    try:
        me_ge_obs = calculate_ecnt_me_ge_obs(input_data, columns_names, logger=logger)
        me_lt_obs = calculate_ecnt_me_lt_obs(input_data, columns_names, logger=logger)
        if me_lt_obs == 0:
            raise ZeroDivisionError("Division by zero encountered in BIAS_RATIO calculation.")
        
        bias_ratio = me_ge_obs / abs(me_lt_obs)
        result = round_half_up(bias_ratio, PRECISION)
        safe_log(logger, "info", "BIAS_RATIO calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"BIAS_RATIO calculation failed: {str(e)}")
        result = None
    finally:
        warnings.filterwarnings('ignore')

    return result


def calculate_ecnt_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for Ensemble Continuous Statistics
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting Total number of matched pairs calculation.")

    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "info", "Total number of matched pairs calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Total number of matched pairs calculation failed: {str(e)}")
        result = None

    return result

def calculate_ecnt_ign_conv_oerr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of IGN_CONV_OERR - The error-convolved logarithmic scoring
       rule (ignorance score)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated IGN_CONV_OERR as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting IGN_CONV_OERR calculation.")

    try:
        result = weighted_average(input_data, np.array(columns_names), 'ign_conv_oerr', aggregation, logger=logger)
        safe_log(logger, "info", "IGN_CONV_OERR calculation completed successfully.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"IGN_CONV_OERR calculation failed: {str(e)}")
        result = None

    return result


def calculate_ecnt_ign_corr_oerr(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of IGN_CORR_OERR - The error-corrected logarithmic scoring
       rule (ignorance score)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated IGN_CORR_OERR as float
            or None if some data values are missing or invalid
    """

    safe_log(logger, "debug", "Starting calculation of IGN_CORR_OERR.")
    
    try:
        result = weighted_average(input_data, np.array(columns_names), 'ign_corr_oerr', aggregation, logger=logger)
        safe_log(logger, "debug", f"Calculated IGN_CORR_OERR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error calculating IGN_CORR_OERR: {str(e)}")
        result = None
    
    return result


