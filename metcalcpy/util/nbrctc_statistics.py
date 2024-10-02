# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: nbrctc_statistics.py
"""

from metcalcpy.util.ctc_statistics import calculate_baser, calculate_acc, calculate_fbias, \
    calculate_fmean, calculate_pody, calculate_pofd, calculate_podn, calculate_far, calculate_csi, \
    calculate_gss, calculate_hk, calculate_hss, calculate_odds
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_nbr_baser(input_data, columns_names, logger=None):
    """Performs calculation of NBR_BASER - Base rate

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_BASER as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_BASER.")
        result = calculate_baser(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_BASER: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_BASER calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_acc(input_data, columns_names, logger=None):
    """Performs calculation of NBR_ACC - Accuracy

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_ACC as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_ACC.")
        result = calculate_acc(input_data, columns_names)
        safe_log(logger, "debug", f"Calculated NBR_ACC: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_ACC calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_fbias(input_data, columns_names, logger=None):
    """Performs calculation of NBR_FBIAS - Frequency Bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_FBIAS as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_FBIAS.")
        result = calculate_fbias(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_FBIAS: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_FBIAS calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_fmean(input_data, columns_names, logger=None):
    """Performs calculation of NBR_FMEAN - Forecast mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_FMEAN as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_FMEAN.")
        result = calculate_fmean(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_FMEAN: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_FMEAN calculation: {str(e)}")
        result = None

    return result

def calculate_nbr_pody(input_data, columns_names, logger=None):
    """Performs calculation of NBR_PODY - Probability of detecting yes

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_PODY as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_PODY.")
        result = calculate_pody(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_PODY: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_PODY calculation: {str(e)}")
        result = None

    return result

def calculate_nbr_pofd(input_data, columns_names, logger=None):
    """Performs calculation of NBR_POFD - Probability of false detection

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_POFD as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_POFD.")
        result = calculate_pofd(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_POFD: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_POFD calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_podn(input_data, columns_names, logger=None):
    """Performs calculation of NBR_PODN - Probability of false detection

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_PODN as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_PODN.")
        result = calculate_podn(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_PODN: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_PODN calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_far(input_data, columns_names, logger=None):
    """Performs calculation of NBR_FAR - False alarm ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_FAR as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_FAR.")
        result = calculate_far(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_FAR: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_FAR calculation: {str(e)}")
        result = None


def calculate_nbr_csi(input_data, columns_names, logger=None):
    """Performs calculation of NBR_CSI - Critical Success Index

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_CSI as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_CSI.")
        result = calculate_csi(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_CSI: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_CSI calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_gss(input_data, columns_names, logger=None):
    """Performs calculation of NBR_GSS - Gilbert Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_GSS as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_GSS.")
        result = calculate_gss(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_GSS: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_GSS calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_hk(input_data, columns_names, logger=None):
    """Performs calculation of NBR_HK - Hanssen-Kuipers Discriminant

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_HK as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_HK.")
        result = calculate_hk(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_HK: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_HK calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_hss(input_data, columns_names, logger=None):
    """Performs calculation of NBR_HSS - Heidke Skil lScore

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_HSS as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_HSS.")
        result = calculate_hss(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_HSS: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_HSS calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_odds(input_data, columns_names, logger=None):
    """Performs calculation of NBR_ODDS - Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated NBR_ODDS as float
            or None if some of the data values are missing or invalid
    """

    try:
        safe_log(logger, "debug", "Starting calculation of NBR_ODDS.")
        result = calculate_odds(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated NBR_ODDS: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_ODDS calculation: {str(e)}")
        result = None

    return result


def calculate_nbr_ctc_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for
        Neighborhood Contingency Table Statistics
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
        safe_log(logger, "debug", "Starting calculation of Total number of matched pairs for Neighborhood Contingency Table Statistics.")
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Calculated total: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during calculation of total: {str(e)}")
        result = None