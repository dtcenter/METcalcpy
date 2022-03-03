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

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_nbr_baser(input_data, columns_names):
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

    return calculate_baser(input_data, columns_names)


def calculate_nbr_acc(input_data, columns_names):
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

    return calculate_acc(input_data, columns_names)


def calculate_nbr_fbias(input_data, columns_names):
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

    return calculate_fbias(input_data, columns_names)


def calculate_nbr_fmean(input_data, columns_names):
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

    return calculate_fmean(input_data, columns_names)


def calculate_nbr_pody(input_data, columns_names):
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

    return calculate_pody(input_data, columns_names)


def calculate_nbr_pofd(input_data, columns_names):
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

    return calculate_pofd(input_data, columns_names)


def calculate_nbr_podn(input_data, columns_names):
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

    return calculate_podn(input_data, columns_names)


def calculate_nbr_far(input_data, columns_names):
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

    return calculate_far(input_data, columns_names)


def calculate_nbr_csi(input_data, columns_names):
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

    return calculate_csi(input_data, columns_names)


def calculate_nbr_gss(input_data, columns_names):
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

    return calculate_gss(input_data, columns_names)


def calculate_nbr_hk(input_data, columns_names):
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

    return calculate_hk(input_data, columns_names)


def calculate_nbr_hss(input_data, columns_names):
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

    return calculate_hss(input_data, columns_names)


def calculate_nbr_odds(input_data, columns_names):
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

    return calculate_odds(input_data, columns_names)


def calculate_nbr_ctc_total(input_data, columns_names):
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
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return round_half_up(total, PRECISION)
