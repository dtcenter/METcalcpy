# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: pstd_statistics.py
"""
import warnings
import numpy as np
import pandas as pd

from metcalcpy.util.met_stats import get_column_index_by_name
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION

__author__ = 'Tatiana Burek'
__version__ = '2.0.1'


def calculate_pstd_brier(input_data, columns_names):
    """Performs calculation of PSTD_BRIER - Brier Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PSTD_BRIER as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar')]

        reliability = calc_reliability(t_table, df_pct_perm)
        resolution = calc_resolution(t_table, df_pct_perm, o_bar)
        uncertainty = calc_uncertainty(o_bar_table)

        brier = reliability - resolution + uncertainty
        result = round_half_up(brier, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_bss_smpl(input_data, columns_names):
    """Performs calculation of BSS_SMPL - Brier Skill Score relative to sample climatology

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BSS_SMPL as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar')]

        reliability = calc_reliability(t_table, df_pct_perm)
        resolution = calc_resolution(t_table, df_pct_perm, o_bar)
        uncertainty = calc_uncertainty(o_bar_table)

        bss_smpl = (resolution - reliability) / uncertainty
        result = round_half_up(bss_smpl, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_baser(input_data, columns_names):
    """Performs calculation of BASER - The Base Rate
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:

        df_pct_perm = _calc_common_stats(columns_names, input_data)
        t_table = df_pct_perm['n_i'].sum()
        baser = df_pct_perm['oy_i'].sum() / t_table
        result = round_half_up(baser, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_reliability(input_data, columns_names):
    """Performs calculation of RELIABILITY - Reliability

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated RELIABILITY as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data)
        t_table = df_pct_perm['n_i'].sum()

        reliability = calc_reliability(t_table, df_pct_perm)
        result = round_half_up(reliability, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_resolution(input_data, columns_names):
    """Performs calculation of RESOLUTION - Resolution

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated RESOLUTION as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data)
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar')]
        t_table = df_pct_perm['n_i'].sum()

        resolution = calc_resolution(t_table, df_pct_perm, o_bar)
        result = round_half_up(resolution, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_uncertainty(input_data, columns_names):
    """Performs calculation of UNCERTAINTY - Uncertainty

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated UNCERTAINTY as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table

        uncertainty = calc_uncertainty(o_bar_table)
        result = round_half_up(uncertainty, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_calibration(input_data, columns_names):
    """Performs calculation of calibration

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated calibration as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        oy_i = sum(input_data[:, get_column_index_by_name(columns_names, 'oy_i')])
        n_i = calculate_pstd_ni(input_data, columns_names)
        calibration = oy_i / n_i
        result = round_half_up(calibration, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_ni(input_data, columns_names):
    """Performs calculation of ni - Uncertainty

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ni as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        oy_i = sum(input_data[:, get_column_index_by_name(columns_names, 'oy_i')])
        on_i = sum(input_data[:, get_column_index_by_name(columns_names, 'on_i')])
        n_i = oy_i + on_i
        result = round_half_up(n_i, PRECISION)

    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_roc_auc(input_data, columns_names):
    """Performs calculation of ROC_AUC - Area under the receiver operating characteristic curve

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ROC_AUC as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')

    df_pct_perm = _calc_common_stats(columns_names, input_data)
    roc = _calc_pct_roc(df_pct_perm)
    # add 1st and last rows
    final_roc = pd.DataFrame(
        {'thresh': 0, 'n11': 0, 'n10': 0, 'n01': 0, 'n00': 0, 'pody': 1, 'pofd': 1},
        index=[0])
    final_roc = pd.concat([final_roc, roc])
    final_roc = pd.concat([final_roc,
        pd.DataFrame(
            {'thresh': 0, 'n11': 0, 'n10': 0, 'n01': 0, 'n00': 0, 'pody': 0, 'pofd': 0},
            index=[0]) ])

    roc_auc = 0
    for index, row in final_roc.iterrows():
        if index != 0:
            roc_auc = roc_auc + 0.5 * (final_roc.iloc[index - 1]['pody'] + row.pody) \
                      * (final_roc.iloc[index - 1]['pofd'] - row.pofd)

    result = round_half_up(roc_auc, PRECISION)

    return result


def calc_uncertainty(o_bar_table):
    """Performs calculation of uncertainty
         Args: o_bar_table
         Returns: uncertainty
    """
    uncertainty = o_bar_table * (1 - o_bar_table)
    return uncertainty


def calc_resolution(t_table, df_pct_perm, o_bar):
    """Performs calculation of resolution
         Args: t_table
            df_pct_perm
            o_bar

         Returns: resolution
    """
    resolution = sum([row.n_i * (row.o_bar_i - o_bar) * (row.o_bar_i - o_bar)
                      for index, row in df_pct_perm.iterrows()]) \
                 / t_table
    return resolution


def calc_reliability(t_table, df_pct_perm):
    """Performs calculation of reliability
         Args: t_table
            df_pct_perm
            o_bar

         Returns: reliability
    """
    reliability = sum([row.n_i * (row.thresh_i - row.o_bar_i) * (row.thresh_i - row.o_bar_i)
                       for index, row in df_pct_perm.iterrows()]) \
                  / t_table
    return reliability


def _calc_common_stats(columns_names, input_data):
    """ Creates a data frame to hold the aggregated contingency table and ROC data
            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                pandas data frame with columns:
                    - thresh_i
                    - oy_i
                    - on_i
                    - n_i
                    - o_bar_i
    """
    pct_perm = {'thresh_i': [], 'oy_i': [], 'on_i': []}
    for column in columns_names:
        index = get_column_index_by_name(columns_names, column)

        if "oy_i" in column:
            sum_val = sum_column_data_by_name(input_data, columns_names, column)
            pct_perm['oy_i'].append(sum_val)
        elif "on_i" in column:
            sum_val = sum_column_data_by_name(input_data, columns_names, column)
            pct_perm['on_i'].append(sum_val)
        elif 'thresh_i' in column:
            pct_perm['thresh_i'].append(input_data[0, index])
    # calculate vectors and constants to use below
    df_pct_perm = pd.DataFrame(pct_perm)
    n_i = [row.oy_i + row.on_i for index, row in df_pct_perm.iterrows()]
    df_pct_perm['n_i'] = n_i

    # use only records with n_i != 0
    df_pct_perm = df_pct_perm[df_pct_perm['n_i'] != 0]
    o_bar_i = [row.oy_i / row.n_i for index, row in df_pct_perm.iterrows()]
    calibration_i = [row.oy_i / row.n_i for index, row in df_pct_perm.iterrows()]
    df_pct_perm['o_bar_i'] = o_bar_i
    return df_pct_perm


def _calc_pct_roc(data):
    """ Creates a data frame to hold the aggregated contingency table and ROC data
            Args:
                data: pandas data frame with pstd data and column names:
                     - thresh_i
                    - oy_i
                    - on_i
                    - n_i
                    - o_bar_i

            Returns:
                pandas data frame with ROC data and columns:
                - thresh
                - n11
                - n10
                - n01
                - n00
                - pody
                - pofd
    """
    # create a data frame to hold the aggregated contingency table and ROC data
    list_thresh = np.unique(np.sort(data['thresh_i'].to_numpy()))
    df_roc = pd.DataFrame(
        {'thresh': list_thresh, 'n11': None, 'n10': None,
         'n01': None, 'n00': None, 'pody': None, 'pofd': None})

    # build the ROC contingency data table
    for thresh in list_thresh:
        is_bigger = data['thresh_i'] > thresh
        # use df_roc.loc rather than df_roc.at in pandas versions above 1.2.3
        df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n11'] = sum(data[is_bigger]['oy_i'])
        df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n10'] = sum(data[is_bigger]['on_i'])

        is_less = data['thresh_i'] <= thresh
        # use df_roc.loc rather than df_roc.at in pandas versions above 1.2.3
        df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n01'] = sum(data[is_less]['oy_i'])
        df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n00'] = sum(data[is_less]['on_i'])

    # generate the pody and pofd scores from the contingency tables
    df_roc['pody'] = [row.n11 / (row.n11 + row.n01) for index, row in df_roc.iterrows()]
    df_roc['pofd'] = [row.n10 / (row.n10 + row.n00) for index, row in df_roc.iterrows()]

    return df_roc


def calculate_pct_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for
        Contingency Table Counts for Probabilistic forecasts
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
