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
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '2.0.1'


def calculate_pstd_brier(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar', logger=logger)]
        reliability = calc_reliability(t_table, df_pct_perm, logger=logger)
        resolution = calc_resolution(t_table, df_pct_perm, o_bar, logger=logger)
        uncertainty = calc_uncertainty(o_bar_table, logger=logger)
        
        safe_log(logger, "debug", f"reliability: {reliability}, resolution: {resolution}, uncertainty: {uncertainty}")

        brier = reliability - resolution + uncertainty
        result = round_half_up(brier, PRECISION)
        
        safe_log(logger, "debug", f"Calculated Brier Score: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError, AttributeError) as e:
        safe_log(logger, "warning", f"Exception occurred during Brier Score calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_bss_smpl(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar', logger=logger)]

        reliability = calc_reliability(t_table, df_pct_perm, logger=logger)
        resolution = calc_resolution(t_table, df_pct_perm, o_bar, logger=logger)
        uncertainty = calc_uncertainty(o_bar_table, logger=logger)
        
        safe_log(logger, "debug", f"reliability: {reliability}, resolution: {resolution}, uncertainty: {uncertainty}")

        bss_smpl = (resolution - reliability) / uncertainty
        result = round_half_up(bss_smpl, PRECISION)
        
        safe_log(logger, "debug", f"Calculated Brier Skill Score (BSS_SMPL): {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during BSS_SMPL calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_baser(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        t_table = df_pct_perm['n_i'].sum()
        baser = df_pct_perm['oy_i'].sum() / t_table
        result = round_half_up(baser, PRECISION)
        safe_log(logger, "debug", f"Calculated BASER: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during BASER calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_reliability(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        t_table = df_pct_perm['n_i'].sum()
        reliability = calc_reliability(t_table, df_pct_perm, logger=logger)
        result = round_half_up(reliability, PRECISION)
        safe_log(logger, "debug", f"Calculated RELIABILITY: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RELIABILITY calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_resolution(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        o_bar = input_data[0, get_column_index_by_name(columns_names, 'o_bar', logger=logger)]
        t_table = df_pct_perm['n_i'].sum()
        resolution = calc_resolution(t_table, df_pct_perm, o_bar, logger=logger)
        result = round_half_up(resolution, PRECISION)
        safe_log(logger, "debug", f"Calculated RESOLUTION: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during RESOLUTION calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_uncertainty(input_data, columns_names, logger=None):
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
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        t_table = df_pct_perm['n_i'].sum()
        o_bar_table = df_pct_perm['oy_i'].sum() / t_table
        uncertainty = calc_uncertainty(o_bar_table, logger=logger)
        result = round_half_up(uncertainty, PRECISION)
        safe_log(logger, "debug", f"Calculated UNCERTAINTY: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during UNCERTAINTY calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_calibration(input_data, columns_names, logger=None):
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
        oy_i = sum(input_data[:, get_column_index_by_name(columns_names, 'oy_i', logger=logger)])
        n_i = calculate_pstd_ni(input_data, columns_names, logger=logger)
        calibration = oy_i / n_i
        result = round_half_up(calibration, PRECISION)
        safe_log(logger, "debug", f"Calculated calibration: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during calibration calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_ni(input_data, columns_names, logger=None):
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
        oy_i = sum(input_data[:, get_column_index_by_name(columns_names, 'oy_i', logger=logger)])
        on_i = sum(input_data[:, get_column_index_by_name(columns_names, 'on_i', logger=logger)])
        n_i = oy_i + on_i
        result = round_half_up(n_i, PRECISION)
        safe_log(logger, "debug", f"Calculated ni: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during ni calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pstd_roc_auc(input_data, columns_names, logger=None):
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

    try:
        df_pct_perm = _calc_common_stats(columns_names, input_data, logger=logger)
        roc = _calc_pct_roc(df_pct_perm, logger=logger)

        # Add first and last rows
        final_roc = pd.DataFrame(
            {'thresh': 0, 'n11': 0, 'n10': 0, 'n01': 0, 'n00': 0, 'pody': 1, 'pofd': 1},
            index=[0])
        final_roc = pd.concat([final_roc, roc])
        final_roc = pd.concat([final_roc,
            pd.DataFrame(
                {'thresh': 0, 'n11': 0, 'n10': 0, 'n01': 0, 'n00': 0, 'pody': 0, 'pofd': 0},
                index=[0]) ])

        final_roc.reset_index(inplace=True, drop=True)
        safe_log(logger, "debug", "Prepared final ROC curve with added boundary points.")

        roc_auc = 0
        for index, row in final_roc.iterrows():
            if index != 0:
                increment = 0.5 * (final_roc.iloc[index - 1]['pody'] + row.pody) \
                            * (final_roc.iloc[index - 1]['pofd'] - row.pofd)
                roc_auc += increment
                safe_log(logger, "debug", f"Step {index}: Added area increment {increment} to ROC_AUC, current value: {roc_auc}.")

        result = round_half_up(roc_auc, PRECISION)
        safe_log(logger, "debug", f"Final calculated ROC_AUC: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError, AttributeError) as e:
        safe_log(logger, "warning", f"Exception occurred during ROC_AUC calculation: {str(e)}")
        result = None

    return result


def calc_uncertainty(o_bar_table, logger=None):
    """Performs calculation of uncertainty
         Args: o_bar_table
         Returns: uncertainty
    """
    try:
        uncertainty = o_bar_table * (1 - o_bar_table)
        safe_log(logger, "debug", f"Calculated uncertainty: {uncertainty}")
        return uncertainty
    except (TypeError, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during uncertainty calculation: {str(e)}")
        return None


def calc_resolution(t_table, df_pct_perm, o_bar, logger=None):
    """Performs calculation of resolution
         Args: t_table
            df_pct_perm
            o_bar

         Returns: resolution
    """
    try:
        resolution = sum([row.n_i * (row.o_bar_i - o_bar) ** 2 
                          for index, row in df_pct_perm.iterrows()]) / t_table
        safe_log(logger, "debug", f"Calculated resolution: {resolution}")
        return resolution
    except (TypeError, ZeroDivisionError, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during resolution calculation: {str(e)}")
        return None


def calc_reliability(t_table, df_pct_perm, logger=None):
    """Performs calculation of reliability
         Args: t_table
            df_pct_perm
            o_bar

         Returns: reliability
    """
    try:
        reliability = sum([row.n_i * (row.thresh_i - row.o_bar_i) ** 2
                           for index, row in df_pct_perm.iterrows()]) / t_table
        safe_log(logger, "debug", f"Calculated reliability: {reliability}")
        return reliability
    except (TypeError, ZeroDivisionError, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during reliability calculation: {str(e)}")
        return None


def _calc_common_stats(columns_names, input_data, logger=None):
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
    try:
        pct_perm = {'thresh_i': [], 'oy_i': [], 'on_i': []}
        
        for column in columns_names:
            index = get_column_index_by_name(columns_names, column, logger=logger)
            
            if "oy_i" in column:
                sum_val = sum_column_data_by_name(input_data, columns_names, column)
                pct_perm['oy_i'].append(sum_val)
                safe_log(logger, "debug", f"Aggregated oy_i for {column}: {sum_val}")
                
            elif "on_i" in column:
                sum_val = sum_column_data_by_name(input_data, columns_names, column)
                pct_perm['on_i'].append(sum_val)
                safe_log(logger, "debug", f"Aggregated on_i for {column}: {sum_val}")
                
            elif 'thresh_i' in column:
                thresh_val = input_data[0, index]
                pct_perm['thresh_i'].append(thresh_val)
                safe_log(logger, "debug", f"Threshold value for {column}: {thresh_val}")
        
        # Create DataFrame
        df_pct_perm = pd.DataFrame(pct_perm)
        df_pct_perm.reset_index(inplace=True, drop=True)

        # Calculate n_i and o_bar_i
        df_pct_perm['n_i'] = [row.oy_i + row.on_i for index, row in df_pct_perm.iterrows()]
        safe_log(logger, "debug", "Calculated n_i for all rows.")

        # Filter out rows where n_i is 0
        df_pct_perm = df_pct_perm[df_pct_perm['n_i'] != 0]
        safe_log(logger, "debug", "Filtered out rows with n_i = 0.")

        df_pct_perm['o_bar_i'] = [row.oy_i / row.n_i for index, row in df_pct_perm.iterrows()]
        safe_log(logger, "debug", "Calculated o_bar_i for all rows.")

        return df_pct_perm
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during common stats calculation: {str(e)}")
        return None


def _calc_pct_roc(data, logger=None):
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
    try:
        safe_log(logger, "debug", "Starting calculation of ROC data.")

        # Create a DataFrame to hold the aggregated contingency table and ROC data
        list_thresh = np.unique(np.sort(data['thresh_i'].to_numpy()))
        safe_log(logger, "debug", f"Thresholds identified: {list_thresh}")

        df_roc = pd.DataFrame(
            {'thresh': list_thresh, 'n11': None, 'n10': None,
             'n01': None, 'n00': None, 'pody': None, 'pofd': None})

        # Build the ROC contingency data table
        for thresh in list_thresh:
            safe_log(logger, "debug", f"Processing threshold: {thresh}")

            is_bigger = data['thresh_i'] > thresh
            df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n11'] = sum(data[is_bigger]['oy_i'])
            df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n10'] = sum(data[is_bigger]['on_i'])
            safe_log(logger, "debug", f"Calculated n11 and n10 for threshold {thresh}.")

            is_less = data['thresh_i'] <= thresh
            df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n01'] = sum(data[is_less]['oy_i'])
            df_roc.loc[df_roc.index[df_roc["thresh"] == thresh], 'n00'] = sum(data[is_less]['on_i'])
            safe_log(logger, "debug", f"Calculated n01 and n00 for threshold {thresh}.")

        df_roc.reset_index(inplace=True, drop=True)

        # Generate the pody and pofd scores from the contingency tables
        df_roc['pody'] = [row.n11 / (row.n11 + row.n01) for index, row in df_roc.iterrows()]
        df_roc['pofd'] = [row.n10 / (row.n10 + row.n00) for index, row in df_roc.iterrows()]
        safe_log(logger, "debug", "Calculated pody and pofd scores.")

        return df_roc
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during ROC calculation: {str(e)}")
        return None


def calculate_pct_total(input_data, columns_names, logger=None):
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
    try:

        # Summing up the 'total' column data
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total matched pairs calculated: {total}")

        result = round_half_up(total, PRECISION)
        safe_log(logger, "debug", f"Result after rounding: {result}")

        return result
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during total matched pairs calculation: {str(e)}")
        return None