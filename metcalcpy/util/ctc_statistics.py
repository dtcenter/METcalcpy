# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: ctc_statistics.py
"""
import warnings
from typing import Union
import math
import re
import numpy as np
import pandas as pd
from scipy.special import lambertw
from metcalcpy.util.utils import round_half_up, column_data_by_name, \
    sum_column_data_by_name, PRECISION
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_baser(input_data, columns_names, logger=None):
    """Performs calculation of BASER - Base rate, aka Observed relative frequency

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculate_baser function.")
    safe_log(logger, "debug", f"Input columns: {columns_names}")
    
    warnings.filterwarnings('error')

    try:
        fy_oy_index = np.where(columns_names == 'fy_oy')[0][0]
        fn_oy_index = np.where(columns_names == 'fn_oy')[0][0]
        total_index = np.where(columns_names == 'total')[0][0]
        
        safe_log(logger, "debug", f"Indexes found - fy_oy: {fy_oy_index}, fn_oy: {fn_oy_index}, total: {total_index}")

        # Perform the BASER calculation
        result = (sum(input_data[:, fy_oy_index]) + sum(input_data[:, fn_oy_index])) / sum(input_data[:, total_index])
        result = round_half_up(result, PRECISION)

        safe_log(logger, "info", f"BASER calculation successful: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in BASER calculation: {e}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_acc(input_data, columns_names):
    """Performs calculation of ACC - Accuracy

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ACC as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = (sum_column_data_by_name(input_data, columns_names, 'fy_oy')
                  + sum_column_data_by_name(input_data, columns_names, 'fn_on')) \
                 / sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fbias(input_data, columns_names, logger=None):
    """Performs calculation of FBIAS - Bias, aka Frequency bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FBIAS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
             + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        if oy == 0:
            return None
        oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = oyn / oy
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"ACC calculation successful: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        safe_log(logger, "error", f"Error in ACC calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fmean(input_data, columns_names, logger=None):
    """Performs calculation of FMEAN - Forecast mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FMEAN as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculate_fmean function.")
    safe_log(logger, "debug", f"Input columns: {columns_names}")

    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total value: {total}")
        if total == 0:
            safe_log(logger, "warning", "Total value is 0, returning None.")
            return None
        oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"Summed values (fy_oy + fy_on): {oyn}")
        result = oyn / total
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"FMEAN calculation successful: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        safe_log(logger, "error", f"Error in FMEAN calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pody(input_data, columns_names, logger=None):
    """Performs calculation of PODY - Probability of Detecting Yes

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PODY as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculate_pody function.")
    safe_log(logger, "debug", f"Input columns: {columns_names}")
    warnings.filterwarnings('error')

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        safe_log(logger, "debug", f"fy_oy value: {fy_oy}")
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        safe_log(logger, "debug", f"fn_oy value: {fn_oy}")
        oy = fy_oy + fn_oy
        safe_log(logger, "debug", f"Total oy value (fy_oy + fn_oy): {oy}")
        result = fy_oy / oy
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"PODY calculation successful: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in PODY calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pofd(input_data, columns_names, logger=None):
    """Performs calculation of POFD - Probability of false detection

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated POFD as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculate_pofd function.")
    safe_log(logger, "debug", f"Input columns: {columns_names}")
    warnings.filterwarnings('error')
    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"fy_on value: {fy_on}")
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        safe_log(logger, "debug", f"fn_on value: {fn_on}")
        oy = fy_on + sum_column_data_by_name(input_data, columns_names, 'fn_on')
        oy = fy_on + fn_on
        safe_log(logger, "debug", f"Total oy value (fy_on + fn_on): {oy}")
        result = fy_on / oy
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"POFD calculation successful: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in POFD calculation: {e}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ctc_roc(data, ascending, logger=None):
    """ Creates a data frame to hold the aggregated contingency table and ROC data
            Args:
                :param data: pandas data frame with ctc data and column names:
                    - fcst_thresh
                    - fy_oy
                    - fy_on
                    - fn_oy
                    - fn_on
                    - fcst_valid_beg
                    - fcst_lead
                :param ascending: order in which to sort the input data by fcst_thresh. Default is
                                  True, set to False to sort by descending order.

            Returns:
                pandas data frame with ROC data and columns:
                - thresh
                - pody
                - pofd
    """
    safe_log(logger, "debug", "Starting calculate_ctc_roc function.")
    safe_log(logger, "debug", f"Sorting data with ascending={ascending}.")
    # create a data frame to hold the aggregated contingency table and ROC data
    sorted_data = sort_by_thresh(data, ascending=ascending, logger=logger)
    safe_log(logger, "debug", f"Data sorted. Number of rows: {len(sorted_data)}")
    list_thresh = np.sort(np.unique(sorted_data['fcst_thresh'].to_numpy()))
    safe_log(logger, "debug", f"Unique thresholds identified: {list_thresh}")

    # If descending order was requested for sorting the input dataframe,
    # reverse the order of the list_thresh to
    # maintain results in descending order.
    if not ascending:
        list_thresh = list_thresh[::-1]
        safe_log(logger, "debug", "Reversed the order of list_thresh due to descending sort.")
    df_roc = pd.DataFrame(
        {'thresh': list_thresh, 'pody': None, 'pofd': None})
    safe_log(logger, "debug", "Initialized ROC DataFrame.")

    index = 0
    for thresh in list_thresh:
        safe_log(logger, "debug", f"Processing threshold: {thresh}")
        # create a subset of the sorted_data that contains only the rows of the unique
        # threshold values
        subset_data = sorted_data[sorted_data['fcst_thresh'] == thresh]
        safe_log(logger, "debug", f"Subset data for threshold {thresh} has {len(subset_data)} rows.")
        data_np = subset_data.to_numpy()
        columns = subset_data.columns.values
        pody = calculate_pody(data_np, columns, logger=logger)
        pofd = calculate_pofd(data_np, columns, logger=logger)
        df_roc.loc[index] = [thresh, pody, pofd]
        safe_log(logger, "info", f"ROC values for threshold {thresh}: PODY={pody}, POFD={pofd}")
        index += 1

    safe_log(logger, "debug", "Finished calculating ROC DataFrame.")
    return df_roc


def calculate_podn(input_data, columns_names, logger=None):
    """Performs calculation of PODN - Probability of Detecting No

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PODN as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')

    safe_log(logger, "debug", "Starting calculation of PODN.")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        safe_log(logger, "debug", f"Sum of fn_on: {fn_on}")
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"Sum of fy_on: {fy_on}")
        oy = sum_column_data_by_name(input_data, columns_names, 'fy_on') + fn_on
        safe_log(logger, "debug", f"Sum of observations (oy): {oy}")
        result = fn_on / oy
        result = round_half_up(result, PRECISION)
        safe_log(logger, "debug", f"Calculated PODN before rounding: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating PODN: {str(e)}")

    warnings.filterwarnings('ignore')
    return result


def calculate_far(input_data, column_names, logger=None):
    """Performs calculation of FAR - false alarms

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FAR as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of FAR.")
    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"Sum of fy_on: {fy_on}")
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        safe_log(logger, "debug", f"Sum of fy_oy: {fy_oy}")
        oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') + fy_on
        safe_log(logger, "debug", f"Sum of observations (oy): {oy}")
        result = fy_on / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating FAR: {str(e)}")

    warnings.filterwarnings('ignore')
    return result


def calculate_csi(input_data, columns_names, logger=None):
    """Performs calculation of CSI - Critical success index, aka Threat score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated CSI as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of CSI.")
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        safe_log(logger, "debug", f"Sum of fy_oy: {fy_oy}")
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"Sum of fy_on: {fy_on}")    
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        safe_log(logger, "debug", f"Sum of fn_oy: {fn_oy}")
        oy = fy_oy \
             + sum_column_data_by_name(input_data, columns_names, 'fy_on') \
             + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        safe_log(logger, "debug", f"Total sum (oy): {oy}")
        result = fy_oy / oy
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Final CSI result: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CSI: {str(e)}")

    warnings.filterwarnings('ignore')
    return result


def calculate_gss(input_data, columns_names, logger=None):
    """Performs calculation of GSS = Gilbert skill score, aka Equitable threat score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated GSS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of GSS.")

    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Sum of total: {total}")
        if total == 0:
            safe_log(logger, "warning", "Total is 0, returning None.")
            return None
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        safe_log(logger, "debug", f"Sum of fy_oy: {fy_oy}")
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"Sum of fy_on: {fy_on}")
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        safe_log(logger, "debug", f"Sum of fn_oy: {fn_oy}")
        dbl_c = ((fy_oy + fy_on) / total) * (fy_oy + fn_oy)
        safe_log(logger, "debug", f"Calculated dbl_c: {dbl_c}")
        gss = ((fy_oy - dbl_c) / (fy_oy + fy_on + fn_oy - dbl_c))
        gss = round_half_up(gss, PRECISION)
        safe_log(logger, "info", f"Final GSS result: {gss}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        gss = None
        safe_log(logger, "error", f"Error in calculating GSS: {str(e)}")
        gss = None

    warnings.filterwarnings('ignore')
    return gss


def calculate_hk(input_data, columns_names, logger=None):
    """Performs calculation of HK - Hanssen Kuipers Discriminant

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated HK as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of HK.")
    try:
        pody = calculate_pody(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated PODY: {pody}")
        pofd = calculate_pofd(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"Calculated POFD: {pofd}")
        if pody is None or pofd is None:
            safe_log(logger, "warning", "PODY or POFD is None, returning None.")
            result = None
        else:
            result = pody - pofd
            safe_log(logger, "debug", f"HK before rounding: {result}")
            result = round_half_up(result, PRECISION)
            safe_log(logger, "info", f"Final HK result: {result}")
    except (TypeError, Warning) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating HK: {str(e)}")
        
    warnings.filterwarnings('ignore')
    return result


def calculate_hss(input_data, columns_names, logger=None):
    """Performs calculation of HSS - Heidke skill score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated HSS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of HSS.")
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"Total: {total}")
        if total == 0:
            safe_log(logger, "warning", "Total is zero, returning None.")
            return None
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')

        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FY_ON: {fy_on}, FN_OY: {fn_oy}, FN_ON: {fn_on}")

        dbl_c = ((fy_oy + fy_on) / total) * (fy_oy + fn_oy) + ((fn_oy + fn_on) / total) * (fy_on + fn_on)

        hss = ((fy_oy + fn_on - dbl_c)/ (total - dbl_c))
        hss = round_half_up(hss, PRECISION)
        safe_log(logger, "info", f"Final HSS result: {hss}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        hss = None
        safe_log(logger, "error", f"Error in calculating HSS: {str(e)}")

    warnings.filterwarnings('ignore')
    return hss


def calculate_odds(input_data, columns_names, logger=None):
    """Performs calculation of ODDS - Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ODDS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of ODDS.")

    try:
        pody = calculate_pody(input_data, columns_names, logger=logger)
        pofd = calculate_pofd(input_data, columns_names, logger=logger)
        safe_log(logger, "debug", f"PODY: {pody}, POFD: {pofd}")
        if pody is None or pofd is None:
            safe_log(logger, "warning", "PODY or POFD is None, returning None.")
            result = None
        else:
            result = (pody * (1 - pofd)) / (pofd * (1 - pody))
            result = round_half_up(result, PRECISION)
            safe_log(logger, "info", f"Final ODDS result: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating ODDS: {str(e)}")
    
    warnings.filterwarnings('ignore')

    return result


def calculate_lodds(input_data, columns_names, logger=None):
    """Performs calculation of LODDS - Log Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated LODDS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of LODDS.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')

        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FY_ON: {fy_on}, FN_OY: {fn_oy}, FN_ON: {fn_on}")

        if fy_oy is None or fy_on is None or fn_oy is None or fn_on is None:
            safe_log(logger, "warning", "One or more input values are None, returning None.")
            return None
        v = np.log(fy_oy) + np.log(fn_on) - np.log(fy_on) - np.log(fn_oy)
        v = round_half_up(v, PRECISION)
        safe_log(logger, "info", f"Final LODDS result: {v}")

    except (TypeError, Warning) as e:
        v = None
        safe_log(logger, "error", f"Error in calculating LODDS: {str(e)}")
        
    warnings.filterwarnings('ignore')
    return v


def calculate_bagss(input_data, columns_names, logger=None):
    """Performs calculation of BAGSS - Bias-Corrected Gilbert Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BAGSS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of BAGSS.")
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FN_OY: {fn_oy}, FY_ON: {fy_on}, TOTAL: {total}")

        if fy_oy is None or fn_oy is None or fy_on is None or total is None:
            safe_log(logger, "warning", "One or more input values are None, returning None.")
            return None
        if fy_oy == 0 or fn_oy == 0 or total == 0:
            safe_log(logger, "warning", "One or more input values are zero, returning None.")
            return None
        dbl_o = fy_oy + fn_oy
        dbl_lf = np.log(dbl_o / fn_oy)
        lambert = lambertw(dbl_o / fy_on * dbl_lf).real
        dbl_ha = dbl_o - (fy_on / dbl_lf) * lambert
        result = (dbl_ha - (dbl_o ** 2 / total)) / (2 * dbl_o - dbl_ha - (dbl_o ** 2 / total))

        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Final BAGSS result: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating BAGSS: {str(e)}")
        
    warnings.filterwarnings('ignore')
    return result


def calculate_eclv(input_data, columns_names, logger=None):
    """Performs calculation of ECLV - Economic Cost/Loss  Value
        Implements R version that returns an array instead of the single value
        IT WILL NOT WORK - NEED TO CONSULT WITH STATISTICIAN
        Build list of X-axis points between 0 and 1

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BAGSS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of ECLV.")

    try:
        cl_step = 0.05
        cl_pts = np.arange(start=cl_step, stop=1, step=cl_step)
        safe_log(logger, "debug", f"CL points: {cl_pts}")

        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')

        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FY_ON: {fy_on}, FN_OY: {fn_oy}, FN_ON: {fn_on}")

        eclv = calculate_economic_value(np.array([fy_oy, fy_on, fn_oy, fn_on]), cl_pts, logger=logger)
        common_cases_ind = pd.Series(eclv['cl']).isin(cl_pts)
        v = eclv['V'][common_cases_ind]
        v = round_half_up(v, PRECISION)
        safe_log(logger, "info", f"Calculated ECLV: {v}")

    except (TypeError, Warning) as e:
        v = None
        safe_log(logger, "error", f"Error in calculating ECLV: {str(e)}")
        
    warnings.filterwarnings('ignore')
    return v


def calculate_economic_value(values, cost_lost_ratio=np.arange(start=0.05, stop=0.95, step=0.05),
                             add_base_rate: bool = False, logger=None) -> Union[dict, None]:
    """Calculates the economic value of a forecast based on a cost/loss ratio.
       Similar to R script function 'value' from  the 'verification' package

        Args:
            values: An array vector of a contingency table summary of values in the form
                c(n11, n01, n10, n00) where in nab a = obs, b = forecast.
            cost_lost_ratio:  Cost loss ratio. The relative value of being unprepared
                and taking a loss to that of un-necessarily preparing. For example,
                cl = 0.1 indicates it would cost $ 1 to prevent a $10 loss.
                This defaults to the sequence 0.05 to 0.95 by 0.05.
            add_base_rate: add Base rate point to cl or not

        Returns:
            If assigned to an object, the following values are reported in the dictionary :
                vmax - Maximum value
                V    - Vector of values for each cl value
                F    - Conditional false alarm rate.
                H    - Conditional hit rate
                cl   - Vector of cost loss ratios.
                s    - Base rate
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of economic value.")

    try:
        if len(values) == 4:
            n = sum(values)
            f = values[1] / (values[1] + values[3])
            h = values[0] / (values[0] + values[2])
            s = (values[0] + values[2]) / n

            safe_log(logger, "debug", f"Values: n={n}, F={f}, H={h}, s={s}")

            if add_base_rate is True:
                cl_local = np.append(cost_lost_ratio, s)
                safe_log(logger, "debug", f"Base rate added to cost/loss ratio: {cl_local}")
            else:
                cl_local = np.copy(cost_lost_ratio)
                safe_log(logger, "debug", f"Cost/loss ratio: {cl_local}")
            cl_local.sort()

            v_1 = (1 - f) - s / (1 - s) * (1 - cl_local) / cl_local * (1 - h)
            v_2 = h - (1 - s) / s * cl_local / (1 - cl_local) * f
            v = np.zeros(len(cl_local))

            indexes = cl_local < s
            v[indexes] = v_1[indexes]
            indexes = cl_local >= s
            v[indexes] = v_2[indexes]

            v_max = h - f
            result = {'vmax': round_half_up(v_max, PRECISION),
                      'V': v,
                      'F': round_half_up(f, PRECISION),
                      'H': round_half_up(h, PRECISION),
                      'cl': cl_local,
                      's': round_half_up(s, PRECISION),
                      'n': n}
            safe_log(logger, "info", f"Economic value calculated successfully: vmax={result['vmax']}")
        else:
            result = None
            safe_log(logger, "warning", "Invalid input values; calculation returned None.")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating economic value: {str(e)}")
        
    warnings.filterwarnings('ignore')
    return result


def calculate_ctc_total(input_data, columns_names, logger=None):
    """Calculates the Total number of matched pairs for Contingency Table Counts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC Total.")

    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "info", f"Calculated CTC Total: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC Total: {str(e)}")

    return result


def calculate_cts_total(input_data, columns_names, logger=None):
    """Calculates the Total number of matched pairs for Contingency Table Statistics

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated Total number of matched pairs as float
                or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTS Total.")

    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = round_half_up(total, PRECISION)
        safe_log(logger, "info", f"Calculated CTS Total: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTS Total: {str(e)}")

    return result


def calculate_ctc_fn_on(input_data, columns_names, logger=None):
    """Calculates the Number of forecast no and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast no and observation no as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC FN_ON.")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        result = round_half_up(fn_on, PRECISION)
        safe_log(logger, "info", f"Calculated CTC FN_ON: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC FN_ON: {str(e)}")

    return result


def calculate_ctc_fn_oy(input_data, columns_names, logger=None):
    """Calculates the Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast no and observation yes as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC FN_OY.")

    try:
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = round_half_up(fn_oy, PRECISION)
        safe_log(logger, "info", f"Calculated CTC FN_OY: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC FN_OY: {str(e)}")

    return result


def calculate_ctc_fy_on(input_data, columns_names, logger=None):
    """Calculates the Number of forecast yes and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast yes and observation no as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC FY_ON.")

    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = round_half_up(fy_on, PRECISION)
        safe_log(logger, "info", f"Calculated CTC FY_ON: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC FY_ON: {str(e)}")

    return result


def calculate_ctc_fy_oy(input_data, columns_names, logger=None):
    """Calculates the Number of forecast yes and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast yes and observation yes as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC FY_OY.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        result = round_half_up(fy_oy, PRECISION)
        safe_log(logger, "info", f"Calculated CTC FY_OY: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC FY_OY: {str(e)}")

    return result


def calculate_ctc_oy(input_data, columns_names, logger=None):
    """Calculates the Total Number of forecast yes and observation yes plus
        Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated OY as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC OY.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = round_half_up(fy_oy + fn_oy, PRECISION)
        safe_log(logger, "info", f"Calculated CTC OY: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        result = None
        safe_log(logger, "error", f"Error in calculating CTC OY: {str(e)}")

    return result


def calculate_ctc_on(input_data, columns_names, logger=None):
    """Calculates the Total Number of forecast yes and observation no plus
        Number of forecast no and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ON as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC ON.")
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    return round_half_up(fy_on + fn_on, PRECISION)


def calculate_ctc_fy(input_data, columns_names, logger=None):
    """Calculates the Total Number of forecast yes and observation no plus
        Number of forecast yes and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FY as float
            or None if some data values are missing or invalid
    """
    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        result = round_half_up(fy_on + fn_on, PRECISION)

        # Logging the successful calculation
        safe_log(logger, "info", f"Calculated CTC ON: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Logging the error if an exception occurs
        safe_log(logger, "error", f"Error in calculating CTC ON: {str(e)}")
        result = None

    return result


def calculate_ctc_fn(input_data, columns_names, logger=None):
    """Calculates the Total Number of forecast no and observation no plus
        Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FN as float
            or None if some data values are missing or invalid
    """
    safe_log(logger, "debug", "Starting calculation of CTC FN.")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = round_half_up(fn_on + fn_oy, PRECISION)

        # Logging the successful calculation
        safe_log(logger, "info", f"Calculated CTC FN: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Logging the error if an exception occurs
        safe_log(logger, "error", f"Error in calculating CTC FN: {str(e)}")
        result = None

    return result


def pod_yes(input_data, columns_names, logger=None):
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of POD (yes).")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        num = fy_oy
        den = fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = num / den
        result = round_half_up(result, PRECISION)

        # Log the successful calculation
        safe_log(logger, "info", f"Calculated POD (yes): {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Log the error
        safe_log(logger, "error", f"Error in calculating POD (yes): {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def pod_no(input_data, columns_names, logger=None):
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of POD (no).")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        num = fn_on
        den = fn_on + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = num / den
        result = round_half_up(result, PRECISION)

        # Log the successful calculation
        safe_log(logger, "info", f"Calculated POD (no): {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Log the error
        safe_log(logger, "error", f"Error in calculating POD (no): {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_odds1(input_data, columns_names, logger=None):
    """Performs calculation of ODDS - Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ODDS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of ODDS.")

    try:
        py = pod_yes(input_data, columns_names, logger=logger)
        pn = calculate_pofd(input_data, columns_names, logger=logger)

        # Log the intermediate values
        safe_log(logger, "debug", f"POD (yes): {py}, POFD: {pn}")

        num = py / (1 - py)
        den = pn / (1 - pn)
        result = num / den
        result = round_half_up(result, PRECISION)

        # Log the successful calculation
        safe_log(logger, "info", f"Calculated ODDS: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Log the error
        safe_log(logger, "error", f"Error in calculating ODDS: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_orss(input_data, columns_names, logger=None):
    """Performs calculation of ORSS - Odds Ratio Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ORSS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')

    safe_log(logger, "debug", "Starting calculation of ORSS.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')

        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FN_ON: {fn_on}, FY_ON: {fy_on}, FN_OY: {fn_oy}")

        num = fy_oy * fn_on - fy_on * fn_oy
        den = fy_oy * fn_on + fy_on * fn_oy
        result = num / den
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Calculated ORSS: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Log the error
        safe_log(logger, "error", f"Error in calculating ORSS: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_sedi(input_data, columns_names, logger=None):
    """Performs calculation of SEDI - Symmetric Extremal Depenency Index

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated SEDI as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of SEDI.")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')

        safe_log(logger, "debug", f"FN_ON: {fn_on}, FY_ON: {fy_on}")

        f = fy_on / (fy_on + fn_on)
        h = pod_yes(input_data, columns_names, logger=logger)

        safe_log(logger, "debug", f"F (false alarm rate): {f}, H (hit rate): {h}")


        num = math.log(f) - math.log(h) - math.log(1 - f) + math.log(1 - h)
        den = math.log(f) + math.log(h) + math.log(1 - f) + math.log(1 - h)
        result = num / den
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Calculated SEDI: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in calculating SEDI: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_seds(input_data, columns_names, logger=None):
    """Performs calculation of SEDS - Symmetric Extreme Dependency Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated SEDS as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of SEDS.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')

        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FY_ON: {fy_on}, FN_OY: {fn_oy}, Total: {total}")

        num = math.log((fy_oy + fy_on) / total) + math.log((fy_oy + fn_oy) / total)

        den = math.log(fy_oy / total)
        result = num / den - 1.0
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Calculated SEDS: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in calculating SEDS: {str(e)}")
        result = None

    warnings.filterwarnings('ignore')
    return result


def calculate_edi(input_data, columns_names, logger=None):
    """Performs calculation of EDI - Extreme Dependency Index

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated EDI as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of EDI.")

    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"FN_ON: {fn_on}, FY_ON: {fy_on}, Total: {total}")
        f = fy_on / (fy_on + fn_on)
        h = pod_yes(input_data, columns_names, logger=logger)

        num = math.log(f) - math.log(h)
        den = math.log(f) + math.log(h)
        result = num / den
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Calculated EDI: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        # Log the error
        safe_log(logger, "error", f"Error in calculating EDI: {str(e)}")
        result = None
    
    warnings.filterwarnings('ignore')
    return result


def calculate_eds(input_data, columns_names, logger=None):
    """Performs calculation of EDS - Extreme Dependency Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated EDs as float
            or None if some data values are missing or invalid
    """
    warnings.filterwarnings('error')
    safe_log(logger, "debug", "Starting calculation of EDS.")

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        safe_log(logger, "debug", f"FY_OY: {fy_oy}, FN_OY: {fn_oy}, Total: {total}")

        num = math.log((fy_oy + fn_oy) / total)
        den = math.log(fy_oy / total)

        result = 2.0 * num / den - 1.0
        result = round_half_up(result, PRECISION)
        safe_log(logger, "info", f"Calculated EDS: {result}")

    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "error", f"Error in calculating EDS: {str(e)}")
        result = None
    
    warnings.filterwarnings('ignore')
    return result


def sort_by_thresh(input_dataframe: pd.DataFrame, sort_column_name: str = 'fcst_thresh',
                   ascending: bool = True, logger=None) -> pd.DataFrame:
    """
        Sorts the input pandas dataframe by threshold values in the specified column that have
        format "operator value", ie >=1.  This is done by first separating
        the fcst_thresh column into a threshold operator (<,<=, ==, >=, >)
        and thresh value column.  Assign a weight to each operator so that > has the highest weight.
        Assign a "fill" value of -999999.99 for nan values and handle compound thresholds (e.g. >=3.14 && <=10.2)
        Finally, sort the input dataframe by these two new columns resulting in a new dataframe sorted by fcst_thresh
        (default) in ascending order (default) or descending order.  NA values will always be last (following the
        behavior of pandas sort_values()).

        Args:

            :param input_dataframe: A pandas dataframe representing data that is to be
                                   sorted according to the specifiec column name.


            :param sort_column_name:  The column to base the sorting.  The default is 'fcst_thresh'
                                     (which is a colunn in CTC output)

            :param ascending: A boolean value, by default is set to True to sort by
                              ascending value, False to sort in descending order.

        Returns:
            sorted_df:  A pandas dataframe that is sorted based on the specified column's values and
                        in the specified order (ascending or descending-default is ascending).


    """

    operators = []
    values = []
    second_compararison_operator = []
    text_strings = []


    # Replace nan with the string 'NA' for just the sort_column_name threshold
    input_dataframe[sort_column_name] = input_dataframe[sort_column_name].fillna('NA')

    requested_thresh = input_dataframe[sort_column_name]


    # If the df_input dataframe is empty (most likely as a result of event equalization),
    # return the df_input data frame.
    if input_dataframe.empty:
        safe_log(logger, "warning", "Input dataframe is empty. Returning original dataframe.")
        return input_dataframe

    for thresh in requested_thresh:

        # for thresholds that are comprised of an operator and value, ie >=3,
        # separate the fcst_thresh into two parts: the operator (ie <, <=, ==, >=, >)
        # and the numerical value of the threshold (which can be a negative value)

        # Capture the entire threshold expression
        match = re.match(r'(NA)|(\<|\<=|\==|\>=|\>)*((-)*(\d+\.*\d*))(.\s*&*)?', thresh)

        # Handle NA values separately
        nan_value = -999999.99
        match_na = re.match(r'(NA)', thresh)

        # Handle expressions with operators
        # match_text = re.match(r'(\<|\<=|\==|\>=|\>)*(.*)', thrsh)
        match_num = re.match(r'((\<|\<=|\==|\>=|\>)*((-)*(\d+\.*\d*))(.\s*&*)?)', thresh)
        match_text = re.match(r'(\<|\<=|\==|\>=|\>)*((.)*)', thresh)

        if match:
            if match.group(1):
                if match_na:
                   operators.append('NA')
                   value = nan_value
                   second_compararison_operator.append(False)
                else:
                    # Raw number, no comparison operator.
                    operators.append(None)
                    value = float(match_text.group(3))
                    second_compararison_operator.append(False)
                values.append(value)

            elif match_num:
                # value after the comparison operator is a numerical value
                operators.append(match_num.group(2))
                value = float(match_num.group(3))
                values.append(value)

                # && or & found, signifying a second comparison operation
                if match_num.group(6):
                    second_compararison_operator.append(True)
                else:
                    second_compararison_operator.append(False)

        elif match_text:
                # value after the comparison operator is text e.g. > SPF20
                operators.append(match_text.group(1))
                text = match_text.group(2)
                text_strings.append(text)
                second_compararison_operator.append(False)

     # Assign weights to the operators so that
    # > supercedes all other operators and there is spread
    # between comparisons to allow decreasing the weights if there
    # are more constraints (i.e. second comparison: >=3.2 && <9.9 vs >=3.2,
    # where the >=3.2 will have a higher weight value).
    wt_maps = {'NA': -1, '<': 3, '<=': 5, '==': 7, '>=': 9, '>': 11}

    operator_wts = []

    for idx, operator in enumerate(operators):
        # if no operator is found, assign the same
        # weight as used for the == operator (i.e. assume
        # that if a bare number is observed, assume that
        # a == is implied)
        if operator is None:
            # no operator, assume ==
            operator_wts.append(wt_maps['=='])
        else:
            wt_value = int(wt_maps[operator])
            if second_compararison_operator[idx]:
                # Reduce the weighting value by 1 to give precedence to lower-bounded thresholds:
                # e.g. >=5.1 has higher weight over >=5.1 && <9.3
                wt_value =  wt_value - 1

            operator_wts.append(wt_value)

    # Columns to use in pandas dataframe's sort_values()
    sort_by_cols = ['thresh_values', 'op_wts']
    input_dataframe['op_wts'] = operator_wts

    # assign new column based on the format of the threshold values
    if match:
        input_dataframe['thresh_values'] = values
    elif match_text:
        input_dataframe['thresh_values'] = text_strings
    else:
        # if the threshold values don't conform to what is expected, then
        # use the requested threshold column name for sorting (rely on the pandas
        # sort_values() to do the sorting of the original threshold values).
        sort_by_cols = [sort_column_name]

    # sort with ignore_index=True because we don't need to keep the original index values. We
    # want the rows to be newly indexed to reflect the reordering. Use inplace=False because
    # we don't want to modify the input dataframe's order, we want a new dataframe.
    safe_log(logger, "debug", f"Sorting by columns: {sort_by_cols} in {'ascending' if ascending else 'descending'} order.")
    sorted_dataframe = input_dataframe.sort_values(by=sort_by_cols, inplace=False,
                                                   ascending=ascending, ignore_index=True)
    safe_log(logger, "info", "Dataframe sorted successfully.")

    return sorted_dataframe