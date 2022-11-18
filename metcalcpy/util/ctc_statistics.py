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

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_baser(input_data, columns_names):
    """Performs calculation of BASER - Base rate, aka Observed relative frequency

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
        result = (sum(input_data[:, np.where(columns_names == 'fy_oy')[0][0]])
                  + sum(input_data[:, np.where(columns_names == 'fn_oy')[0][0]])) \
                 / sum(input_data[:, np.where(columns_names == 'total')[0][0]])
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
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
            or None if some of the data values are missing or invalid
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


def calculate_fbias(input_data, columns_names):
    """Performs calculation of FBIAS - Bias, aka Frequency bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FBIAS as float
            or None if some of the data values are missing or invalid
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
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fmean(input_data, columns_names):
    """Performs calculation of FMEAN - Forecast mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FMEAN as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if total == 0:
            return None
        oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = oyn / total
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pody(input_data, columns_names):
    """Performs calculation of PODY - Probability of Detecting Yes

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PODY as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')

    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        oy = fy_oy + fn_oy
        result = fy_oy / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pofd(input_data, columns_names):
    """Performs calculation of POFD - Probability of false detection

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated POFD as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        oy = fy_on + sum_column_data_by_name(input_data, columns_names, 'fn_on')
        result = fy_on / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ctc_roc(data, ascending):
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
    # create a data frame to hold the aggregated contingency table and ROC data
    sorted_data = sort_by_thresh(data, ascending=ascending)
    list_thresh = np.sort(np.unique(sorted_data['fcst_thresh'].to_numpy()))

    # If descending order was requested for sorting the input dataframe,
    # reverse the order of the list_thresh to
    # maintain results in descending order.
    if not ascending:
        list_thresh = list_thresh[::-1]

    df_roc = pd.DataFrame(
        {'thresh': list_thresh, 'pody': None, 'pofd': None})

    index = 0
    for thresh in list_thresh:
        # create a subset of the sorted_data that contains only the rows of the unique
        # threshold values
        subset_data = sorted_data[sorted_data['fcst_thresh'] == thresh]
        data_np = subset_data.to_numpy()
        columns = subset_data.columns.values
        pody = calculate_pody(data_np, columns)
        pofd = calculate_pofd(data_np, columns)
        df_roc.loc[index] = [thresh, pody, pofd]
        index += 1

    return df_roc


def calculate_podn(input_data, columns_names):
    """Performs calculation of PODN - Probability of Detecting No

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PODN as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        oy = sum_column_data_by_name(input_data, columns_names, 'fy_on') + fn_on
        result = fn_on / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_far(input_data, columns_names):
    """Performs calculation of FAR - false alarms

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') + fy_on
        result = fy_on / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_csi(input_data, columns_names):
    """Performs calculation of CSI - Critical success index, aka Threat score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated CSI as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        oy = fy_oy \
             + sum_column_data_by_name(input_data, columns_names, 'fy_on') \
             + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = fy_oy / oy
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_gss(input_data, columns_names):
    """Performs calculation of GSS = Gilbert skill score, aka Equitable threat score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated GSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if total == 0:
            return None
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        dbl_c = ((fy_oy + fy_on) / total) * (fy_oy + fn_oy)
        gss = ((fy_oy - dbl_c) / (fy_oy + fy_on + fn_oy - dbl_c))
        gss = round_half_up(gss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        gss = None
    warnings.filterwarnings('ignore')
    return gss


def calculate_hk(input_data, columns_names):
    """Performs calculation of HK - Hanssen Kuipers Discriminant

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated HK as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        pody = calculate_pody(input_data, columns_names)
        pofd = calculate_pofd(input_data, columns_names)
        if pody is None or pofd is None:
            result = None
        else:
            result = pody - pofd
            result = round_half_up(result, PRECISION)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_hss(input_data, columns_names):
    """Performs calculation of HSS - Heidke skill score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated HSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        if total == 0:
            return None
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        dbl_c = ((fy_oy + fy_on) / total) * (fy_oy + fn_oy) + ((fn_oy + fn_on) / total) * (fy_on + fn_on)
        hss = ((fy_oy + fn_on - dbl_c)/ (total - dbl_c))
        hss = round_half_up(hss, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        hss = None
    warnings.filterwarnings('ignore')
    return hss


def calculate_odds(input_data, columns_names):
    """Performs calculation of ODDS - Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ODDS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        pody = calculate_pody(input_data, columns_names)
        pofd = calculate_pofd(input_data, columns_names)
        if pody is None or pofd is None:
            result = None
        else:

            result = (pody * (1 - pofd)) / (pofd * (1 - pody))
            result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')

    return result


def calculate_lodds(input_data, columns_names):
    """Performs calculation of LODDS - Log Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated LODDS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        if fy_oy is None or fy_on is None or fn_oy is None or fn_on is None:
            return None
        v = np.log(fy_oy) + np.log(fn_on) - np.log(fy_on) - np.log(fn_oy)
        v = round_half_up(v, PRECISION)
    except (TypeError, Warning):
        v = None
    warnings.filterwarnings('ignore')
    return v


def calculate_bagss(input_data, columns_names):
    """Performs calculation of BAGSS - Bias-Corrected Gilbert Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BAGSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        if fy_oy is None or fn_oy is None or fy_on is None or total is None:
            return None
        if fy_oy == 0 or fn_oy == 0 or total == 0:
            return None
        dbl_o = fy_oy + fn_oy
        dbl_lf = np.log(dbl_o / fn_oy)
        lambert = lambertw(dbl_o / fy_on * dbl_lf).real
        dbl_ha = dbl_o - (fy_on / dbl_lf) * lambert
        result = (dbl_ha - (dbl_o ** 2 / total)) / (2 * dbl_o - dbl_ha - (dbl_o ** 2 / total))
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_eclv(input_data, columns_names):
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
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        cl_step = 0.05
        cl_pts = np.arange(start=cl_step, stop=1, step=cl_step)
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        eclv = calculate_economic_value(np.array([fy_oy, fy_on, fn_oy, fn_on]), cl_pts)
        common_cases_ind = pd.Series(eclv['cl']).isin(cl_pts)
        v = eclv['V'][common_cases_ind]
        v = round_half_up(v, PRECISION)
    except (TypeError, Warning):
        v = None
    warnings.filterwarnings('ignore')
    return v


def calculate_economic_value(values, cost_lost_ratio=np.arange(start=0.05, stop=0.95, step=0.05),
                             add_base_rate: bool = False) -> Union[dict, None]:
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
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        if len(values) == 4:
            n = sum(values)
            f = values[1] / (values[1] + values[3])
            h = values[0] / (values[0] + values[2])
            s = (values[0] + values[2]) / n

            if add_base_rate is True:
                cl_local = np.append(cost_lost_ratio, s)
            else:
                cl_local = np.copy(cost_lost_ratio)
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
        else:
            result = None
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ctc_total(input_data, columns_names):
    """Calculates the Total number of matched pairs for Contingency Table Counts

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


def calculate_cts_total(input_data, columns_names):
    """Calculates the Total number of matched pairs for Contingency Table Statistics

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


def calculate_ctc_fn_on(input_data, columns_names):
    """Calculates the Number of forecast no and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast no and observation no as float
            or None if some of the data values are missing or invalid
    """
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    return round_half_up(fn_on, PRECISION)


def calculate_ctc_fn_oy(input_data, columns_names):
    """Calculates the Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast no and observation yes as float
            or None if some of the data values are missing or invalid
    """
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    return round_half_up(fn_oy, PRECISION)


def calculate_ctc_fy_on(input_data, columns_names):
    """Calculates the Number of forecast yes and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast yes and observation no as float
            or None if some of the data values are missing or invalid
    """
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    return round_half_up(fy_on, PRECISION)


def calculate_ctc_fy_oy(input_data, columns_names):
    """Calculates the Number of forecast yes and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Number of forecast yes and observation yes as float
            or None if some of the data values are missing or invalid
    """
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    return round_half_up(fy_oy, PRECISION)


def calculate_ctc_oy(input_data, columns_names):
    """Calculates the Total Number of forecast yes and observation yes plus
        Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated OY as float
            or None if some of the data values are missing or invalid
    """
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    return round_half_up(fy_oy + fn_oy, PRECISION)


def calculate_ctc_on(input_data, columns_names):
    """Calculates the Total Number of forecast yes and observation no plus
        Number of forecast no and observation no for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ON as float
            or None if some of the data values are missing or invalid
    """
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    return round_half_up(fy_on + fn_on, PRECISION)


def calculate_ctc_fy(input_data, columns_names):
    """Calculates the Total Number of forecast yes and observation no plus
        Number of forecast yes and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FY as float
            or None if some of the data values are missing or invalid
    """
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    return round_half_up(fy_on + fy_oy, PRECISION)


def calculate_ctc_fn(input_data, columns_names):
    """Calculates the Total Number of forecast no and observation no plus
        Number of forecast no and observation yes for Contingency Table Statistics

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FN as float
            or None if some of the data values are missing or invalid
    """
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    return round_half_up(fn_on + fn_oy, PRECISION)


def pod_yes(input_data, columns_names):
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        num = fy_oy
        den = fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def pod_no(input_data, columns_names):
    warnings.filterwarnings('error')
    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        num = fn_on
        den = fn_on + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_odds1(input_data, columns_names):
    """Performs calculation of ODDS - Odds Ratio

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ODDS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        py = pod_yes(input_data, columns_names)
        pn = calculate_pofd(input_data, columns_names)

        num = py / (1 - py)
        den = pn / (1 - pn)
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_orss(input_data, columns_names):
    """Performs calculation of ORSS - Odds Ratio Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ORSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')

        num = fy_oy * fn_on - fy_on * fn_oy
        den = fy_oy * fn_on + fy_on * fn_oy
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_sedi(input_data, columns_names):
    """Performs calculation of SEDI - Symmetric Extremal Depenency Index

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated SEDI as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')

        f = fy_on / (fy_on + fn_on)
        h = pod_yes(input_data, columns_names)

        num = math.log(f) - math.log(h) - math.log(1 - f) + math.log(1 - h)
        den = math.log(f) + math.log(h) + math.log(1 - f) + math.log(1 - h)
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_seds(input_data, columns_names):
    """Performs calculation of SEDS - Symmetric Extreme Dependency Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated SEDS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')

        num = math.log((fy_oy + fy_on) / total) + math.log((fy_oy + fn_oy) / total)

        den = math.log(fy_oy / total)
        result = num / den - 1.0
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_edi(input_data, columns_names):
    """Performs calculation of EDI - Extreme Dependency Index

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated EDI as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
        fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        f = fy_on / (fy_on + fn_on)
        h = pod_yes(input_data, columns_names)

        num = math.log(f) - math.log(h)
        den = math.log(f) + math.log(h)
        result = num / den
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_eds(input_data, columns_names):
    """Performs calculation of EDS - Extreme Dependency Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated EDs as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
        fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        total = sum_column_data_by_name(input_data, columns_names, 'total')

        num = math.log((fy_oy + fn_oy) / total)
        den = math.log(fy_oy / total)

        result = 2.0 * num / den - 1.0
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def sort_by_thresh(input_dataframe: pd.DataFrame, sort_column_name: str = 'fcst_thresh',
                   ascending: bool = True) -> pd.DataFrame:
    """
        Sorts the input pandas dataframe by threshold values in the specified column that have
        format "operator value", ie >=1.  This is done by first separating
        the fcst_thresh column into a threshold operator (<,<=, ==, >=, >)
        and thresh value column.  Assign a weight to each operator (1 for <,
        2 for <=, 3 for ==  and no operator, 4 for >=, and 5 for >).  Finally,
        sort the input dataframe by these two new columns resulting in a new
        dataframe sorted by fcst_thresh in ascending order (default) or descending
        order.

        Args:

            :param input_dataframe: A pandas dataframe representing data that is to be
                                   sorted according to the specifiec column name.


            :param sort_column_name:  The column to base the sorting.  The default is 'fcst_thresh'
                                     (which is a colunn in CTC output)

            :param ascending: A boolean value, by default is set to True to sort by
                              ascending value.  Set to False to sort in descending order.

        Returns:
            sorted_df:  A pandas dataframe that is sorted based on the specified column's values and
                        in the specified order (ascending or descending-default is ascending).


    """

    operators = []
    values = []
    text_strings = []
    requested_thresh = input_dataframe[sort_column_name]

    for thresh in requested_thresh:
        # for thresholds that are comprised of an operator and value, ie >=3,
        # separate the fcst_thresh into two parts: the operator (ie <, <=, ==, >=, >)
        # and the numerical value of the threshold (which can be a negative value)
        match = re.match(r'(\<|\<=|\==|\>=|\>)*((-)*([0-9])(.)*)', thresh)
        match_text = re.match(r'(\<|\<=|\==|\>=|\>)*((.)*)', thresh)

        if match:
            operators.append(match.group(1))
            value = float(match.group(2))
            values.append(value)
        elif match_text:
            operators.append(match_text.group(1))
            text = match_text.group(2)
            text_strings.append(text)

    # apply a numerical weighting to each operator: 1 for <, 2 for <=, etc.
    wt_maps = {'<': 1, '<=': 2, '==': 3, '>=': 4, '>': 5}
    operator_wts = []

    for operator in operators:
        # if no operator is found, assign the same
        # weight as used for the == operator (i.e. assume
        # that if a bare number is observed, assume that
        # a == is implied)
        if operator is None:
            # no operator, assume ==
            operator_wts.append(wt_maps['=='])
        else:
            operator_wts.append(wt_maps[operator])

    # Columns to use in pandas dataframe sorting
    sort_by_cols = ['thresh_values', 'op_wts']
    input_dataframe['op_wts'] = operator_wts

    # assign new column based on the format of the threshold values
    if match:
        input_dataframe['thresh_values'] = values
    elif match_text:
        input_dataframe['thresh_values'] = text_strings
    else:
        # if the threshold values don't conform to what is expected, then
        # use the requested threshold column name for sorting.
        sort_by_cols = [sort_column_name]

    # sort with ignore_index=True because we don't need to keep the original index values. We
    # want the rows to be newly indexed to reflect the reordering. Use inplace=False because
    # we don't want to modify the input dataframe's order, we want a new dataframe.
    sorted_dataframe = input_dataframe.sort_values(by=sort_by_cols, inplace=False,
                                                   ascending=ascending, ignore_index=True)
    return sorted_dataframe
