"""
Program Name: statistics.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy.special import lambertw

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

# CTC stat calculations
from metcalcpy.util.utils import round_half_up


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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
             + sum_column_data_by_name(input_data, columns_names, 'fn_on')
        if oy == 0:
            return None
        oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')
        result = oyn / oy
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        oy = fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
        result = fy_oy / oy
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        gss = round_half_up(gss, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
            result = round_half_up(result, 5)
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
        dbl_c = ((fy_oy + sum_column_data_by_name(input_data, columns_names, 'fy_on')) / total) \
                * (fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_oy'))
        hss = ((fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_on') - dbl_c)
               / (total - dbl_c))
        hss = round_half_up(hss, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
            result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        v = round_half_up(v, 5)
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
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
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
        v = round_half_up(v, 5)
    except (TypeError, Warning):
        v = None
    warnings.filterwarnings('ignore')
    return v


def calculate_economic_value(values, cost_lost_ratio=np.arange(start=0.05, stop=0.95, step=0.05)):
    """Calculates the economic value of a forecast based on a cost/loss ratio.

        Args:
            values: An array vector of a contingency table summary of values in the form
                c(n11, n01, n10, n00) where in nab a = obs, b = forecast.
            cost_lost_ratio:  Cost loss ratio. The relative value of being unprepared
                and taking a loss to that of un-necessarily preparing. For example,
                cl = 0.1 indicates it would cost $ 1 to prevent a $10 loss.
                This defaults to the sequence 0.05 to 0.95 by 0.05.

        Returns:
            calculated economic_value as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        if len(values) == 4:
            n = sum(values)
            a = values[0]
            b = values[1]
            c = values[2]
            d = values[3]
            f = b / (b + d)
            h = a / (a + c)
            s = (a + c) / n

            cl_local = np.append(cost_lost_ratio, s)
            cl_local.sort()

            v_1 = (1 - f) - s / (1 - s) * (1 - cl_local) / cl_local * (1 - h)
            v_2 = h - (1 - s) / s * cl_local / (1 - cl_local) * f
            v = np.zeros(len(cl_local))

            indexes = cl_local < s
            v[indexes] = v_1[indexes]
            indexes = cl_local >= s
            v[indexes] = v_2[indexes]

            v_max = h - f
            result = {'vmax': v_max, 'V': v, 'F': f, 'H': h, 'cl': cl_local, 's': s, 'n': n}
            result = round_half_up(result, 5)
        else:
            result = None
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


# SL1L2 stat calculations

def calculate_fbar(input_data, columns_names):
    """Performs calculation of FBAR - Forecast mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_obar(input_data, columns_names):
    """Performs calculation of OBAR - Observation Mean

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated OBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fstdev(input_data, columns_names):
    """Performs calculation of FSTDEV - Forecast standard deviation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        result = calculate_stddev(fbar * total, ffbar * total, total)
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ostdev(input_data, columns_names):
    """Performs calculation of OSTDEV - Observation Standard Deviation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated OSTDEV as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        result = calculate_stddev(obar * total, oobar * total, total)
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fobar(input_data, columns_names):
    """Performs calculation of FOBAR - Average product of forecast and observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FOBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ffbar(input_data, columns_names):
    """Performs calculation of FFBAR - Average of forecast squared

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated FFBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_oobar(input_data, columns_names):
    """Performs calculation of OOBAR - Average of observation squared

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated OOBAR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mae(input_data, columns_names):
    """Performs calculation of MAE - Mean absolute error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated MAE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'mae') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mbias(input_data, columns_names):
    """Performs calculation of MBIAS - Multiplicative Bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated MBIAS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        if obar == 0:
            result = None
        else:
            fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
            result = fbar / obar
            result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_pr_corr(input_data, columns_names):
    """Performs calculation of PR_CORR - Pearson correlation coefficient
        including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PR_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) \
            * (total ** 2 * oobar - total ** 2 * obar ** 2)
        pr_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
        if v <= 0 or pr_corr > 1:
            pr_corr = None
        else:
            pr_corr = round_half_up(pr_corr, 5)
    except (TypeError, ZeroDivisionError, Warning):
        pr_corr = None
    warnings.filterwarnings('ignore')
    return pr_corr


def calculate_anom_corr(input_data, columns_names):
    """Performs calculation of ANOM_CORR - The Anomoly Correlation
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated PR_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) \
            * (total ** 2 * oobar - total ** 2 * obar ** 2)
        if v <= 0:
            return None
        anom_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
        if anom_corr > 1:
            anom_corr = None
        else:
            anom_corr = round_half_up(anom_corr, 5)
    except (TypeError, ZeroDivisionError, Warning):
        anom_corr = None
    warnings.filterwarnings('ignore')
    return anom_corr


def calculate_rmsfa(input_data, columns_names):
    """Performs calculation of RMSFA - Root mean squared forecast anomaly (f-c)
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated RMSFA as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        if ffbar is None or ffbar < 0:
            result = None
        else:
            result = np.sqrt(ffbar)
            result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rmsoa(input_data, columns_names):
    """Performs calculation of RMSOA - Root mean squared observation anomaly (o-c)
     including normal confidence limits

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated RMSOA as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        if oobar is None or oobar < 0:
            result = None
        else:
            result = np.sqrt(oobar)
            result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me(input_data, columns_names):
    """Performs calculation of ME - Mean error, aka Additive bias

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ME as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
        obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
        result = fbar - obar
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_me2(input_data, columns_names):
    """Performs calculation of ME2 - The square of the mean error (bias)

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated ME2 as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        me = calculate_me(input_data, columns_names)
        result = me ** 2
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mse(input_data, columns_names):
    """Performs calculation of MSE - Mean squared error

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated MSE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
        oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
        fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
        result = ffbar + oobar - 2 * fobar
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_msess(input_data, columns_names):
    """Performs calculation of MSESS - The mean squared error skill score

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated MSESS as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        ostdev = calculate_ostdev(input_data, columns_names)
        mse = calculate_mse(input_data, columns_names)
        result = 1.0 - mse / ostdev ** 2
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_rmse(input_data, columns_names):
    """Performs calculation of RMSE - Root-mean squared error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated RMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_mse(input_data, columns_names))
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_estdev(input_data, columns_names):
    """Performs calculation of ESTDEV - Standard deviation of the error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

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
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcmse(input_data, columns_names):
    """Performs calculation of BCMSE - Bias-corrected mean squared error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated BCMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        mse = calculate_mse(input_data, columns_names)
        me = calculate_me(input_data, columns_names)
        result = mse - me ** 2
        result = round_half_up(result, 5)
    except (TypeError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_bcrmse(input_data, columns_names):
    """Performs calculation of BCRMSE - Bias-corrected root mean square error

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated BCRMSE as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        result = np.sqrt(calculate_bcmse(input_data, columns_names))
        result = round_half_up(result, 5)
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


# GRAD stat calculations

def calculate_fgbar(input_data, columns_names):
    """Performs calculation of FGBAR - Mean of absolute value of forecast gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated FGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_ogbar(input_data, columns_names):
    """Performs calculation of OGBAR - Mean of absolute value of observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated OGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_mgbar(input_data, columns_names):
    """Performs calculation of MGBAR - Mean of maximum of absolute values of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated MGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_egbar(input_data, columns_names):
    """Performs calculation of EGBAR - Mean of absolute value of forecast minus observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated EGBAR as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        result = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1(input_data, columns_names):
    """Performs calculation of S1 - S1 score

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated S1 as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        mgbar = sum_column_data_by_name(input_data, columns_names, 'mgbar') / total
        result = 100 * egbar / mgbar
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_s1_og(input_data, columns_names):
    """Performs calculation of S1_OG - S1 score with respect to observed gradient

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated S1_OG as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        egbar = sum_column_data_by_name(input_data, columns_names, 'egbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * egbar / ogbar
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fgog_ratio(input_data, columns_names):
    """Performs calculation of FGOG_RATIO - Ratio of forecast and observed gradients

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns_names: names of the columns for the 2nd dimension as Numpy array

            Returns:
                calculated FGOG_RATIO as float
                or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        fgbar = sum_column_data_by_name(input_data, columns_names, 'fgbar') / total
        ogbar = sum_column_data_by_name(input_data, columns_names, 'ogbar') / total
        result = 100 * fgbar / ogbar
        result = round_half_up(result, 5)
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result


def sum_column_data_by_name(input_data, columns, column_name):
    """Calculates  SUM of all values in the specified column

            Args:
                input_data: 2-dimensional numpy array with data for the calculation
                    1st dimension - the row of data frame
                    2nd dimension - the column of data frame
                columns: names of the columns for the 2nd dimension as Numpy array
                column_name: the name of the column for SUM

            Returns:
                calculated SUM as float
                or None if some of the data values are None
    """
    # find the index of specified column
    index_array = np.where(columns == column_name)[0]
    if index_array.size == 0:
        return None

    # get column's data
    try:
        data_array = input_data[:, index_array[0]]
    except IndexError:
        data_array = None
    if data_array is None or None in data_array:
        return None

    # return sum
    try:
        result = sum(data_array.astype(np.float))
    except TypeError:
        result = None

    return result


def get_column_index_by_name(columns, column_name):
    """Finds teh index of the specified column in the array

            Args:
                columns: names of the columns as Numpy array
                column_name: the name of the column

            Returns:
                the index of the column
                or None if the column name does not exist in the array
    """
    index_array = np.where(columns == column_name)[0]
    if index_array.size == 0:
        return None
    return index_array[0]
