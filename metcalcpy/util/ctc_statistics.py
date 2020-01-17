"""
Program Name: ctc_statistics.py
"""
import warnings
import numpy as np
import pandas as pd
from scipy.special import lambertw
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        result = round_half_up(result, PRECISION)
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
        gss = round_half_up(gss, PRECISION)
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
        dbl_c = ((fy_oy + sum_column_data_by_name(input_data, columns_names, 'fy_on')) / total) \
                * (fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_oy'))
        hss = ((fy_oy + sum_column_data_by_name(input_data, columns_names, 'fn_on') - dbl_c)
               / (total - dbl_c))
        hss = round_half_up(hss, PRECISION)
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
            result = round_half_up(result, PRECISION)
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
        v = round_half_up(v, PRECISION)
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
            f = values[1] / (values[1] + values[3])
            h = values[0] / (values[0] + values[2])
            s = (values[0] + values[2]) / n

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
            result = round_half_up(result, PRECISION)
        else:
            result = None
    except (TypeError, ZeroDivisionError, Warning):
        result = None
    warnings.filterwarnings('ignore')
    return result
