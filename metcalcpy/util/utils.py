# ============================*
# ** Copyright UCAR (c) 2020
# ** University Corporation for Atmospheric Research (UCAR)
# ** National Center for Atmospheric Research (NCAR)
# ** Research Applications Lab (RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ============================*


"""
Program Name: utils.py
"""

__author__ = 'Tatiana Burek'

from typing import Union
import math
import sys
import re
import itertools
import statistics as st
import numpy as np
import pandas as pd
from pandas import DataFrame
import warnings

from scipy import stats
from scipy.stats import t, nct
from metcalcpy.util.correlation import corr, remove_none, acf
from metcalcpy import GROUP_SEPARATOR, DATE_TIME_REGEX
from metcalcpy.event_equalize import event_equalize
from metcalcpy.util.wald_wolfowitz_runs_test import runs_test


OPERATION_TO_SIGN = {
    'DIFF': '-',
    'RATIO': '/',
    'SS': ' and ',
    'DIFF_SIG': '-',
    'SINGLE': '-',
    'ETB': ' and '
}
STR_TO_BOOL = {'True': True, 'False': False}

# precision value for statistics calculations
PRECISION = 7

TWO_D_DATA_FILTER = {'object_type': '2d'}
THREE_D_DATA_FILTER = {'object_type': '3d'}

CODE_TO_OUTCOME_TO_MESSAGE = {
    'diff_eqv': 'statistically different from zero and statistically equivalent to zero',
    'diff_no_eqv': 'statistically different from zero and statistically not equivalent to zero',
    'no_diff_eqv': 'statistically not different from zero and statistically equivalent to zero',
    'no_diff_no_eqv': 'statistically not different from zero and statistically not equivalent to zero'
}


class DerivedCurveComponent:
    """ Holds components and the operation for a derived series
    """

    def __init__(self, first_component, second_component, derived_operation):
        self.first_component = first_component
        self.second_component = second_component
        self.derived_operation = derived_operation


def represents_int(possible_int):
    """Checks if the value is integer.

        Args:
            possible_int: value to check

        Returns:
            True - if the input value is an integer
            False - if the input value is not an integer
    """
    return isinstance(possible_int, int)


def is_string_integer(str_int):
    """Checks if the input string is integer.

         Args:
             str_int: string value to check

        Returns:
            True - if the input value is an integer
            False - if the input value is not an integer
    """
    try:
        int(str_int)
        return True
    except (ValueError, TypeError):
        return False


def is_string_strictly_float(str_float) -> bool:
    """Checks if the input string is strictly float.

         Args:
             str_float: string value to check

        Returns:
            True - if the input value is strictly float
            False - if the input value is not strictly float
    """
    if str_float is None:
        return False
    if str_float.startswith('-'):
        str_float = str_float[1:]
    return '.' in str_float and str_float.replace('.', '', 1).isdecimal()


def get_derived_curve_name(list_of_names):
    """Creates the derived series name from the list of series name components

         Args:
             list_of_names: list of series name components
                1st element - name of 1st series
                2st element - name of 2st series
                3st element - operation. Can be 'DIFF','RATIO', 'SS', 'SINGLE', 'ETB'

        Returns:
            derived series name
    """
    size = len(list_of_names)
    operation = 'DIFF'
    if size < 2:
        return ""
    if size == 3:
        operation = list_of_names[2]
    return f"{operation}({list_of_names[0]}{OPERATION_TO_SIGN[operation]}{list_of_names[1]})"


def calc_derived_curve_value(val1, val2, operation):
    """Performs the operation with two numpy arrays.
        Operations can be
            'DIFF' - difference between elements of array 1 and 2
            'RATIO' - ratio between elements of array 1 and 2
            'SS' - skill score between elements of array 1 and 2
            'SINGLE' - unchanged elements of array 1
            'ETB' - Equivalence Testing Bounds of array 1 and 2

        Args:
            val1:  array of floats
            val2:  array of floats
            operation: operation to perform

        Returns:
             array
            or None if one of arrays is None or one of the elements
            is None or arrays have different size
       """

    if val1 is None or val2 is None or None in val1 \
            or None in val2 or len(val1) != len(val2):
        return None

    result_val = None
    if operation in ('DIFF', 'DIFF_SIG'):
        result_val = [a - b for a, b in zip(val1, val2)]
    elif operation == 'RATIO':
        if 0 not in val2:
            result_val = [a / b for a, b in zip(val1, val2)]
    elif operation == 'SS':
        if 0 not in val1:
            result_val = [(a - b) / a for a, b in zip(val1, val2)]
    elif operation == 'SINGLE':
        result_val = val1
    elif operation == 'ETB':
        corr_val = corr(x=val1, y=val2)['r'].tolist()[0]
        result_val = tost_paired(len(val1),
                                 st.mean(val1),
                                 st.mean(val2),
                                 st.stdev(val1),
                                 st.stdev(val2),
                                 corr_val,
                                 -0.001, 0.001
                                 )
    return result_val


def unique(in_list):
    """Extracts unique values from the list.
        Args:
            in_list: list of values

        Returns:
            list of unique elements from the input list
    """
    if not in_list:
        return None
    # insert the list to the set
    list_set = set(in_list)
    # convert the set to the list
    return list(list_set)


def intersection(l_1, l_2):
    """Finds intersection between two lists
        Args:
            l_1: 1st list
            l_2: 2nd list

        Returns:
            list of intersection
    """
    if l_1 is None or l_2 is None:
        return None
    l_3 = [value for value in l_1 if value in l_2]
    return l_3


def is_derived_point(point):
    """Determines if this point is a derived point
        Args:
            point: a list or tuple with point component values

        Returns:
            True - if this point is derived
            False - if this point is not derived
    """
    is_derived = False
    if point is not None:
        for operation in OPERATION_TO_SIGN:
            for point_component in point:
                if point_component.startswith((operation + '(', operation + ' (')):
                    is_derived = True
                    break
    return is_derived


def parse_bool(in_str):
    """Converts string to a boolean
        Args:
            in_str: a string that represents a boolean

        Returns:
            boolean representation of the input string
            ot string itself
    """
    return STR_TO_BOOL.get(in_str, in_str)


def round_half_up(num, decimals=0):
    """The “rounding half up” strategy rounds every number to the nearest number
        with the specified precision,
     and breaks ties by rounding up.
        Args:
            n:  number
            decimals: decimal place
        Returns:
            rounded number

    """
    multiplier = 10 ** decimals
    return math.floor(num * multiplier + 0.5) / multiplier


def sum_column_data_by_name(input_data, columns, column_name, rm_none=True):
    """Calculates  SUM of all values in the specified column

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns: names of the columns for the 2nd dimension as Numpy array
            column_name: the name of the column for SUM
            rm_none: Should missing values (including non) be removed? Default - True

        Returns:
            calculated SUM as float
            or None if all of the data values are non
    """

    data_array = column_data_by_name(input_data, columns, column_name)

    if data_array is None or np.isnan(data_array).all():
        return None

    try:
        if rm_none:
            result = np.nansum(data_array.astype(float))
        else:
            if np.isnan(data_array).any():
                result = None
            else:
                result = sum(data_array.astype(float))
    except TypeError:
        result = None

    return result


def column_data_by_name(input_data, columns, column_name, rm_none=False) -> Union[list, None]:
    """Returns all values in the specified column. Removes None if requested

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns: names of the columns for the 2nd dimension as Numpy array
            column_name: the name of the column for SUM
            rm_none: Should missing values (including non) be removed? Default - False

        Returns:
            values of requested column or None
    """
    # find the index of specified column
    index_array = np.where(columns == column_name)[0]
    if index_array.size == 0:
        return None

    # get column's data and convert it into float array
    try:
        data_array = np.array(input_data[:, index_array[0]], dtype=np.float64)

        if rm_none:
            # remove non values
            data_array = [i for i in data_array if not np.isnan(i)]
    except IndexError:
        data_array = None

    return data_array


def column_data_by_name_value(input_data, columns, filters):
    """Filters  the input array by the criteria from the filters array

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns: names of the columns for the 2nd dimension as Numpy array
            filters: a dictionary of filters in 'column': 'value' format

        Returns:
            filtered 2-dimensional numpy array
    """
    input_data_filtered = np.copy(input_data)
    try:
        # for each filter
        for key, value in filters.items():
            # get an index og the column
            index_array = np.where(columns == key)[0]
            if index_array.size == 0:
                return 0

            filter_ind = input_data_filtered[:, index_array[0]].astype(type(value)) == value
            input_data_filtered = input_data_filtered[filter_ind]

    except IndexError:
        input_data_filtered = []

    return input_data_filtered


def nrow_column_data_by_name_value(input_data, columns, filters):
    """Calculates  the number of rows  that satisfy the criteria from the filters array

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns: names of the columns for the 2nd dimension as Numpy array
            filters: a dictionary of filters in 'column': 'value' format

        Returns:
            calculated number of rows
    """

    input_data_filtered = column_data_by_name_value(input_data, columns, filters)
    return input_data_filtered.shape[0]


def perfect_score_adjustment(mean_stats_1, mean_stats_2, statistic, pval):
    """ Adjusts the perfect score depending on the statistic

        Args:
            mean_stats_1: statistic value for the 1st point
            mean_stats_2: statistic value for the 2nd point
            statistic: name of the statistic
            pval: perfect score


        Returns:
            Adjusted perfect score or None if statistic is unknown
    """
    na_perf_score_stats = ('BASER', 'FMEAN', 'FBAR', 'FSTDEV', 'OBAR', 'OSTDEV',
                           'FRANK_TIES', 'ORANK_TIES',
                           'FBAR', 'FSTDEV', 'OBAR', 'OSTDEV', 'RANKS', 'FRANK_TIES',
                           'ORANK_TIES', 'VL1L2_FBAR', 'VL1L2_OBAR',
                           'VL1L2_FSTDEV', 'VL1L2_OSTDEV', 'VL1L2_FOSTDEV', 'PSTD_BASER',
                           'PSTD_RESOLUTION', 'PSTD_UNCERTAINTY',
                           'NBR_UFSS', 'NBR_F_RATE', 'NBR_O_RATE',
                           'NBR_BASER', 'NBR_FMEAN')

    zero_perf_score_stats = ('POFD', 'FAR', 'ESTDEV', 'MAE', 'MSE', 'BCMSE',
                             'RMSE', 'E10', 'E25', 'E50', 'E75',
                             'E90', 'EIQR', 'MAD', 'ME2', 'ME', 'ESTDEV', 'ODDS',
                             'LODDS', 'VL1L2_MSE', 'VL1L2_RMSE',
                             'VL1L2_RMSVE', 'PSTD_BRIER', 'PSTD_RELIABILITY',
                             'NBR_FBS', 'VL1L2_SPEED_ERR',
                             'NBR_POFD', 'NBR_FAR', 'NBR_ODDS', 'BCRMSE', 'ECNT_ME',
                             'ECNT_RMSE', 'CRPS', 'ECNT_CRPS', 'ECNT_MAE', 'ECNT_MAE_OERR',
                             'ECNT_ME_GE_OBS', 'ECNT_ME_LT_OBS')

    one_perf_score_stats = ('ACC', 'FBIAS', 'PODY', 'PODN', 'CSI', 'GSS',
                            'HK', 'HSS', 'ORSS', 'EDS', 'SEDS',
                            'EDI', 'SEDI', 'BAGSS', 'PR_CORR', 'SP_CORR',
                            'KT_CORR', 'MBIAS', 'ANOM_CORR', 'ANOM_CORR_RAW',
                            'VL1L2_BIAS', 'VL1L2_CORR',
                            'PSTD_BSS', 'PSTD_BSS_SMPL', 'NBR_FSS', 'NBR_AFSS',
                            'VAL1L2_ANOM_CORR', 'NBR_ACC',
                            'NBR_FBIAS', 'NBR_PODY', 'PSTD_ROC_AUC',
                            'NBR_PODN', 'NBR_CSI', 'NBR_GSS', 'NBR_HK', 'NBR_HSS',
                            'ECNT_BIAS_RATIO')

    if statistic.upper() in na_perf_score_stats:
        result = None
    elif statistic.upper() in zero_perf_score_stats \
            and abs(mean_stats_1) > abs(mean_stats_2):
        result = pval * -1
    elif statistic.upper() in one_perf_score_stats \
            and abs(mean_stats_1 - 1) > abs(mean_stats_2 - 1):
        result = pval * -1
    else:
        print(
            f"WARNING: statistic {statistic} doesn't belong to any of the perfect score groups. Returning unprocessed p-value")
        result = pval

    return result


def get_total_values(input_data, columns_names, aggregation):
    """Returns the total value for the given numpy array

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
                1 - if the aggregation was not preformed on the array
                sum of all values from 'total' columns
                    - if the aggregation was preformed on the array
        """
    total = 1
    if aggregation:
        total = sum_column_data_by_name(input_data, columns_names, 'total')
    return total


def aggregate_field_values(series_var_val, input_data_frame, line_type):
    """Finds and aggregates statistics for fields with values containing ';'.
      Aggregation  happens by valid and lead times
        These fields are coming from the scorecard and looks like this: vx_mask : ['EAST;NMT'].
        This method finds these values and calculate aggregated stats for them

            Args:
                series_var_val: dictionary describing the series
                input_data_frame: Pandas DataFrame
                line_type: the line type

            Returns:
                Pandas DataFrame with aggregates statistics
            """

    warnings.filterwarnings('error')

    # get unique values for valid date/time and lead time
    unique_valid = input_data_frame.fcst_valid_beg.unique()
    unique_lead = input_data_frame.fcst_lead.unique()

    for series_var, series_vals in series_var_val.items():
        for series_val in series_vals:
            if ';' in series_val:
                # found the aggregated field
                single_values = series_val.split(';')

                # for each valid
                for valid in unique_valid:
                    if series_var != 'fcst_lead':
                        for lead in unique_lead:
                            # find rows for the aggregation and their indexes
                            rows_for_agg = input_data_frame[
                                (input_data_frame.fcst_valid_beg == valid)
                                & (input_data_frame.fcst_lead == lead)
                                & (input_data_frame[series_var].isin(single_values))
                                ]
                            rows_indexes = rows_for_agg.index.values
                            # reset indexes
                            rows_for_agg.reset_index(inplace=True, drop=True)

                            # remove these rows from the main data_frame
                            input_data_frame = input_data_frame.drop(index=rows_indexes)

                            aggregated_result = calc_series_sums(rows_for_agg, line_type)

                            # record the result as a first row in the old selection
                            for field in input_data_frame.columns:
                                if field in aggregated_result.columns.values:
                                    rows_for_agg.at[0, field] = aggregated_result[field][0]

                            # replace the aggregated field name
                            rows_for_agg.at[0, series_var] = series_val

                            # add it to the result
                            input_data_frame = pd.concat([input_data_frame, rows_for_agg.iloc[:1]])

                    else:
                        # if the aggregated field is 'fcst_lead'

                        # pandas treats the fcst_lead values as dtype int64, therefore convert
                        # the single_values to int type so the pd.isin() properly matches
                        # the input_data_frame[series_var].
                        single_values_int = []
                        for val in single_values:
                            if val != '':
                                single_values_int.append((int(val)))

                        # find rows for the aggregation and their indexes
                        rows_for_agg = input_data_frame[
                            (input_data_frame.fcst_valid_beg == valid)
                            & (input_data_frame[series_var].isin(single_values_int))
                            ]

                        rows_indexes = rows_for_agg.index.values

                        # reset indexes
                        rows_for_agg.reset_index(inplace=True, drop=True)

                        # remove these rows from the main data_frame
                        input_data_frame = input_data_frame.drop(index=rows_indexes)

                        aggregated_result = calc_series_sums(rows_for_agg, line_type)

                        # record the result as a first row in the old selection
                        for field in input_data_frame.columns:
                            if field in aggregated_result.columns.values:
                                rows_for_agg.at[0, field] = aggregated_result[field][0]

                        # replace the aggregated field name

                        # some versions of pandas will generate an error when assigning a numerical value with a ';'
                        # strip off the ';' to make this "version-independent"
                        if ';' in series_val:
                            sep_val = series_val.split(';')

                        # different versions of pandas may generate a SettingWithCopy Warning
                        # For pandas 1.2, 1.3, and 1.5, use the df.at, but for pandas 1.4, use the df.loc.
                        try:
                            rows_for_agg.at[0, series_var] = sep_val[0]
                        except:
                            rows_for_agg.loc[0, series_var] = sep_val[0]

                        # add it to the result
                        input_data_frame = pd.concat([input_data_frame,(rows_for_agg.iloc[:1])])

    return input_data_frame


def calc_series_sums(input_df, line_type):
    """ Aggregates column values of the input data frame. Aggregation depends on the line type.
        Following line types are currently supported : ctc, sl1l2, sal1l2, vl1l2,
        val1l2, grad, nbrcnt, ecnt, rps

           Args:
               input_df: input data as Pandas DataFrame
               line_type: one of the supported line types
           Returns:
               Pandas DataFrame with aggregated values
       """
    # create an array from the dataframe
    sums_data_frame = pd.DataFrame()

    # calculate aggregated total value and add it to the result
    total = sum_column_data_by_name(input_df.to_numpy(), input_df.columns, 'total')
    sums_data_frame['total'] = [total]

    # proceed for the line type
    if line_type in ('ctc', 'nbrctc'):
        column_names = ['fy_oy', 'fy_on', 'fn_oy', 'fn_on']
        for column in column_names:
            sums_data_frame[column] = [sum_column_data_by_name(input_df.to_numpy(),
                                                               column_names, column_names)]

    elif line_type == 'sl1l2':
        column_names = ['fbar', 'obar', 'fobar', 'ffbar', 'oobar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(float))
                                       / total]

    elif line_type == 'sal1l2':
        sums_data_frame['fbar'] = [np.nansum(input_df['fabar'] * input_df.total.astype(float))
                                   / total]
        sums_data_frame['obar'] = [np.nansum(input_df['oabar'] * input_df.total.astype(float))
                                   / total]
        sums_data_frame['fobar'] = [np.nansum(input_df['foabar'] * input_df.total.astype(float))
                                    / total]
        sums_data_frame['ffbar'] = [np.nansum(input_df['ffabar'] * input_df.total.astype(float))
                                    / total]
        sums_data_frame['oobar'] = [np.nansum(input_df['ooabar'] * input_df.total.astype(float))
                                    / total]

    elif line_type == 'vl1l2':
        column_names = ['ufbar', 'vfbar', 'uobar', 'vobar', 'uvfobar',
                        'uvffbar', 'uvoobar', 'f_speed_bar', 'o_speed_bar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(float))
                                       / total]

    elif line_type == 'val1l2':
        column_names = ['ufabar', 'vfabar', 'uoabar', 'voabar', 'uvfoabar', 'uvffabar', 'uvooabar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(float))
                                       / total]

    elif line_type == 'grad':
        column_names = ['fgbar', 'ogbar', 'mgbar', 'egbar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(float))
                                       / total]

    elif line_type == 'nbrcnt':
        dbl_fbs = np.nansum(input_df['fbs'] * input_df.total.astype(float)) / total
        dbl_fss_den = np.nansum(
            (input_df['fbs'] / (1.0 - input_df['fss'])) * input_df.total.astype(float)) \
                      / total
        dbl_fss = 1.0 - dbl_fbs / dbl_fss_den
        dbl_f_rate = np.nansum(input_df['f_rate'] * input_df.total.astype(float)) / total
        dbl_o_rate = np.nansum(input_df['o_rate'] * input_df.total.astype(float)) / total
        dbl_a_fss_num = 2.0 * dbl_f_rate * dbl_o_rate
        dbl_a_fss_den = dbl_f_rate * dbl_f_rate + dbl_o_rate * dbl_o_rate
        dbl_a_fss = dbl_a_fss_num / dbl_a_fss_den
        dbl_u_fss = 0.5 + dbl_o_rate / 2.0
        sums_data_frame['fbs'] = [dbl_fbs]
        sums_data_frame['fss'] = [dbl_fss]
        sums_data_frame['afss'] = [dbl_a_fss]
        sums_data_frame['ufss'] = [dbl_u_fss]
        sums_data_frame['f_rate'] = [dbl_f_rate]
        sums_data_frame['o_rate'] = [dbl_o_rate]

    elif line_type == 'ecnt':
        mse = input_df['rmse'] * input_df['rmse']
        mse_oerr = input_df['rmse_oerr'] * input_df['rmse_oerr']
        crps_climo = input_df['crps'] / (1 - input_df['crpss'])

        sums_data_frame['mse'] = [np.nansum(input_df['total'] * mse) / total]
        sums_data_frame['mse_oerr'] = [np.nansum(input_df['total'] * mse_oerr) / total]
        sums_data_frame['crps_climo'] = [np.nansum(input_df['total'] * crps_climo) / total]
        column_names = ['me', 'crps', 'ign', 'spread', 'me_oerr', 'spread_oerr', 'spread_plus_oerr']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(float))
                                       / total]
    elif line_type == 'rps':
        d_rps_climo = input_df['rps'] / (1 - input_df['rpss'])
        sums_data_frame['rps'] = [np.nansum(input_df["rps"] * input_df.total.astype(float))
                                  / total]
        sums_data_frame['rps_comp'] = [np.nansum(input_df["rps_comp"] * input_df.total.astype(float))
                                       / total]
        sums_data_frame['rps_climo'] = [np.nansum(d_rps_climo * input_df.total.astype(float))
                                        / total]

    return sums_data_frame


def equalize_axis_data(fix_vals_keys, fix_vals_permuted, params, input_data, axis='1'):
    """ Performs event equalisation on the specified axis on input data.
        Args:
            fix_vals_permuted - fixed values
            params:        parameters for the statistic calculations  and data description
            input_data:    data as DataFrame
        Returns:
            DataFrame with equalised data for the specified axis
    """
    output_ee_data = pd.DataFrame()

    # for each statistic for the specified axis

    if 'fcst_var_val_' + axis in params:
        fcst_var_val = params['fcst_var_val_' + axis]
        if fcst_var_val is None:
            fcst_var_val = {}
    else:
        fcst_var_val = {'': ['']}

    for fcst_var, fcst_var_stats in fcst_var_val.items():
        series_data_for_ee = pd.DataFrame()
        for fcst_var_stat in fcst_var_stats:
            # for each series for the specified axis
            if len(params['series_val_' + axis]) == 0:
                series_data_for_ee = input_data
            else:
                for series_var, series_var_vals in params['series_val_' + axis].items():
                    # ungroup series value if needed
                    series_var_vals_no_group = []
                    for val in series_var_vals:
                        split_val = re.findall(DATE_TIME_REGEX, val)
                        if len(split_val) == 0:
                            split_val = val.split(GROUP_SEPARATOR)
                        series_var_vals_no_group.extend(split_val)

                    # filter input data based on fcst_var, statistic
                    # and all series variables values
                    series_data_for_ee = input_data
                    if series_var in input_data.keys():
                        series_data_for_ee = series_data_for_ee[
                            series_data_for_ee[series_var].isin(series_var_vals_no_group)]
                    if 'fcst_var' in input_data.keys():
                        series_data_for_ee = series_data_for_ee[series_data_for_ee['fcst_var'] == fcst_var]
                    if 'stat_name' in input_data.keys():
                        series_data_for_ee = series_data_for_ee[series_data_for_ee["stat_name"] == fcst_var_stat]

            # perform EE on filtered data
            # for SSVAR line_type use equalization of multiple events
            series_data_after_ee = \
                event_equalize(series_data_for_ee, params['indy_var'],
                               params['series_val_' + axis],
                               fix_vals_keys,
                               fix_vals_permuted, True,
                               params['line_type'] == "ssvar")

            # append EE data to result
            if output_ee_data.empty:
                output_ee_data = series_data_after_ee
            else:
                warnings.simplefilter(action="error", category=FutureWarning)
                output_ee_data = pd.concat([output_ee_data, series_data_after_ee])

    try:
        output_ee_data_valid = output_ee_data.drop('equalize', axis=1)

        # It is possible to produce an empty data frame after applying event equalization. Print an informational
        # message before returning the data frame.
        if output_ee_data_valid.empty:
            print(f"\nINFO: Event equalization has produced no results.  Data frame is empty.")

        return output_ee_data_valid
    except (KeyError, AttributeError):
        # Two possible exceptions are raised when the data frame is empty *and* is missing the 'equalize' column
        # following event equalization. Return the empty dataframe
        # without dropping the 'equalize' column, and print an informational message.
        print(f"\nINFO: No resulting data after performing event equalization of axis", axis)

    return output_ee_data


def perform_event_equalization(params, input_data):
    """ Performs event equalisation on input data. If there ara 2 axis:
        perform EE on each and then on both
        Args:
            params:        parameters for the statistic calculations  and data description
            input_data:    data as DataFrame
        Returns:
            DataFrame with equalised data
    """

    # list all fixed variables
    fix_vals_permuted_list = []
    fix_vals_keys = []
    if 'fixed_vars_vals_input' in params:
        for key in params['fixed_vars_vals_input']:
            if type(params['fixed_vars_vals_input'][key]) is dict:
                list_for_permut = params['fixed_vars_vals_input'][key].values()
            else:
                list_for_permut = [params['fixed_vars_vals_input'][key]]
            vals_permuted = list(itertools.product(*list_for_permut))
            vals_permuted_list = [item for sublist in vals_permuted for item in sublist]
            fix_vals_permuted_list.append(vals_permuted_list)

        fix_vals_keys = list(params['fixed_vars_vals_input'].keys())

    # perform EE for each forecast variable on the axis 1
    output_ee_data = \
        equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, params, input_data, axis='1')

    # if the second Y axis is present - run event equalizer on Y1
    # and then run event equalizer on Y1 and Y2 equalized data
    if 'series_val_2' in params.keys() and params['series_val_2']:
        # perform EE for each forecast variable on the axis 2
        output_ee_data_2 = \
            equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, params, input_data, axis='2')

        # append and reindex output from both axis
        all_ee_records = pd.concat([output_ee_data, output_ee_data_2]).reindex()

        # create a single unique dictionary from series for Y1 and Y2 to us in EE
        all_series = {**params['series_val_1'], **params['series_val_2']}
        for key in all_series:
            all_series[key] = list(set(all_series[key]))

        # run EE on run event equalizer on Y1 and Y2
        output_ee_data = event_equalize(all_ee_records, params['indy_var'],
                                        all_series,
                                        fix_vals_keys,
                                        fix_vals_permuted_list, True,
                                        params['line_type'] == "ssvar")

        output_ee_data = output_ee_data.drop('equalize', axis=1)

    return output_ee_data


def create_permutations(input_list):
    """
       Create all permutations (ie cartesian products) between the
       elements in the lists of list under the input_list
       of lists:

       for example:

       input:
          model:
            - GFS_0p25_G193
          vx_mask:
            - NH_CMORPH_G193
            - SH_CMORPH_G193
            - TROP_CMORPH_G193


        So for the above case, we have two lists in the input_dict dictionary,
        one for model and another for vx_mask:
        model_list = ["GFS_0p25_G193"]
        vx_mask_list = ["NH_CMORPH_G193", "SH_CMORPH_G193", "TROP_CMORPH_G193"]

        and a cartesian product representing all permutations of the lists
        above results in the following:

       ("GFS_0p25_G193", "NH_CMORPH_G193")
       ("GFS_0p25_G193", "SH_CMORPH_G193")
       ("GFS_0p25_G193", "TROP_CMORPH_G193")

       Args:
            input_list: an input list containing lists of values to
                        permute
       Returns:
           permutation: a list of tuples that represent the possible
           permutations of values in all lists
    """

    # Retrieve the lists from the input_dict dictionary
    vals_list = input_list

    # Utilize itertools' product() to create the cartesian product of all elements
    # in the lists to produce all permutations of the values in the lists.
    permutations = [p for p in itertools.product(*vals_list)]

    return permutations


def compute_std_err_from_mean(data):
    """
    Function to compute the Standard Error of an uncorrelated time series using mean
    Arg:
        data: array of values presorted by date/time

    Returns: Standard Error, variance inflation factor flag, AR1 coefficient, the length of data
    """
    ratio_flag = 0
    variance = st.variance(data)

    number_of_none = sum(x is None for x in data)
    if variance > 0.0 and (len(data) - number_of_none) > 2:
        # Compute the first order auto-correlation coefficient.
        ar_1 = autocor_coef(data)

        # Compute a variance inflation factor
        # (having removed that portion of the time series that was correlated).
        ratio = (1 + ar_1) / (1 - ar_1)

        # Check for a zero RATIO, that will then be operated on by SQRT.
        # If necessary, try a different arima method, or just set RATIO to one.
        if ratio < 0.0:
            ratio = 1.0
            ratio_flag = 1

        variance_inflation_factor = math.sqrt(ratio)

        # If the AR1 coefficient is less than 0.3, then don't even use a vif!  Set vif = 1.0
        if ar_1 < 0.3 or ar_1 >= 0.99:
            variance_inflation_factor = 1.0

        # Compute the Standard Error using the variance inflation factor.
        std_err = variance_inflation_factor * math.sqrt(variance) / math.sqrt(len(data))

    else:
        std_err = 0.0
        ar_1 = 0.0
    return std_err, ratio_flag, ar_1, len(data)


def compute_std_err_from_median_no_variance_inflation_factor(data):
    """
    Function to compute the Standard Error of an uncorrelated time series.
    Remove the correlated portion of a time series, using a first order auto-correlation coefficient
    to help compute an inflated variance factor for the uncorrelated portion of the time series.
    Originator Rscript:  Eric Gilleland and Andrew Loughe, 08 JUL 2008
    Arg:
        data: array of values presorted by date/time

    Returns: Standard Error, variance inflation factor flag, AR1 coefficient, the length of data
    """
    data_without_non = []
    for val in data:
        if val is not None and not np.isnan(val):
            data_without_non.append(val)
    iqr = stats.iqr(data_without_non, interpolation='linear')
    number_of_none = sum(x is None or np.isnan(x) for x in data)
    if iqr > 0.0 and (len(data) - number_of_none) > 2:
        # Compute the Standard Error using the variance inflation factor.
        std_err = (iqr * math.sqrt(math.pi / 2.)) / (1.349 * math.sqrt(len(data) - number_of_none))
    else:
        std_err = 0.0
    return std_err, 0, 0, len(data) - number_of_none


def compute_std_err_from_median_variance_inflation_factor(data):
    """
        Function to compute the Standard Error of an uncorrelated time series
        from median variance inflation factor.
        Arg:
            data: array of values presorted by date/time

        Returns: Standard Error, variance inflation factor flag, AR1 coefficient, the length of data
    """
    data_without_non = []
    for val in data:
        if val is not None and not np.isnan(val):
            data_without_non.append(val)
    ratio_flag = 0
    iqr = stats.iqr(data_without_non, interpolation='linear')
    number_of_none = sum(x is None or np.isnan(x) for x in data)

    if iqr > 0.0 and (len(data) - number_of_none) > 2:
        # Compute the first order auto-correlation coefficient
        # using a vector that is the same size as "data", but represents
        # represents excusions from the median of the data.

        # Use excursions from the median to compute the first order auto-correlation coefficient.
        data_excursions = list()
        median = st.median(data)
        for val in data:
            if val >= median:
                data_excursions.append(1)
            else:
                data_excursions.append(0)

        arima = ARIMA(data_excursions, order=(1, 0, 0))
        ar_1 = arima.fit().arparams[0]

        # Compute an variance inflation factor
        # (having removed that portion of the time series that was correlated).
        ratio = (1 + ar_1) / (1 - ar_1)

        # Check for a zero RATIO, that will then be operated on by SQRT.
        if ratio < 0.0:
            ratio = 1.0
            ratio_flag = 1

        variance_inflation_factor = math.sqrt(ratio)

        # If the AR1 coefficient is less than 0.3, then don't even use a vif!  Set vif = 1.0
        if ar_1 < 0.3 or ar_1 >= 0.99:
            variance_inflation_factor = 1.0

        # Compute the Standard Error using the variance inflation factor.
        iqr = stats.iqr(data, interpolation='linear')
        std_err = variance_inflation_factor * (iqr * math.sqrt(math.pi / 2.)) / (1.349 * math.sqrt(len(data)))

    else:
        std_err = 0.0
        ar_1 = 0
    return std_err, ratio_flag, ar_1, len(data)


def compute_std_err_from_sum(data):
    """
        Function to compute the Standard Error of an uncorrelated time series
        from sum.
        Arg:
            data: array of values presorted by date/time

        Returns: Standard Error, variance inflation factor flag, AR1 coefficient, the length of data
    """
    std_err = compute_std_err_from_mean(data)

    # multiply Standard Error by data size
    return std_err[0] * len(data), std_err[1], std_err[2], std_err[3]


def convert_lon_360_to_180(longitude):
    """
        Convert a list or numpy array of longitudes from 0,360 to -180 to 180 (West-East)

        Args:
        @params

        longitude: a numpy array or python list containing integer or float values from 0 to 360
                   to be converted to values from -180 to 180

        Returns:
            a numpy array containing values that range from -180 to 180 (West to East lons)
            Maintains the input type, ie if longitudes are int, then the numpy array returned will
            consist of int64.  If longitudes are float, then the returned numpy array will consist of
            float.
    """

    # First, convert lists to numpy array
    lons = np.asarray(longitude)

    # Use formula ((lons + 180) % 360) - 180 where % is the modulo operator
    west_east_lons = np.mod((lons + 180), 360) - 180
    negative_to_positive = np.sort(west_east_lons)

    return negative_to_positive


def convert_lons_indices(lons_in, minlon_in, range_in):
    '''

    Input:
    @param lons_in: A list of longitudes to convert
    @param minlon_in: The minimum value/starting value of converted longitudes
    @param range_in: The number of longitudes to convert

    Returns:
      reordered_lons:  sorted array of longitudes
      lonsortlocs:  sorted array indices
    '''

    minlon = abs(minlon_in)

    # Use formula to convert longitude values based on the starting lon value and
    # the target number of longitudes
    newlons = np.mod((lons_in + minlon), range_in) - minlon

    # get the sorted array indices
    lonsortlocs = np.argsort(newlons)

    # get the sorted, converted array
    reordered_lons = newlons[lonsortlocs]

    return reordered_lons, lonsortlocs


def create_permutations_mv(fields_values: Union[dict, list], index: int) -> list:
    """
    Creates a list of all permutations of the dictionary or list values using METviewer logic
    Input:
    :param fields_values: dictionary of field-values, where values are lists
        or a list of lists
    :param index: the regression index
    :return: the list of permutations
    """

    if isinstance(fields_values, dict):
        # remove all values with len = 0
        fields_values_clean = {}
        for key, value in fields_values.items():
            if len(value) > 0:
                fields_values_clean[key] = value

        keys = list(fields_values_clean.keys())
        # return an empty list if the dictionary is empty
        if len(keys) == 0:
            return []

        values = fields_values_clean[keys[index]]
        # if the index has reached the end of the list, return the selected values
        # from the last control
        if len(keys) == index + 1:
            return values
        # otherwise, get the list for the next fcst_var and build upon it
        val_next = create_permutations_mv(fields_values_clean, index + 1)
    else:
        if len(fields_values) == 0:
            return []

        values = fields_values[index]
        # if the index has reached the end of the list, return the selected values
        # from the last control
        if len(fields_values) == index + 1:
            return values

        # otherwise, get the list for the next fcst_var and build upon it
        val_next = create_permutations_mv(fields_values, index + 1)

    if len(values) == 0:
        return val_next

    result = []
    for val_next_el in val_next:
        for list_val_el in values:

            if isinstance(val_next_el, list):
                # prepend value to the existing list and add it to the result
                val_next_el_cp = val_next_el.copy()
                val_next_el_cp.insert(0, list_val_el)
                result.append(val_next_el_cp)
            else:
                # create a new array and add it to the result
                result.append([list_val_el, val_next_el])
    return result


def pt(q, df, ncp=0, lower_tail=True):
    """
    Calculates the cumulative of the t-distribution

    Args:
        q - vector of quantiles
        df - degrees of freedom (> 0)
        ncp - array_like shape parameters
        lower_tail - if True (default), probabilities are P[X ≤ x], otherwise, P[X > x].

    Returns:
        the cumulative of the t-distribution
    """

    if ncp == 0:
        result = t.cdf(x=q, df=df, loc=0, scale=1)
    else:
        result = nct.cdf(x=q, df=df, nc=ncp, loc=0, scale=1)
    if lower_tail is False:
        result = 1 - result
    return result


def qt(p, df, ncp=0):
    """
    Calculates the quantile function of the t-distribution

     Args:
        p - array_like quantiles
        df - array_like shape parameters
        ncp - array_like shape parameters

    Returns:
        tquantile function of the t-distribution
    """
    if ncp == 0:
        result = t.ppf(q=p, df=df, loc=0, scale=1)
    else:
        result = nct.ppf(q=p, df=df, nc=ncp, loc=0, scale=1)
    return result


def tost_paired(n: int, m1: float, m2: float, sd1: float, sd2: float, r12: float, low_eqbound_dz: float,
                high_eqbound_dz: float, alpha: float = None) -> dict:
    """
    TOST function for a dependent t-test (Cohen's dz). Based on Rscript function TOSTpaired

    Args:
        n: sample size (pairs)
        m1: mean of group 1
        m2: mean of group 2
        sd1: standard deviation of group 1
        sd2: standard deviation of group 2
        r12: correlation of dependent variable between group 1 and group 2
        low_eqbound_dz: lower equivalence bounds (e.g., -0.5) expressed in standardized mean difference (Cohen's dz)
        high_eqbound_dz: upper equivalence bounds (e.g., 0.5) expressed in standardized mean difference (Cohen's dz)
        alpha: alpha level (default = 0.05)

    Returns:
        Returns a dictionary with calculated TOST values
                dif - Mean Difference
                t - TOST t-values 1 and 2 as a tuple
                p - TOST p-values and 2 as a tuple
                degrees_of_freedom - degrees of freedom
                ci_tost - confidence interval TOST Lower and Upper limit as a tuple
                ci_ttest - confidence interval TTEST Lower and Upper limit as a tuple
                eqbound - equivalence bound low and high as a tuple
                xlim - limits for x-axis
                combined_outcome - outcome
                test_outcome - pt test outcome
                tist_outcome - TOST outcome

    """
    if not alpha:
        alpha = 0.05
    if low_eqbound_dz >= high_eqbound_dz:
        print(
            'WARNING: The lower bound is equal to or larger than the upper bound.'
            ' Check the plot and output to see if the bounds are specified as you intended.')

    if n < 2:
        print("The sample size should be larger than 1.")
        sys.exit()

    if 1 <= alpha or alpha <= 0:
        print("The alpha level should be a positive value between 0 and 1.")
        sys.exit()
    if sd1 <= 0 or sd2 <= 0:
        print("The standard deviation should be a positive value.")
        sys.exit()
    if 1 < r12 or r12 < -1:
        print("The correlation should be a value between -1 and 1.")
        sys.exit()

    sdif = math.sqrt(sd1 * sd1 + sd2 * sd2 - 2 * r12 * sd1 * sd2)
    low_eqbound = low_eqbound_dz * sdif
    high_eqbound = high_eqbound_dz * sdif
    se = sdif / math.sqrt(n)
    degree_f = n - 1

    if se != 0:
        t = (m1 - m2) / se
        pttest = 2 * pt(abs(t), degree_f, lower_tail=False)
        t1 = ((m1 - m2) - (low_eqbound_dz * sdif)) / se
        p1 = pt(t1, degree_f, lower_tail=False)
        t2 = ((m1 - m2) - (high_eqbound_dz * sdif)) / se
        p2 = pt(t2, degree_f, lower_tail=True)
        ptost = max(p1, p2)
    else:
        pttest = None
        t1 = None
        p1 = None
        t2 = None
        p2 = None
        ptost = None

    ll90 = ((m1 - m2) - qt(1 - alpha, degree_f) * se)
    ul90 = ((m1 - m2) + qt(1 - alpha, degree_f) * se)

    dif = (m1 - m2)
    ll95 = ((m1 - m2) - qt(1 - (alpha / 2), degree_f) * se)
    ul95 = ((m1 - m2) + qt(1 - (alpha / 2), degree_f) * se)
    xlim_l = min(ll90, low_eqbound) - max(ul90 - ll90, high_eqbound - low_eqbound) / 10
    xlim_u = max(ul90, high_eqbound) + max(ul90 - ll90, high_eqbound - low_eqbound) / 10

    if pttest and ptost:
        if pttest <= alpha and ptost <= alpha:
            combined_outcome = 'diff_eqv'

        if pttest < alpha and ptost > alpha:
            combined_outcome = 'diff_no_eqv'

        if pttest > alpha and ptost <= alpha:
            combined_outcome = 'no_diff_eqv'

        if pttest > alpha and ptost > alpha:
            combined_outcome = 'no_diff_no_eqv'

        if pttest < alpha:
            test_outcome = 'significant'
        else:
            test_outcome = 'non-significant'

        if ptost < alpha:
            tost_outcome = 'significant'
        else:
            tost_outcome = 'non-significant'

        t = (round_half_up(t1, PRECISION), round_half_up(t2, PRECISION))
        p = (round_half_up(p1, PRECISION), round_half_up(p2, PRECISION))
    else:
        combined_outcome = 'none'
        tost_outcome = 'none'
        test_outcome = 'none'
        t = (None, None)
        p = (None, None)

    return {
        'dif': round_half_up(dif, PRECISION),
        't': t,
        'p': p,
        'degrees_of_freedom': round_half_up(degree_f, PRECISION),
        'ci_tost': (round_half_up(ll90, PRECISION), round_half_up(ul90, PRECISION)),
        'ci_ttest': (round_half_up(ll95, PRECISION), round_half_up(ul95, PRECISION)),
        'eqbound': (round_half_up(low_eqbound, PRECISION), round_half_up(high_eqbound, PRECISION)),
        'xlim': (round_half_up(xlim_l, PRECISION), round_half_up(xlim_u, PRECISION)),
        'combined_outcome': combined_outcome,
        'test_outcome': test_outcome,
        'tost_outcome': tost_outcome
    }


def calculate_mtd_revision_stats(series_data: DataFrame, lag_max: Union[int, None] = None) -> dict:
    """
    Calculates Mode-TD revision stats
    :param series_data - DataFrame with columns 'stat_value' and 'revision_id'
    :param  lag_max - maximum lag at which to calculate the acf.
                Can be an integer or None (the default).
                Default is 10*log10(N/m) where N is the number of
                observations and m the number of series.
                Will be automatically limited to one less than the number of
                observations in the series.
    :return: a dictionary containing this statistics:
            ww_run -  p-value of the Wald-Wolfowitz runs test
            auto_cor_p - p-value of autocorrelation
            auto_cor_r - estimated  autocorrelation for lag_max
    """
    result = {
        'ww_run': None,
        'auto_cor_p': None,
        'auto_cor_r': None
    }
    if len(series_data) == 0:
        return result
    if not {'stat_value', 'revision_id'}.issubset(series_data.columns):
        print("DataFrame doesn't have correct columns")
        return result

    unique_ids = series_data.revision_id.unique()

    data_for_stats = []
    for revision_id in unique_ids:
        data_for_id = series_data[series_data['revision_id'] == revision_id]['stat_value'].tolist()
        data_for_stats.extend(data_for_id)
        data_for_stats.extend([None])

    data_for_stats.pop()

    # subtract mean from each value
    mean_val = st.mean(remove_none(data_for_stats))

    def func(a):
        if a is not None:
            return a - mean_val
        return None

    data_for_stats = list(map(func, data_for_stats))

    acf_value = acf(data_for_stats, 'correlation', lag_max)
    if acf_value is not None:
        result['auto_cor_r'] = round(acf_value[-1], 2)

    # qnorm((1 + 0.05)/2) = 0.06270678
    result['auto_cor_p'] = round(0.06270678 / math.sqrt(np.size(data_for_stats)), 2)

    p_value = runs_test(data_for_stats, 'left.sided', 'median')['p_value']
    if p_value is not None:
        result['ww_run'] = round(p_value, 2)

    return result


def sort_data(series_data):
    """ Sorts input data frame by fcst_valid, fcst_lead and stat_name

        Args:
            input pandas data frame
    """
    fields = series_data.keys()
    if "fcst_valid_beg" in fields:
        by_fields = ["fcst_valid_beg", "fcst_lead"]
    elif "fcst_valid" in fields:
        by_fields = ["fcst_valid", "fcst_lead"]
    elif "fcst_init_beg" in fields:
        by_fields = ["fcst_init_beg", "fcst_lead"]
    else:
        by_fields = ["fcst_init", "fcst_lead"]
    if "stat_name" in fields:
        by_fields.append("stat_name")
    series_data = series_data.sort_values(by=by_fields)
    return series_data


def autocor_coef(data: list) -> Union[None, float]:
    """ Calculate the least-squares estimate of the lag-1 regression
         or autocorrelation coefficient
         :param data: input data array
         :return: am autocorrelation coefficient or None
    """

    # remove None and nan
    data_valid = [i for i in data if i is not None and not np.isnan(i)]
    # if the list is too short - return None
    if len(data_valid) < 2:
        return None

    x_i = data_valid[0: len(data_valid) - 1]
    y_i = data_valid[1: len(data_valid)]
    data_mean = st.mean(data_valid)

    x = [x - data_mean for x in x_i]
    y = [x - data_mean for x in y_i]

    xx = [m ** 2 for m in x]
    xy = []
    for item_x, item_y in zip(x, y):
        xy.append(item_x * item_y)

    sx = sum(x)
    sy = sum(y)
    sxx = sum(xx)
    sxy = sum(xy)

    n = len(data_valid)
    return sx * sy / (sx - (n - 1) * sxx) + sxy / (sxx - sx * sx / (n - 1))
