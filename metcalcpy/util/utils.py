"""
Program Name: statistics.py
"""

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

import math
import itertools
import numpy as np
import pandas as pd

from metcalcpy.event_equalize import event_equalize

OPERATION_TO_SIGN = {
    'DIFF': '-',
    'RATIO': '/',
    'SS': 'and',
    'DIFF_SIG': '-'
}
STR_TO_BOOL = {'True': True, 'False': False}

# precision value for statistics calculations
PRECISION = 7

TWO_D_DATA_FILTER = {'object_type': '2d'}
THREE_D_DATA_FILTER = {'object_type': '3d'}


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


def get_derived_curve_name(list_of_names):
    """Creates the derived series name from the list of series name components

         Args:
             list_of_names: list of series name components
                1st element - name of 1st series
                2st element - name of 2st series
                3st element - operation. Can be 'DIFF','RATIO', 'SS', 'SINGLE'

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
            n: a number
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
    # find the index of specified column
    index_array = np.where(columns == column_name)[0]
    if index_array.size == 0:
        return None

    # get column's data and convert it into float array
    try:
        data_array = np.array(input_data[:, index_array[0]], dtype=np.float64)
    except IndexError:
        data_array = None

    if data_array is None or np.isnan(data_array).all():
        return None

    try:
        if rm_none:
            result = np.nansum(data_array.astype(np.float))
        else:
            if np.isnan(data_array).any():
                result = None
            else:
                result = sum(data_array.astype(np.float))
    except TypeError:
        result = None

    return result


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
                           'PSTD_ROC_AUC', 'NBR_UFSS', 'NBR_F_RATE', 'NBR_O_RATE',
                           'NBR_BASER', 'NBR_FMEAN')

    zero_perf_score_stats = ('POFD', 'FAR', 'ESTDEV', 'MAE', 'MSE', 'BCMSE',
                             'RMSE', 'E10', 'E25', 'E50', 'E75',
                             'E90', 'EIQR', 'MAD', 'ME2', 'ME', 'ESTDEV', 'ODDS',
                             'LODDS', 'VL1L2_MSE', 'VL1L2_RMSE',
                             'VL1L2_RMSVE', 'PSTD_BRIER', 'PSTD_RELIABILITY',
                             'NBR_FBS', 'VL1L2_SPEED_ERR',
                             'NBR_POFD', 'NBR_FAR', 'NBR_ODDS', 'BCRMSE')

    one_perf_score_stats = ('ACC', 'FBIAS', 'PODY', 'PODN', 'CSI', 'GSS',
                            'HK', 'HSS', 'ORSS', 'EDS', 'SEDS',
                            'EDI', 'SEDI', 'BAGSS', 'PR_CORR', 'SP_CORR',
                            'KT_CORR', 'MBIAS', 'ANOM_CORR', 'ANOM_CORR_RAW',
                            'VL1L2_BIAS', 'VL1L2_CORR',
                            'PSTD_BSS', 'PSTD_BSS_SMPL', 'NBR_FSS', 'NBR_AFSS',
                            'VAL1L2_ANOM_CORR', 'NBR_ACC',
                            'NBR_FBIAS', 'NBR_PODY',
                            'NBR_PODN', 'NBR_CSI', 'NBR_GSS', 'NBR_HK', 'NBR_HSS')

    if statistic.upper() in na_perf_score_stats:
        result = None
    elif statistic.upper() in zero_perf_score_stats \
            and abs(mean_stats_1) > abs(mean_stats_2):
        result = pval * -1
    elif statistic.upper() in one_perf_score_stats \
            and abs(mean_stats_1 - 1) > abs(mean_stats_2 - 1):
        result = pval * -1
    else:
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
                            input_data_frame = input_data_frame.append(rows_for_agg.iloc[:1])
                    else:
                        # if the aggregated field is 'fcst_lead'

                        # find rows for the aggregation and their indexes
                        rows_for_agg = input_data_frame[
                            (input_data_frame.fcst_valid_beg == valid)
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
                        input_data_frame = input_data_frame.append(rows_for_agg.iloc[:1])

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
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(np.float))
                                       / total]

    elif line_type == 'sal1l2':
        sums_data_frame['fbar'] = [np.nansum(input_df['fabar'] * input_df.total.astype(np.float))
                                   / total]
        sums_data_frame['obar'] = [np.nansum(input_df['oabar'] * input_df.total.astype(np.float))
                                   / total]
        sums_data_frame['fobar'] = [np.nansum(input_df['foabar'] * input_df.total.astype(np.float))
                                    / total]
        sums_data_frame['ffbar'] = [np.nansum(input_df['ffabar'] * input_df.total.astype(np.float))
                                    / total]
        sums_data_frame['oobar'] = [np.nansum(input_df['ooabar'] * input_df.total.astype(np.float))
                                    / total]

    elif line_type == 'vl1l2':
        column_names = ['ufbar', 'vfbar', 'uobar', 'vobar', 'uvfobar',
                        'uvffbar', 'uvoobar', 'f_speed_bar', 'o_speed_bar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(np.float))
                                       / total]

    elif line_type == 'val1l2':
        column_names = ['ufabar', 'vfabar', 'uoabar', 'voabar', 'uvfoabar', 'uvffabar', 'uvooabar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(np.float))
                                       / total]

    elif line_type == 'grad':
        column_names = ['fgbar', 'ogbar', 'mgbar', 'egbar']
        for column in column_names:
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(np.float))
                                       / total]

    elif line_type == 'nbrcnt':
        dbl_fbs = np.nansum(input_df['fbs'] * input_df.total.astype(np.float)) / total
        dbl_fss_den = np.nansum(
            (input_df['fbs'] / (1.0 - input_df['fss'])) * input_df.total.astype(np.float)) \
                      / total
        dbl_fss = 1.0 - dbl_fbs / dbl_fss_den
        dbl_f_rate = np.nansum(input_df['f_rate'] * input_df.total.astype(np.float)) / total
        dbl_o_rate = np.nansum(input_df['o_rate'] * input_df.total.astype(np.float)) / total
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
            sums_data_frame[column] = [np.nansum(input_df[column] * input_df.total.astype(np.float))
                                       / total]
    elif line_type == 'rps':
        d_rps_climo = input_df['rps'] / (1 - input_df['rpss'])
        sums_data_frame['rps'] = [np.nansum(input_df["rps"] * input_df.total.astype(np.float))
                                  / total]
        sums_data_frame['rps_comp'] = [np.nansum(input_df["rps_comp"] * input_df.total.astype(np.float))
                                  / total]
        sums_data_frame['rps_climo'] = [np.nansum(d_rps_climo * input_df.total.astype(np.float))
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
    for fcst_var, fcst_var_stats in params['fcst_var_val_' + axis].items():
        series_data_for_ee = pd.DataFrame()
        for fcst_var_stat in fcst_var_stats:
            # for each series for the specified axis
            for series_var, series_var_vals in params['series_val_' + axis].items():
                # ungroup series value if needed
                series_var_vals_no_group = []
                for val in series_var_vals:
                    split_val = val.split(',')
                    series_var_vals_no_group.extend(split_val)

                # filter input data based on fcst_var, statistic
                # and all series variables values
                series_data_for_ee = input_data[
                    (input_data['fcst_var'] == fcst_var)
                    & (input_data["stat_name"] == fcst_var_stat)
                    & (input_data[series_var].isin(series_var_vals_no_group))
                    ]
            # perform EE on filtered data
            # for SSVAR use equalization of multiple events
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
                output_ee_data = output_ee_data.append(series_data_after_ee)
    return output_ee_data.drop('equalize', axis=1)


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
            vals_permuted = list(itertools.product(*params['fixed_vars_vals_input'][key].values()))
            vals_permuted_list = [item for sublist in vals_permuted for item in sublist]
            fix_vals_permuted_list.append(vals_permuted_list)

        fix_vals_keys = list(params['fixed_vars_vals_input'].keys())

    # perform EE for each forecast variable on the axis 1
    output_ee_data = \
        equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, params, input_data, axis='1')

    # if the second Y axis is present - run event equalizer on Y1
    # and then run event equalizer on Y1 and Y2 equalized data
    if params['series_val_2']:
        # perform EE for each forecast variable on the axis 2
        output_ee_data_2 = \
            equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, params, input_data, axis='2')

        # append and reindex output from both axis
        all_ee_records = output_ee_data.append(output_ee_data_2).reindex()

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


def create_permutations(input_dict):
        """
           Create all permutations (ie cartesian products) between the
           elements in the lists of dictionaries under the input_dict
           dictionary:

           for example:

           input_dict:
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
                input_dict: an input dictionary containing lists of values to
                            permute
           Returns:
               permutation: a list of tuples that represent the possible
               permutations of values in all lists
        """

        # Retrieve the lists from the input_dict dictionary
        vals_list = input_dict

        # Utilize itertools' product() to create the cartesian product of all elements
        # in the lists to produce all permutations of the values in the lists.
        permutations = [p for p in itertools.product(*vals_list)]

        return permutations


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

    return west_east_lons
