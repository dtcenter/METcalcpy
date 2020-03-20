"""
Program Name: statistics.py
"""

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

import math
import numpy as np

OPERATION_TO_SIGN = {
    'DIFF': '-',
    'RATIO': '/',
    'SS': 'and',
    'DIFF_SIG': '-'
}
STR_TO_BOOL = {'True': True, 'False': False}

# precision value for statistics calculations
PRECISION = 5

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
                             'NBR_POFD', 'NBR_FAR', 'NBR_ODDS')

    one_perf_score_stats = ('ACC', 'FBIAS', 'PODY', 'PODN', 'CSI', 'GSS',
                            'HK', 'HSS', 'ORSS', 'EDS', 'SEDS',
                            'EDI', 'SEDI', 'BAGSS', 'PR_CORR', 'SP_CORR',
                            'KT_CORR', 'MBIAS', 'ANOM_CORR',
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
