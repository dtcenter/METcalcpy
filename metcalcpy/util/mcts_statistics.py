# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mcts_statistics.py
"""
import warnings
import numpy as np
from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION

__author__ = 'Tatiana Burek'


def calculate_mcts_hss_ec(input_data, columns_names):
    """Performs calculation of HSS_EC - a skill score based on Accuracy,

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated HSS_EC as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')

    try:
        row = input_data[0].copy()
        n_cat = row[np.where(columns_names == 'n_cat')[0][0]]
        ec_value = row[np.where(columns_names == 'ec_value')[0][0]]

        # aggregate all fi_oj in one row
        for index in range(n_cat * n_cat):
            column_name = 'fi_oj_' + str(index)
            row[np.where(columns_names == column_name)[0][0]] = \
                sum_column_data_by_name(input_data, columns_names, column_name)

        # init contingency table
        cont_table = [[0] * n_cat for _ in range(n_cat)]

        # fill contingency table
        for index in range(n_cat * n_cat):
            i_value = row[np.where(columns_names == 'i_value_' + str(index))[0][0]]
            j_value = row[np.where(columns_names == 'j_value_' + str(index))[0][0]]
            fi_oj = row[np.where(columns_names == 'fi_oj_' + str(index))[0][0]]
            cont_table[i_value - 1][j_value - 1] = fi_oj

        # calculate the sum of the counts on the diagonal and
        # the sum of the counts across the whole MCTC table
        diag_count = sum([cont_table[i][j] for i in range(n_cat) for j in range(n_cat) if i == j])
        sum_all = sum(sum(cont_table, []))
        result = (diag_count - (ec_value * sum_all)) / (sum_all - (ec_value * sum_all))
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result
