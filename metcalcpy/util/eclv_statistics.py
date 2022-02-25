# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: eclv_statistics.py
"""
from typing import Union
import warnings
import numpy as np

from metcalcpy.util.ctc_statistics import calculate_economic_value
from metcalcpy.util.utils import sum_column_data_by_name

__author__ = 'Tatiana Burek'


def calculate_eclv(input_data: np.array, columns_names: np.array,
                   thresh: Union[float, None], line_type: str, cl_pts: list, add_base_rate: int = 0) \
        -> Union[dict, None]:
    """Performs calculation of ECLV - The Economic Cost Loss  Value

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            thresh: threshold value for 'pct' line type
            cl_pts: Cost loss ratio. The relative value of being unprepared
                and taking a loss to that of un-necessarily preparing. For example,
                cl = 0.1 indicates it would cost $ 1 to prevent a $10 loss.
                This defaults to the sequence 0.05 to 0.95 by 0.05.
            line_type: line type of the data 'ctc' or 'pct'
            add_base_rate: add Base rate point to cl or not (1 = add, 0 = don't add)

        Returns:
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

    # some validation
    if line_type != 'ctc' and line_type != 'pct':
        print(f'ERROR: incorrect line type {line_type} for calculating ECLV  ')
        return None
    if line_type == 'pct' and thresh is None:
        print(f'ERROR: provide thresh for calculating ECLV  ')
        return None

    try:
        if line_type == 'pct':
            index_thresh_i = np.where(columns_names == 'thresh_i')[0]
            index_oy_i = np.where(columns_names == 'oy_i')[0]
            index_on_i = np.where(columns_names == 'on_i')[0]
            thresh_i_more = input_data[:, index_thresh_i] > thresh
            thresh_i_less = input_data[:, index_thresh_i] <= thresh

            n11 = np.nansum(input_data[:, index_oy_i][thresh_i_more].astype(np.float))
            n10 = np.nansum(input_data[:, index_on_i][thresh_i_more].astype(np.float))
            n01 = np.nansum(input_data[:, index_oy_i][thresh_i_less].astype(np.float))
            n00 = np.nansum(input_data[:, index_on_i][thresh_i_less].astype(np.float))
        else:
            n11 = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
            n10 = sum_column_data_by_name(input_data, columns_names, 'fy_on')
            n01 = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
            n00 = sum_column_data_by_name(input_data, columns_names, 'fn_on')

        result = calculate_economic_value(np.array([n11, n10, n01, n00]), cl_pts, add_base_rate == 1)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result
