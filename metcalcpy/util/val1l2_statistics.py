# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: val1l2_statistics.py
"""
import warnings
import numpy as np

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_val1l2_anom_corr(input_data, columns_names, aggregation=False):
    """Performs calculation of VAL1L2_ANOM_CORR -

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated VAL1L2_ANOM_CORR as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        total = get_total_values(input_data, columns_names, aggregation)
        ufabar = sum_column_data_by_name(input_data, columns_names, 'ufabar') / total
        vfabar = sum_column_data_by_name(input_data, columns_names, 'vfabar') / total
        uoabar = sum_column_data_by_name(input_data, columns_names, 'uoabar') / total
        voabar = sum_column_data_by_name(input_data, columns_names, 'voabar') / total
        uvfoabar = sum_column_data_by_name(input_data, columns_names, 'uvfoabar') / total
        uvffabar = sum_column_data_by_name(input_data, columns_names, 'uvffabar') / total
        uvooabar = sum_column_data_by_name(input_data, columns_names, 'uvooabar') / total
        result = calc_wind_corr(ufabar, vfabar, uoabar, voabar, uvfoabar, uvffabar, uvooabar)
        result = round_half_up(result, PRECISION)
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        result = None
    warnings.filterwarnings('ignore')
    return result


def calc_wind_corr(uf, vf, uo, vo, uvfo, uvff, uvoo):
    """Calculates  wind correlation
        Args:
            uf - Mean(uf-uc)
            vf - Mean(vf-vc)
            uo - Mean(uo-uc)
            vo - Mean(vo-vc)
            uvfo - Mean((uf-uc)*(uo-uc)+(vf-vc)*(vo-vc))
            uvff - Mean((uf-uc)^2+(vf-vc)^2)
            uvoo - Mean((uo-uc)^2+(vo-vc)^2)

        Returns:
                calculated wind correlation as float
                or None if some of the data values are None
        """
    try:
        corr = (uvfo - uf * uo - vf * vo) / (np.sqrt(uvff - uf * uf - vf * vf)
                                             * np.sqrt(uvoo - uo * uo - vo * vo))
    except (TypeError, ZeroDivisionError, Warning, ValueError):
        corr = None
    return corr


def calculate_val1l2_total(input_data, columns_names):
    """Performs calculation of Total number of matched pairs for
        Vector Anomaly Partial Sums
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
