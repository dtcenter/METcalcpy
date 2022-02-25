# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: met_stats.py
"""
import math
import numpy as np

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def get_column_index_by_name(columns, column_name):
    """Finds the index of the specified column in the array

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


def calc_direction(u_comp, v_comp):
    """ Calculated the direction of the wind from it's u and v components in degrees
        Args:
            u_comp: u wind component
            v: v wind component

        Returns:
            direction of the wind in degrees or None if one of the components is less then tolerance
    """
    tolerance = 1e-5
    if abs(u_comp) < tolerance and abs(v_comp) < tolerance:
        return None

    direction = np.arctan2(u_comp, v_comp)
    # convert to [0,360]
    direction = direction - 360 * math.floor(direction / 360)
    return direction


def calc_speed(u_comp, v_comp):
    """ Calculated the speed of the wind from it's u and v components
        Args:
            u_comp: u wind component
            v_comp: v wind component

        Returns:
            speed of the wind  or None
    """
    try:
        result = np.sqrt(u_comp * u_comp + v_comp * v_comp)
    except (TypeError, Warning):
        result = None
    return result
