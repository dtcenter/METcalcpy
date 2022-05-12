# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: mode_3d_volrat_statistics.py
"""
from metcalcpy.util.mode_arearat_statistics import *
from metcalcpy.util.utils import column_data_by_name_value, THREE_D_DATA_FILTER

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_3d_volrat_fsa_asa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsa_asa(filtered_data, columns_names_new)


def calculate_3d_volrat_osa_asa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osa_asa(filtered_data, columns_names_new)


def calculate_3d_volrat_asm_asa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """

    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_asm_asa(filtered_data, columns_names_new)


def calculate_3d_volrat_asu_asa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_asu_asa(filtered_data, columns_names_new)


def calculate_3d_volrat_fsm_fsa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsm_fsa(filtered_data, columns_names_new)


def calculate_3d_volrat_fsu_fsa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsu_fsa(filtered_data, columns_names_new)


def calculate_3d_volrat_osm_osa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osm_osa(filtered_data, columns_names_new)


def calculate_3d_volrat_osu_osa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osu_osa(filtered_data, columns_names_new)


def calculate_3d_volrat_fsm_asm(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)

    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsm_asm(filtered_data, columns_names_new)


def calculate_3d_volrat_osm_asm(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osm_asm(filtered_data, columns_names_new)


def calculate_3d_volrat_osu_asu(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osu_asu(filtered_data, columns_names_new)


def calculate_3d_volrat_fsa_aaa(input_data, columns_names):
    """Performs calculation of Volume-weighted ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsa_aaa(filtered_data, columns_names_new)


def calculate_3d_volrat_osa_aaa(input_data, columns_names):
    """Performs calculation of Volume-weighted ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osa_aaa(filtered_data, columns_names_new)


def calculate_3d_volrat_fsa_faa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of all 3d forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """

    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsa_faa(filtered_data, columns_names_new)


def calculate_3d_volrat_fca_faa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of all 3d forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fca_faa(filtered_data, columns_names_new)


def calculate_3d_volrat_osa_oaa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of all 3d observation objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osa_oaa(filtered_data, columns_names_new)


def calculate_3d_volrat_oca_oaa(input_data, columns_names):
    """Performs calculation of Volume-weighted % of all 3d observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_oca_oaa(filtered_data, columns_names_new)


def calculate_3d_volrat_fca_aca(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fca_aca(filtered_data, columns_names_new)


def calculate_3d_volrat_oca_aca(input_data, columns_names):
    """Performs calculation of Volume-weighted % of 3d cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_oca_aca(filtered_data, columns_names_new)


def calculate_3d_volrat_fsa_osa(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d simple forecasts to
        3d simple observations [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)

    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsa_osa(filtered_data, columns_names_new)


def calculate_3d_volrat_osa_fsa(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d simple observations to
        3d simple forecasts [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osa_fsa(filtered_data, columns_names_new)


def calculate_3d_volrat_aca_asa(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d cluster objects to 3d simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_aca_asa(filtered_data, columns_names_new)


def calculate_3d_volrat_asa_aca(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d simple objects to 3d cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_asa_aca(filtered_data, columns_names_new)


def calculate_3d_volrat_fca_fsa(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d cluster forecast objects to
        3d simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fca_fsa(filtered_data, columns_names_new)


def calculate_3d_volrat_fsa_fca(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d simple forecast objects to
        3d cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)

    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_fsa_fca(filtered_data, columns_names_new)


def calculate_3d_volrat_oca_osa(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d cluster observation objects to
        3d simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_oca_osa(filtered_data, columns_names_new)


def calculate_3d_volrat_osa_oca(input_data, columns_names):
    """Performs calculation of Volume Ratio of 3d simple observation objects to
        3d cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_arearat_osa_oca(filtered_data, columns_names_new)


def calculate_3d_objvhits(input_data, columns_names):
    """Performs calculation of Volume 3d Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)

    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objahits(filtered_data, columns_names_new)


def calculate_3d_objvmisses(input_data, columns_names):
    """Performs calculation of Volume 3d Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objamisses(filtered_data, columns_names_new)


def calculate_3d_objvfas(input_data, columns_names):
    """Performs calculation of Volume 3d False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objafas(filtered_data, columns_names_new)


def calculate_3d_objvcsi(input_data, columns_names):
    """Performs calculation of Volume  3d critical success index CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objacsi(filtered_data, columns_names_new)


def calculate_3d_objvpody(input_data, columns_names):
    """Performs calculation of Volume  3d prob of detecting yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objapody(filtered_data, columns_names_new)


def calculate_3d_objvfar(input_data, columns_names):
    """Performs calculation of Volume 3d FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic
            or None if some of the data values are missing or invalid
    """
    columns_names_new = rename_column(columns_names)
    filtered_data = column_data_by_name_value(input_data, columns_names_new, THREE_D_DATA_FILTER)
    return calculate_objafar(filtered_data, columns_names_new)


def rename_column(columns_names):
    """Change the column name array element from 'volume' to 'area'

        Args:
            columns_names: array of column names

        Returns:
            array with changed column name
    """
    columns_names_new = []
    for index, name in enumerate(columns_names):
        if name == 'volume':
            columns_names_new.insert(index, 'area')
        else:
            columns_names_new.insert(index, name)
    return columns_names_new
