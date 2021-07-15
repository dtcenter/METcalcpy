"""
Program Name: mode_2d_ratio_statistics.py
"""
from metcalcpy.util.mode_ratio_statistics import *
from metcalcpy.util.utils import column_data_by_name_value, TWO_D_DATA_FILTER

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_2d_ratio_fsa_asa(input_data, columns_names):
    """Performs calculation  % of 2d simple objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """

    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsa_asa(filtered_data, columns_names)


def calculate_2d_ratio_osa_asa(input_data, columns_names):
    """Performs calculation of % of 2d simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osa_asa(filtered_data, columns_names)


def calculate_2d_ratio_asm_asa(input_data, columns_names):
    """Performs calculation of % of 2d simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_asm_asa(filtered_data, columns_names)


def calculate_2d_ratio_asu_asa(input_data, columns_names):
    """Performs calculation of % of 2d simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_asu_asa(filtered_data, columns_names)


def calculate_2d_ratio_fsm_fsa(input_data, columns_names):
    """Performs calculation of % of 2d simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsm_fsa(filtered_data, columns_names)


def calculate_2d_ratio_fsu_fsa(input_data, columns_names):
    """Performs calculation of % of 2d simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsu_fsa(filtered_data, columns_names)


def calculate_2d_ratio_osm_osa(input_data, columns_names):
    """Performs calculation of % of 2d simple simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osm_osa(filtered_data, columns_names)


def calculate_2d_ratio_osu_osa(input_data, columns_names):
    """Performs calculation of % of 2d simple simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osu_osa(filtered_data, columns_names)


def calculate_2d_ratio_fsm_asm(input_data, columns_names):
    """Performs calculation of % of 2d simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsm_asm(filtered_data, columns_names)


def calculate_2d_ratio_osm_asm(input_data, columns_names):
    """Performs calculation of % of 2d simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osm_asm(filtered_data, columns_names)


def calculate_2d_ratio_fsu_asu(input_data, columns_names):
    """Performs calculation of % of 2d simple unmatched objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsu_asu(filtered_data, columns_names)


def calculate_2d_ratio_osu_asu(input_data, columns_names):
    """Performs calculation of % of 2d simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osu_asu(filtered_data, columns_names)


def calculate_2d_ratio_fsa_aaa(input_data, columns_names):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsa_aaa(filtered_data, columns_names)


def calculate_2d_ratio_osa_aaa(input_data, columns_names):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osa_aaa(filtered_data, columns_names)


def calculate_2d_ratio_fsa_faa(input_data, columns_names):
    """Performs calculation of % of all 2d forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsa_faa(filtered_data, columns_names)


def calculate_2d_ratio_fca_faa(input_data, columns_names):
    """Performs calculation of % of all 2d forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fca_faa(filtered_data, columns_names)


def calculate_2d_ratio_osa_oaa(input_data, columns_names):
    """Performs calculation of % of all 2d observation objects that are simple'

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osa_oaa(filtered_data, columns_names)


def calculate_2d_ratio_oca_oaa(input_data, columns_names):
    """Performs calculation of % of all 2d observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_oca_oaa(filtered_data, columns_names)


def calculate_2d_ratio_fca_aca(input_data, columns_names):
    """Performs calculation of % of 2d cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fca_aca(filtered_data, columns_names)


def calculate_2d_ratio_oca_aca(input_data, columns_names):
    """Performs calculation of % of 2d cluster objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_oca_aca(filtered_data, columns_names)


def calculate_2d_ratio_fsa_osa(input_data, columns_names):
    """Performs calculation of Ratio of 2d simple forecasts to 2d simple observations
        [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsa_osa(filtered_data, columns_names)


def calculate_2d_ratio_osa_fsa(input_data, columns_names):
    """Performs calculation of Ratio of 2d simple observations to 2d simple forecasts
        [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osa_fsa(filtered_data, columns_names)


def calculate_2d_ratio_aca_asa(input_data, columns_names):
    """Performs calculation of Ratio of  2d cluster objects to 2d  simple objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_aca_asa(filtered_data, columns_names)


def calculate_2d_ratio_asa_aca(input_data, columns_names):
    """Performs calculation of Ratio of 2d simple objects to 2d cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_asa_aca(filtered_data, columns_names)


def calculate_2d_ratio_fca_fsa(input_data, columns_names):
    """Performs calculation of Ratio of 2d cluster forecast objects to 2d simple forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fca_fsa(filtered_data, columns_names)


def calculate_2d_ratio_fsa_fca(input_data, columns_names):
    """Performs calculation of Ratio of 2d simple forecast objects to 2d cluster forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_fsa_fca(filtered_data, columns_names)


def calculate_2d_ratio_oca_osa(input_data, columns_names):
    """Performs calculation of Ratio of 2d cluster observation objects
        to 2d simple observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_oca_osa(filtered_data, columns_names)


def calculate_2d_ratio_osa_oca(input_data, columns_names):
    """Performs calculation of Ratio of 2d simple observation objects to
        2d cluster observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_ratio_osa_oca(filtered_data, columns_names)


def calculate_2d_objhits(input_data, columns_names):
    """Performs calculation of 2d Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objhits(filtered_data, columns_names)


def calculate_2d_objmisses(input_data, columns_names):
    """Performs calculation of 2d Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objmisses(filtered_data, columns_names)


def calculate_2d_objfas(input_data, columns_names):
    """Performs calculation of 2d False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objfas(filtered_data, columns_names)


def calculate_2d_objcsi(input_data, columns_names):
    """Performs calculation of 2d CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objcsi(filtered_data, columns_names)


def calculate_2d_objpody(input_data, columns_names):
    """Performs calculation of 2d Probability of Detecting Yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objpody(filtered_data, columns_names)


def calculate_2d_objfar(input_data, columns_names):
    """Performs calculation of False alarm ratio FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated BASER as float
            or None if some of the data values are missing or invalid
    """
    filtered_data = column_data_by_name_value(input_data, columns_names, TWO_D_DATA_FILTER)
    return calculate_objfar(filtered_data, columns_names)
