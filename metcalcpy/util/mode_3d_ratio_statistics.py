# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
from metcalcpy.util.mode_ratio_statistics import *
from metcalcpy.util.utils import column_data_by_name_value, THREE_D_DATA_FILTER
from metcalcpy.util.safe_log import safe_log

def calculate_3d_ratio_fsa_asa(input_data, columns_names, logger=None):
    """Performs calculation of  % of 3d simple objects that are forecast
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio FSA/ASA.")
        result = calculate_ratio_fsa_asa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio FSA/ASA: {str(e)}.")
        return None


def calculate_3d_ratio_osa_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio OSA/ASA.")
        result = calculate_ratio_osa_asa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio OSA/ASA: {str(e)}.")
        return None


def calculate_3d_ratio_asm_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio ASM/ASA.")
        result = calculate_ratio_asm_asa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio ASM/ASA: {str(e)}.")
        return None


def calculate_3d_ratio_asu_asa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio ASU/ASA.")
        result = calculate_ratio_asu_asa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio ASU/ASA: {str(e)}.")
        return None


def calculate_3d_ratio_fsm_fsa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple forecast objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio FSM/FSA.")
        result = calculate_ratio_fsm_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio FSM/FSA: {str(e)}.")
        return None


def calculate_3d_ratio_fsu_fsa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple forecast objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio FSU/FSA.")
        result = calculate_ratio_fsu_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio FSU/FSA: {str(e)}.")
        return None


def calculate_3d_ratio_osm_osa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple simple observation objects that are matched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio OSM/OSA.")
        result = calculate_ratio_osm_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio OSM/OSA: {str(e)}.")
        return None

def calculate_3d_ratio_osu_osa(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple simple observation objects that are unmatched

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio OSU/OSA.")
        result = calculate_ratio_osu_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio OSU/OSA: {str(e)}.")
        return None


def calculate_3d_ratio_fsm_asm(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple matched objects that are forecasts

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio FSM/ASM.")
        result = calculate_ratio_fsm_asm(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio FSM/ASM: {str(e)}.")
        return None


def calculate_3d_ratio_osm_asm(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple matched objects that are observations

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio OSM/ASM.")
        result = calculate_ratio_osm_asm(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio OSM/ASM: {str(e)}.")
        return None


def calculate_3d_ratio_fsu_asu(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple unmatched objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio FSU/ASU.")
        result = calculate_ratio_fsu_asu(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio FSU/ASU: {str(e)}.")
        return None


def calculate_3d_ratio_osu_asu(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d simple unmatched objects that are observation

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3d ratio OSU/ASU.")
        result = calculate_ratio_osu_asu(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3d ratio OSU/ASU: {str(e)}.")
        return None


def calculate_3d_ratio_fsa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FSA/AAA.")
        result = calculate_ratio_fsa_aaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FSA/AAA: {str(e)}.")
        return None


def calculate_3d_ratio_osa_aaa(input_data, columns_names, logger=None):
    """Performs calculation of ?

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OSA/AAA.")
        result = calculate_ratio_osa_aaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OSA/AAA: {str(e)}.")
        return None


def calculate_3d_ratio_fsa_faa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 3d forecast objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FSA/FAA.")
        result = calculate_ratio_fsa_faa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FSA/FAA: {str(e)}.")
        return None


def calculate_3d_ratio_fca_faa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 3d forecast objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FCA/FAA.")
        result = calculate_ratio_fca_faa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FCA/FAA: {str(e)}.")
        return None


def calculate_3d_ratio_osa_oaa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 3d observation objects that are simple

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OSA/OAA.")
        result = calculate_ratio_osa_oaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OSA/OAA: {str(e)}.")
        return None


def calculate_3d_ratio_oca_oaa(input_data, columns_names, logger=None):
    """Performs calculation of % of all 3d observation objects that are cluster

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OCA/OAA.")
        result = calculate_ratio_oca_oaa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OCA/OAA: {str(e)}.")
        return None


def calculate_3d_ratio_fca_aca(input_data, columns_names, logger=None):
    """Performs calculation of % of 3d cluster objects that are forecast

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FCA/ACA.")
        result = calculate_ratio_fca_aca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FCA/ACA: {str(e)}.")
        return None


def calculate_3d_ratio_fsa_osa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of simple 3d forecasts to simple 3d observations [frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FSA/OSA (frequency bias).")
        result = calculate_ratio_fsa_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FSA/OSA: {str(e)}.")
        return None


def calculate_3d_ratio_osa_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of simple 3d observations to simple 3d forecasts [1 / frequency bias]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OSA/FSA [1 / frequency bias].")
        result = calculate_ratio_osa_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OSA/FSA: {str(e)}.")
        return None


def calculate_3d_ratio_asa_aca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of simple 3d objects to 3d cluster objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio ASA/ACA (simple to cluster objects).")
        result = calculate_ratio_asa_aca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio ASA/ACA: {str(e)}.")
        return None


def calculate_3d_ratio_fca_fsa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of 3d cluster forecast objects to 3d simple forecast objects'

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FCA/FSA (cluster to simple forecast objects).")
        result = calculate_ratio_fca_fsa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FCA/FSA: {str(e)}.")
        return None


def calculate_3d_ratio_fsa_fca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of simple 3d forecast objects to cluster 3d forecast objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio FSA/FCA (simple to cluster forecast objects).")
        result = calculate_ratio_fsa_fca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio FSA/FCA: {str(e)}.")
        return None


def calculate_3d_ratio_oca_osa(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of cluster 3d observation objects to simple 3d observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OCA/OSA (cluster to simple observation objects).")
        result = calculate_ratio_oca_osa(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OCA/OSA: {str(e)}.")
        return None


def calculate_3d_ratio_osa_oca(input_data, columns_names, logger=None):
    """Performs calculation of Ratio of simple 3d observation objects to cluster 3d observation objects

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D ratio OSA/OCA (simple to cluster observation objects).")
        result = calculate_ratio_osa_oca(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D ratio OSA/OCA: {str(e)}.")
        return None


def calculate_3d_objhits(input_data, columns_names, logger=None):
    """Performs calculation of Hits =/2

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object hits.")
        result = calculate_objhits(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object hits: {str(e)}.")
        return None


def calculate_3d_objmisses(input_data, columns_names, logger=None):
    """Performs calculation of Misses = OSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object misses.")
        result = calculate_objmisses(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object misses: {str(e)}.")
        return None


def calculate_3d_objfas(input_data, columns_names, logger=None):
    """Performs calculation of False Alarms = FSU

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object false alarms.")
        result = calculate_objfas(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object false alarms: {str(e)}.")
        return None


def calculate_3d_objcsi(input_data, columns_names, logger=None):
    """Performs calculation of critical success index CSI = hits //2 + OSU + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object critical success index (CSI).")
        result = calculate_objcsi(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object critical success index (CSI): {str(e)}.")
        return None


def calculate_3d_objpody(input_data, columns_names, logger=None):
    """Performs calculation of Probability of Detecting Yes PODY = hits //2 + OSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object Probability of Detecting Yes (PODY).")
        result = calculate_objpody(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object Probability of Detecting Yes (PODY): {str(e)}.")
        return None


def calculate_3d_objfar(input_data, columns_names, logger=None):
    """Performs calculation of False alarm ratio FAR = false alarms //2 + FSU]

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated statistic as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Filtering data based on THREE_D_DATA_FILTER.")
        filtered_data = column_data_by_name_value(input_data, columns_names, THREE_D_DATA_FILTER)

        safe_log(logger, "debug", "Calculating 3D object False Alarm Ratio (FAR).")
        result = calculate_objfar(filtered_data, columns_names)

        safe_log(logger, "debug", f"Calculation complete. Result: {result}.")
        return result
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Failed to calculate 3D object False Alarm Ratio (FAR): {str(e)}.")
        return None
