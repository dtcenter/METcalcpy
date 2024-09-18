# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: event_equalize_against_values.py
"""

import pandas as pd

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'

def safe_log(logger, log_level, message):
    """
    Safely logs a message using the provided logger and log level.
    
    Args:
        logger (logging.Logger): The logger object. If None, the message will not be logged.
        log_level (str): The logging level to use (e.g., "info", "debug").
        message (str): The message to log.
    """
    if logger:
        log_method = getattr(logger, log_level, None)
        if callable(log_method):
            log_method(message)
        
def event_equalize_against_values(series_data, input_unique_cases, logger):
    """Performs event equalisation.

    event_equalize_against_values assumes that the input series_data contains data
    indexed by fcst_valid, series values and the independent variable values.
    It builds a new data frame which contains the same
    data except for records that don't have corresponding fcst_valid
    and fcst_lead values from ee_stats_equalize

    Args:
        series_data: data frame containing the records to equalize, including fcst_valid_beg, series
                  values and independent variable values

        input_unique_cases: unique cases to equalize against
    Returns:
        A data frame that contains equalized records or empty frame
    """

    warning_remove = "WARNING: event equalization removed {} rows"

    safe_log(logger, "info", "Starting event equalization.")

    column_names = list(series_data)

    if 'fcst_valid' in column_names:
        # always use fcst_valid for equalization
        # create a unique member to use for equalization
        series_data.insert(len(series_data.columns), 'equalize',
                           series_data['fcst_valid'].astype(str)
                           + ' '
                           + series_data['fcst_lead'].astype(str))
    else:
        safe_log(logger, "warning", "WARNING: eventEqualize() did not run due to lack of valid time field.")
        print("WARNING: eventEqualize() did not run due to lack of valid time field")
        return pd.DataFrame()

    # create an equalized set of data for the minimal list of dates based on the input cases
    data_for_unique_cases = series_data[(series_data['equalize'].isin(input_unique_cases))]
    n_row_cases = len(data_for_unique_cases)
    if n_row_cases == 0:
        safe_log(logger, "warning", "WARNING: discarding all members. No matching cases found.")
        print(" WARNING: discarding all members")
        return pd.DataFrame()

    n_row_ = len(series_data)
    if n_row_cases != n_row_:
        safe_log(logger, "warning", warning_remove.format(n_row_ - n_row_cases))
        print(warning_remove.format(n_row_ - n_row_cases))

    # remove 'equalize' column
    data_for_unique_cases = data_for_unique_cases.drop(['equalize'], axis=1)
    safe_log(logger, "info", "Event equalization completed successfully.")
    return data_for_unique_cases
