 # ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: agg_stat_eqz.py

How to use:
 - Call from other Python function
        AGG_STAT_EVENT_EQZ = AggStatEventEqz(PARAMS)
        AGG_STAT_EVENT_EQZ.calculate_values()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python agg_stat_eqz.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_stat.eqz.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import argparse
import sys
import logging
import pandas as pd
import yaml
import warnings

from metcalcpy import GROUP_SEPARATOR
from metcalcpy.event_equalize_against_values import event_equalize_against_values
from metcalcpy.util.utils import parse_bool
from metcalcpy.logging_config import setup_logging
from metcalcpy.util.safe_log import safe_log

class AggStatEventEqz:
    """A class that performs event equalisation logic on input data
        with MODE and MTD attribute statistics
        EE is executed against previously calculated cases

        All parameters including data description and location is in the parameters dictionary
        Usage:
            initialise this call with the parameters dictionary and then
            calls perform_ee method
            This method will execute EE and save the result to the file
                AGG_STAT_EVENT_EQZ = AggStatEventEqz(PARAMS)
                AGG_STAT_EVENT_EQZ.calculate_values()
        """

    def __init__(self, in_params):
        self.logger = setup_logging(in_params)
        logger = self.logger
        safe_log(logger, "debug", "Initializing AggStatEventEqz with parameters.")
        self.params = in_params

        self.input_data = pd.read_csv(
            self.params['agg_stat_input'],
            header=[0],
            sep='\t'
        )

        self.column_names = self.input_data.columns.values
        self.series_data = None

    def calculate_values(self):
        """Performs event equalisation if needed and saves equalized data to the file.
        """
        logger = self.logger
        is_event_equal = parse_bool(self.params['event_equal'])
        safe_log(logger, "debug", f"Event equalization flag is set to: {is_event_equal}")

        # check if EE is needed
        if not self.input_data.empty and is_event_equal:
            safe_log(logger, "info", "Event equalization is required. Starting EE process.")
            try:
                safe_log(logger, "debug", f"Reading previous cases from: {self.params['agg_stat_input_ee']}")
                prev_cases = pd.read_csv(
                    self.params['agg_stat_input_ee'],
                    header=[0],
                    sep='\t'
                )
                safe_log(logger, "debug", f"Successfully read previous cases. Number of records: {len(prev_cases)}")
            except FileNotFoundError as e:
                safe_log(logger, "error", f"File not found: {self.params['agg_stat_input_ee']}", exc_info=True)
                raise
            except Exception as e:
                safe_log(logger, "error", f"Error reading previous cases: {self.params['agg_stat_input_ee']}", exc_info=True)
                raise

            # perform for axis 1
            safe_log(logger, "info", "Performing event equalization for axis 1.")
            output_ee_data = self.perform_ee_on_axis(prev_cases, '1')
            safe_log(logger, "debug", "Event equalization for axis 1 completed.")

            # perform for axis 2
            if self.params['series_val_2']:
                safe_log(logger, "info", "Series values for axis 2 detected. Performing event equalization for axis 2.")
                output_ee_data = pd.concat([output_ee_data, self.perform_ee_on_axis(prev_cases, '2')])
                safe_log(logger, "debug", "Event equalization for axis 2 completed.")
        else:
            output_ee_data = self.input_data
            if self.input_data.empty:
                safe_log(logger, "warning", 
                    'Event equalisation was not performed because the input data is empty.'
                )

        output_ee_data.to_csv(self.params['agg_stat_output'],
                              index=None, header=True, mode='w',
                              sep="\t", na_rep="NA")
        output_file = self.params['agg_stat_output']
        safe_log(logger, "info", f"Data successfully saved to {output_file}.")

    def perform_ee_on_axis(self, prev_cases, axis='1'):
        """Performs event equalisation against previously calculated cases for the selected axis
            Returns:
                A data frame that contains equalized records
        """
        warnings.filterwarnings('error')
        logger = self.logger
        safe_log(logger, "debug", f"Performing event equalization for axis {axis}.")

        output_ee_data = pd.DataFrame()
        for fcst_var, fcst_var_stats in self.params['fcst_var_val_' + axis].items():
            safe_log(logger, "debug", f"Processing forecast variable: {fcst_var}")

            for series_var, series_var_vals in self.params['series_val_' + axis].items():
                safe_log(logger, "debug", f"Processing series variable: {series_var}")
                # remove group separator from series values
                series_var_vals_no_group = []
                for val in series_var_vals:
                    split_val = val.split(GROUP_SEPARATOR)
                    series_var_vals_no_group.extend(split_val)
                safe_log(logger, "debug", f"Series variable values (no group): {series_var_vals_no_group}")

                # filter input data based on fcst_var, statistic and all series variables values
                safe_log(logger, "debug", "Filtering input data for event equalization.")
                series_data_for_ee = self.input_data[
                    (self.input_data['fcst_var'] == fcst_var)
                    & (self.input_data[series_var].isin(series_var_vals_no_group))
                    ]
                # filter previous cases on the same  fcst_var,
                # statistic and all series variables values
                series_data_for_prev_cases = prev_cases[
                    (prev_cases['fcst_var'] == fcst_var)
                    & (prev_cases[series_var].isin(series_var_vals_no_group))
                    ]
                safe_log(logger, "debug", f"Number of records after filtering input data: {len(series_data_for_ee)}")
                # get unique cases from filtered previous cases
                safe_log(logger, "debug", "Filtering previous cases for event equalization.")
                series_data_for_prev_cases_unique = series_data_for_prev_cases['equalize'].unique()
                safe_log(logger, "debug", f"Unique previous cases for event equalization: {len(series_data_for_prev_cases_unique)}")

                # perform ee
                safe_log(logger, "info", "Performing event equalization against previous cases.")
                series_data_after_ee = event_equalize_against_values(
                    series_data_for_ee,
                    series_data_for_prev_cases_unique, logger=logger)
                safe_log(logger, "debug", f"Number of records after event equalization: {len(series_data_after_ee)}")

                # append EE data to result
                if output_ee_data.empty:
                    output_ee_data = series_data_after_ee
                    safe_log(logger, "debug", "Initialized output data with first set of event equalized data.")
                else:
                    output_ee_data = pd.concat([output_ee_data, series_data_after_ee])
                    safe_log(logger, "debug", "Appended event equalized data to the output.")

        safe_log(logger, "info", f"Event equalization for axis {axis} completed. Total records: {len(output_ee_data)}")
        return output_ee_data


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat_event_eqz arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT_EVENT_EQZ = AggStatEventEqz(PARAMS)
    AGG_STAT_EVENT_EQZ.calculate_values()
