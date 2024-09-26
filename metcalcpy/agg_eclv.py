# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: agg_eclv.py

How to use:
 - Call from other Python function
        AGG_STAT = AggEclv(PARAMS)
        AGG_STAT.calculate_stats_and_ci()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python agg_eclv.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_eclv.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import math
import sys
import itertools
import argparse
import yaml
import pandas as pd
import logging
import signal
import numpy as np
from metcalcpy.bootstrap import bootstrap_and_value, BootstrapResults
from metcalcpy.event_equalize import event_equalize
from metcalcpy.util.utils import PRECISION, is_string_strictly_float
from metcalcpy.util.eclv_statistics import *

from metcalcpy.util.utils import is_string_integer, parse_bool
from metcalcpy.logging_config import setup_logging
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'

class AggEclv:
    """A class that performs aggregation statistic logic for ECLV data type on input data frame.
           All parameters including data description and location is in the parameters dictionary
           Usage:
                initialise this call with the parameters dictionary and than
                call calculate_stats_and_ci method
                This method will crate and save to the file aggregation statistics
                    agg_stat = AggEclv(params)
                    agg_stat.calculate_stats_and_ci()
            Raises: EmptyDataError or ValueError when the input DataFrame is empty
                or doesn't have data
       """
    LINE_TYPE_COLUMNS = {
        'ctc': ["total", "fy_oy", "fy_on", "fn_oy", "fn_on"],
        'pct': ["thresh_i", "oy_i", "on_i"]
    }
    HEADER = ['thresh_i', 'x_pnt_i', 'y_pnt_i', 'stat_btcl', 'stat_btcu', 'nstats']

    def __init__(self, in_params):
        """
        Initializes the class by saving input parameters and reading data from file.

        Args:
            in_params (dict): Input parameters as a dictionary.
        Raises:
            pd.errors.EmptyDataError: When the input DataFrame is empty.
            KeyError: When an expected key is missing in parameters.
        """
            

        self.logger = setup_logging(in_params)
        logger = self.logger
        safe.logger(logger, "debug", "Initializing AggEclv with parameters.")
    
        self.statistic = None
        self.current_thresh = None
        self.params = in_params
        self.steps = np.arange(self.params['cl_step'], 1, self.params['cl_step'])
        self.column_names = np.array(self.LINE_TYPE_COLUMNS[self.params['line_type']])
        self.add_base_rate = self.params.get('add_base_rate', 0)

        if self.add_base_rate not in [0, 1]:
            self.add_base_rate = 0
            safe.logger(logger, "warning", f"add_base_rate parameter was invalid. Reset to 0. Received value: {self.params.get('add_base_rate')}")

        safe.logger(logger, "debug", f"Parameters set: Steps: {self.steps}, Column Names: {self.column_names}, Add Base Rate: {self.add_base_rate}")
    
        try:
            self.input_data = pd.read_csv(self.params['agg_stat_input'], header=0, sep='\t')
            safe.logger(logger, "info", f"Successfully loaded data from {self.params['agg_stat_input']}")
        except pd.errors.EmptyDataError as e:
            safe.logger(logger, "error", "Input data file is empty, raising EmptyDataError.", exc_info=True)
            raise
        except KeyError as e:
            safe.logger(logger, "error", f"Parameter with key {str(e)} is missing, raising KeyError.", exc_info=True)
            raise
        except Exception as e:
            safe.logger(logger, "error", f"Unexpected error occurred during data loading: {str(e)}", exc_info=True)
            raise

        self.group_to_value = {}
        safe.logger(logger, "debug", "AggEclv initialized successfully.")

    def _calc_stats(self, values):
        """Calculate the statistic of values for each bootstrap sample
            Args:
                values: a np.array of values we want to calculate the statistic on
                    This is actually a 2d array (matrix) of values. Each row represents
                    a bootstrap resample simulation that we wish to aggregate across.
             Returns:
                a list of calculated statistics
            Raises:
                an error

        """
        logger = self.logger
        safe.logger(logger, "debug", "Starting to calculate statistics for given values.")
        if values is None:
           safe.logger(logger, "error", "Received None as input for values which is not expected.")
           raise ValueError, "Input values cannot be None.")

        if values.ndim == 2:
            # The single value case
            safe.logger(logger, "debug", "Processing single value case for statistical calculation.")
            try:
                stat_values = [
                    calculate_eclv(values, self.column_names, self.current_thresh, self.params['line_type'], self.steps,
                                   self.add_base_rate, logger=logger)
                ]
                safe.logger(logger, "info", "Statistics calculated successfully for single value case.")
            except Exception as e:
                safe.logger(logger, "error", f"Failed to calculate statistics for single value case: {str(e)}", exc_info=True)
                raise

        elif values.ndim == 3:
            # Bootstrapped case
            safe.logger(logger, "debug", "Processing bootstrapped case for statistical calculation.")
            stat_values = []
            try:
                for row in values:
                    stat_value = [
                        calculate_eclv(row, self.column_names, self.current_thresh, self.params['line_type'], self.steps, logger=logger)
                    ]
                    stat_values.append(stat_value)
                safe.logger(logger, "info", "Statistics calculated successfully for all bootstrap samples.")
            except Exception as e:
                safe.logger(logger, "error", f"Failed to calculate statistics for bootstrapped case: {str(e)}", exc_info=True)
                raise
        else:
            safe.logger(logger, "error", f"Invalid dimension {values.ndim} for values, expected 2 or 3.")
            raise KeyError, f"Invalid data dimensions {values.ndim}; expected 2D or 3D array.")

        return stat_values

    def _get_bootstrapped_stats(self, series_data, thresholds):
        """ Calculates aggregation statistic value and CI intervals if needed for input data
            Args:
                series_data: pandas data frame
            Returns:
                BootstrapDistributionResults object

        """
        logger = self.logger
        safe.logger(logger, "debug", "Starting the calculation of bootstrapped statistics.")

        # if the data frame is empty - do nothing and return an empty object
        if series_data.empty:
            safe.logger(logger, "warning", "Received an empty DataFrame, returning empty results.")
            return BootstrapResults(lower_bound=None, value=None, upper_bound=None)
 
        data = series_data[self.column_names].to_numpy()
        boot_stat_thresh = {}
        for ind, thresh in enumerate(thresholds):
            self.current_thresh = thresh
            safe.logger(logger, "debug", f"Processing threshold {thresh}.")
            if self.params['num_iterations'] == 1:
                safe.logger(logger, "info", "Single iteration mode: no bootstrapping required.")
                stat_val = self._calc_stats(data)[0]
                results = BootstrapResults(lower_bound=None, value=stat_val, upper_bound=None)
                safe.logger(logger, "debug", f"Statistics calculated for threshold {thresh} without bootstrapping.")
            else:
                try:
                    block_length = 1
                    if 'circular_block_bootstrap' in self.params:
                        is_cbb = parse_bool(self.params['circular_block_bootstrap'])
                        if is_cbb:
                            block_length = int(math.sqrt(len(data)))
                            safe.logger(logger, "debug", f"Using circular block bootstrap with block length {block_length}.")

                    results = bootstrap_and_value(
                        data,
                        stat_func=self._calc_stats,
                        num_iterations=self.params['num_iterations'],
                        num_threads=self.params['num_threads'],
                        ci_method=self.params['method'],
                        save_data=False,
                        block_length=block_length,
                        eclv=True
                        logger=logger
                        )
                    safe.logger(logger, "info", f"Bootstrapped statistics calculated for threshold {thresh}.")
                except KeyError as err:
                    safe.logger(logger, "error", f"Failed to calculate bootstrapped statistics due to missing key: {err}", exc_info=True)
                    results = BootstrapResults(None, None, None)

            boot_stat_thresh[ind] = results

        return boot_stat_thresh

    def _init_out_frame(self, fields, row_number):
        """ Initialises the output frame and add series values to each row
            Args:
                series_fields: list of all possible series fields
                series: list of all series definitions
            Returns:
                pandas data frame
        """
        logger = self.logger
        safe.logger(logger, "debug", f"Initializing output frame with fields: {fields} and {row_number} rows.") 
        result = pd.DataFrame(index=range(row_number))
        # fill series variables and values
        for field in fields:
            if field == 'nstats':
                result[field] = 0  # Initialize 'nstats' with 0s
                safe.logger(logger, "debug", f"Field '{field}' initialized with zeros across {row_number} rows.")
            else:
                result[field] = None  # Initialize other fields with None
                safe.logger(logger, "debug", f"Field '{field}' initialized with None across {row_number} rows.")

        safe.logger(logger, "info", f"Output DataFrame initialized successfully with fields: {fields}.")
        return result

    def _proceed_with_axis(self):
        """Calculates stat values for the requested Y axis

             Returns:
                pandas dataframe  with calculated stat values and CI

        """
        logger = self.logger
        safe.logger(logger, "debug", "Starting calculation of stat values for the requested Y axis.")
    
        if self.input_data.empty:
            safe.logger(logger, "warning", "Input data frame is empty. Exiting calculation.")
            return pd.DataFrame()
        series_val = self.params['series_val_1']
        if len(series_val) > 0:
            current_header = list(series_val.keys())
            current_header.extend(self.HEADER)
            safe.logger(logger, "debug", f"Headers set with series values: {current_header}")
        else:
            current_header = self.HEADER.copy()
            safe.logger(logger, "debug", "No series values provided; using default headers.")

        all_points = list(itertools.product(*series_val.values()))
        safe.logger(logger, "info", f"Generated all combinations for points to be processed: {len(all_points)} combinations.")

        out_frame = self._init_out_frame(current_header, 0)
        safe.logger(logger, "debug", "Initialized output DataFrame for storing results.")
        # for each point
        for point in all_points:
            out_frame_local = self._init_out_frame(current_header, len(self.steps) + self.add_base_rate)
            safe.logger(logger, "debug", f"Processing point: {point}")
             # filter point data
            all_filters = []
            for field_ind, field in enumerate(series_val.keys()):
                filter_value = point[field_ind]
                filter_list = [filter_value]
                out_frame_local[field] = [filter_value] * (len(self.steps))
                for i, filter_val in enumerate(filter_list):
                    if is_string_integer(filter_val):
                        filter_list[i] = int(filter_val)
                    elif is_string_strictly_float(filter_val):
                        filter_list[i] = float(filter_val)
                if field in self.input_data.keys():
                    all_filters.append((self.input_data[field].isin(filter_list)))

            mask = np.array(all_filters).all(axis=0)
            point_data = self.input_data.loc[mask]

            if 'thresh_i' in point_data.columns:
                thresholds = sorted(point_data['thresh_i'].unique().tolist())
            else:
                thresholds = [0]

            bootstrap_results = self._get_bootstrapped_stats(point_data, thresholds)
            safe.logger(logger, "debug", f"Bootstrap results obtained for point {point}")

            for thresh_ind, thresh in enumerate(thresholds):
                out_frame_local['thresh_i'] = [thresh] * (len(self.steps) + self.add_base_rate)
                out_frame_local['x_pnt_i'] = bootstrap_results[thresh_ind].value['cl']
                out_frame_local['y_pnt_i'] = bootstrap_results[thresh_ind].value['V']
                out_frame_local['nstats'] = [len(point_data)] * (len(self.steps) + self.add_base_rate)
                if self.params['num_iterations'] > 1:
                    out_frame_local['stat_btcu'] = bootstrap_results[thresh_ind].upper_bound
                    out_frame_local['stat_btcl'] = bootstrap_results[thresh_ind].lower_bound
                frames = [out_frame, out_frame_local]
                out_frame = pd.concat(frames)

            safe.logger(logger, "info", f"Completed processing for point {point}")

        out_frame.reset_index(drop=True, inplace=True)
        safe.logger(logger, "info", "All data processed successfully. Returning compiled DataFrame.")
        return out_frame

    def calculate_stats_and_ci(self):
        """ Calculates aggregated statistics and confidants intervals
            ( if parameter num_iterations > 1) for each series point
            Writes output data to the file

        """
        logger = self.logger
        safe.logger(logger, "debug", "Starting calculation of statistics and confidence intervals.")
    
        # set random seed if present
        if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
            np.random.seed(self.params['random_seed'])
            safe.logger(logger, "info", f"Random seed set to {self.params['random_seed']}.")

        # perform EE if needed
        if parse_bool(self.params.get('event_equal', False)):
            safe.logger(logger, "info", "Event equalization enabled.")
            fix_vals_permuted_list = []

            for key in self.params['fixed_vars_vals_input']:
                vals_permuted = list(itertools.product(*self.params['fixed_vars_vals_input'][key].values()))
                fix_vals_permuted_list.extend(vals_permuted)

            fix_vals_keys = list(self.params['fixed_vars_vals_input'].keys())
            is_equalize_by_indep = parse_bool(self.params.get('equalize_by_indep', False))
            self.input_data = event_equalize(self.input_data, 'stat_name',
                                             self.params['series_val_1'],
                                             fix_vals_keys,
                                             fix_vals_permuted_list, is_equalize_by_indep, False, logger=logger)
            safe.logger(logger, "debug", "Event equalization completed.")

        # Process data to calculate statistics
        out_frame = self._proceed_with_axis()
        safe.logger(logger, "info", "Statistics and confidence intervals calculation completed.")

        # Determine file writing mode based on configuration
        header = True
        mode = 'w'
        if parse_bool(self.params.get('append_to_file', False)):
            header = False
            mode = 'a'
            safe.logger(logger, "debug", "Appending to existing file.")

        # Write output data to file
        try:
            export_csv = out_frame.to_csv(self.params['agg_stat_output'], index=None, header=header, mode=mode,
                                          sep="\t", na_rep="NA", float_format='%.' + str(PRECISION) + 'f')
            safe.logger(logger, "info", f"Data successfully written to {self.params['agg_stat_output']} in mode {mode}.")
        except Exception as e:
            safe.logger(logger, "error", f"Failed to write data to file: {str(e)}", exc_info=True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT = AggEclv(PARAMS)
    AGG_STAT.calculate_stats_and_ci()
