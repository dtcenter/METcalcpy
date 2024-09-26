# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: agg_stat_bootstrap.py

How to use:
 - Call from other Python function
        AAGG_STAT_BOOTSTRAP = AggStatBootstrap(PARAMS)
        AGG_STAT_BOOTSTRAP.calculate_values()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python agg_stat_bootstrap.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_stat_bootstrap.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import argparse
import itertools
import sys

import yaml
import pandas as pd
import numpy as np

from metcalcpy import GROUP_SEPARATOR
from metcalcpy.agg_stat_eqz import AggStatEventEqz
from metcalcpy.bootstrap import  bootstrap_and_value_mode, BootstrapResults
from metcalcpy.util.mode_ratio_statistics import *
from metcalcpy.util.mode_arearat_statistics import *
from metcalcpy.util.mode_2d_arearat_statistics import *
from metcalcpy.util.mode_2d_ratio_statistics import *
from metcalcpy.util.mode_3d_volrat_statistics import *
from metcalcpy.util.mode_3d_ratio_statistics import *
from metcalcpy.util.utils import is_string_integer, parse_bool, sort_data, is_string_strictly_float
from metcalcpy.logging_config import setup_logging
from metcalcpy.util.safe_log import safe_log

class AggStatBootstrap:
    """A class that performs aggregation statistic logic fot MODE and MTD ratio statistics on input data frame.
        All parameters including data description and location is in the parameters dictionary
        Usage:
            initialise this call with the parameters dictionary and than
            call calculate_value_and_ci method
            This method will crate and save to the file aggregation statistics
                agg_stat_boot = AggStatBootstrap(params)
                agg_stat_boot.calculate_value_and_ci()
           """

    def __init__(self, in_params):
        """Initialises the class by saving input parameters and reading data from file

            Args:
                in_params - input parameters as a dictionary
        """
        self.logger = setup_logging(in_params)
        logger = self.logger
        safe.logger(logger, "debug", "Initializing AggStatBootstrap with parameters.")
        self.statistic = None
        self.derived_name_to_values = {}
        self.params = in_params

        self.input_data = pd.read_csv(
            self.params['agg_stat_input'],
            header=[0],
            sep='\t'
        )
        self.column_names = self.input_data.columns.values
        self.group_to_value = {}
        self.series_data = None

    def _init_out_frame(self, series_fields, series):
        """ Initialises the output frame and add series values to each row
            Args:
                series_fields: list of all possible series fields
                series: list of all series definitions
            Returns:
                pandas data frame
        """
        logger = self.logger
        safe.logger(logger, "debug", "Initializing output data frame.")
        result = pd.DataFrame()
        row_number = len(series)
        safe.logger(logger, "debug", f"Number of rows to initialize: {row_number}")
        # fill series variables and values
        for field_ind, field in enumerate(series_fields):
            result[field] = [row[field_ind] for row in series]
            safe.logger(logger, "debug", f"Field '{field}' initialized with {len(result[field])} entries.")
        # fill the stats  and CI values placeholders with None
        result['fcst_var'] = [None] * row_number
        result['stat_value'] = [None] * row_number
        result['stat_btcl'] = [None] * row_number
        result['stat_btcu'] = [None] * row_number
        result['nstats'] = [None] * row_number

        safe.logger(logger, "debug", "Stats and confidence interval placeholders added.")
        safe.logger(logger, "debug", f"DataFrame initialized with columns: {result.columns.tolist()}")

        return result

    def _proceed_with_axis(self, axis="1"):

        logger = self.logger
        safe.logger(logger, "info", f"Proceeding with axis: {axis}")
        if not self.input_data.empty:
            # identify all possible points values by adding series values, indy values
            # and statistics and then permute them
            safe.logger(logger, "debug", "Input data is not empty. Proceeding with calculations.")
            indy_vals = self.params['indy_vals']
            series_val = self.params['series_val_' + axis]
            all_fields_values = series_val.copy()
            all_fields_values[self.params['indy_var']] = indy_vals
            all_fields_values['stat_name'] = self.params['list_stat_' + axis]
            all_points = list(itertools.product(*all_fields_values.values()))
            safe.logger(logger, "debug", f"All points generated: {len(all_points)} points created for axis {axis}.")
            fcst_var = None
            if len(self.params['fcst_var_val_' + axis]) > 0 and 'fcst_var' in self.input_data.columns:
                fcst_var = list(self.params['fcst_var_val_' + axis].keys())[0]
                safe.logger(logger, "debug", f"Forecast variable identified: {fcst_var}")
            cases = []
            out_frame = self._init_out_frame(all_fields_values.keys(), all_points)
            safe.logger(logger, "debug", f"Output DataFrame initialized with {len(out_frame)} rows.")
            point_to_distrib = {}

            # run the bootstrap flow for each independent variable value
            for indy_val in indy_vals:
                safe.logger(logger, "debug", f"Processing independent value: {indy_val}")
                # extract the records for the current indy value
                if is_string_integer(indy_val):
                    filtered_by_indy_data = \
                        self.input_data[self.input_data[self.params['indy_var']] == int(indy_val)]
                elif is_string_strictly_float(indy_val):
                    filtered_by_indy_data = \
                        self.input_data[self.input_data[self.params['indy_var']] == float(indy_val)]
                else:
                    filtered_by_indy_data = \
                        self.input_data[self.input_data[self.params['indy_var']] == indy_val]
                # and statistics and then permute them
                all_fields_values = series_val.copy()

                all_points = list(itertools.product(*all_fields_values.values()))
                safe.logger(logger, "debug", f"Number of points for independent value '{indy_val}': {len(all_points)}.")

                for point in all_points:
                    all_filters = []
                    for field_ind, field in enumerate(all_fields_values.keys()):
                        filter_value = point[field_ind]
                        if GROUP_SEPARATOR in filter_value:
                            filter_list = filter_value.split(GROUP_SEPARATOR)
                        elif ';' in filter_value:
                            filter_list = filter_value.split(';')
                        else:
                            filter_list = [filter_value]
                        for i, filter_val in enumerate(filter_list):
                            if is_string_integer(filter_val):
                                filter_list[i] = int(filter_val)
                            elif is_string_strictly_float(filter_val):
                                filter_list[i] = float(filter_val)

                        all_filters.append((filtered_by_indy_data[field].isin(filter_list)))

                    # add fcst var
                    if fcst_var is not None:
                        all_filters.append((filtered_by_indy_data['fcst_var'].isin([fcst_var])))

                    # use numpy to select the rows where any record evaluates to True
                    mask = np.array(all_filters).all(axis=0)
                    point_data = filtered_by_indy_data.loc[mask]
                    safe.logger(logger, "debug", f"Point data filtered for point {point}. Number of records: {len(point_data)}")

                    # build a list of cases to sample
                    fcst_valid = point_data.loc[:, 'fcst_valid'].astype(str)
                    indy_var = point_data.loc[:, self.params['indy_var']].astype(str)
                    case_cur = np.unique(fcst_valid + '#' + indy_var)
                    cases = np.sort(np.unique(np.append(cases, case_cur)))
                    cases = np.reshape(cases, (cases.shape[0], 1))
                # calculate bootstrap for cases
                for stat_upper in self.params['list_stat_' + axis]:
                    self.statistic = stat_upper.lower()
                    safe.logger(logger, "debug", f"Calculating bootstrap for statistic: {self.statistic}")
                    for point in all_points:
                        all_filters = []
                        out_frame_filter = []
                        for field_ind, field in enumerate(all_fields_values.keys()):
                            filter_value = point[field_ind]
                            if ':' in filter_value:
                                filter_list = filter_value.split(':')
                            elif ';' in filter_value:
                                filter_list = filter_value.split(';')
                            else:
                                filter_list = [filter_value]
                            for i, filter_val in enumerate(filter_list):
                                if is_string_integer(filter_val):
                                    filter_list[i] = int(filter_val)
                                elif is_string_strictly_float(filter_val):
                                    filter_list[i] = float(filter_val)

                            all_filters.append((filtered_by_indy_data[field].isin(filter_list)))
                            out_frame_filter.append((out_frame[field].isin(filter_list)))
                        # use numpy to select the rows where any record evaluates to True
                        mask = np.array(all_filters).all(axis=0)
                        mask_out_frame = np.array(out_frame_filter).all(axis=0)
                        point_data = filtered_by_indy_data.loc[mask]
                        bootstrap_results = self._get_bootstrapped_stats(point_data, cases)
                        safe.logger(logger, "debug", f"Bootstrap results calculated for point {point}: {bootstrap_results.value}")
                        # save bootstrap results
                        point_to_distrib[point] = bootstrap_results
                        n_stats = len(point_data)

                        # find an index of this point in the output data frame
                        rows_with_mask = out_frame.loc[mask_out_frame]
                        rows_with_mask_indy_var = \
                            rows_with_mask[(rows_with_mask[self.params['indy_var']] == indy_val)]
                        index = rows_with_mask_indy_var.index[0]

                        # save results to the output data frame
                        out_frame.loc[index, 'fcst_var'] = fcst_var
                        out_frame.loc[index, 'stat_value'] = bootstrap_results.value
                        out_frame.loc[index, 'stat_btcl'] = bootstrap_results.lower_bound
                        out_frame.loc[index, 'stat_btcu'] = bootstrap_results.upper_bound
                        out_frame.loc[index, 'nstats'] = n_stats
                        safe.logger(logger, "debug", f"Results saved to output DataFrame at index {index} for point {point}.")
        else:
            out_frame = pd.DataFrame()
            safe.logger(logger, "warning", "Input data is empty. Returning an empty DataFrame.")

        safe.logger(logger, "info", f"Completed processing for axis: {axis}")
        return out_frame

    def _get_bootstrapped_stats(self, series_data, cases):
        logger = self.logger
        safe.logger(logger, "info", "Starting bootstrapping process.")

        safe.logger(logger, "debug", "Sorting series data.")
        self.series_data = sort_data(series_data)
        safe.logger(logger, "debug", f"Data sorted. Number of rows: {len(self.series_data)}")
        if self.params['num_iterations'] == 1:
            safe.logger(logger, "info", "Only one iteration specified. Skipping bootstrapping.")
            stat_val = self._calc_stats(cases)[0]
            safe.logger(logger, "debug", f"Statistic calculated: {stat_val}")
            results = BootstrapResults(lower_bound=None,
                                                   value=stat_val,
                                                   upper_bound=None)
            safe.logger(logger, "info", "Statistic calculated without bootstrapping.")
        else:
            # need bootstrapping and CI calculation in addition to 
            safe.logger(logger, "info", "Performing bootstrapping and confidence interval calculation.")
            try:
                results = bootstrap_and_value_mode(
                    self.series_data,
                    cases,
                    stat_func=self._calc_stats,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    
                    )
                safe.logger(logger, "debug", "Bootstrapping completed successfully.")
            except KeyError as err:
                safe.logger(logger, "error", f"Error during bootstrapping: {err}", exc_info=True)
                results = BootstrapResults(None, None, None)
                safe.logger(logger, "info", "Returning empty BootstrapResults due to error.")
                print(err)
        safe.logger(logger, "info", "Bootstrapping process completed.")
        return results

    def _calc_stats(self, cases):
        """Calculate the statistic of values for each bootstrap sample
            Args:
                cases: a np.array of values we want to calculate the statistic on
                    This is actually a 2d array (matrix) of values. Each row represents
                    a bootstrap resample simulation that we wish to aggregate across.
             Returns:
                a list of calculated statistics
            Raises:
                an error

        """
        logger = self.logger
        func_name = f'calculate_{self.statistic}'
        safe.logger(logger, "info", f"Starting statistic calculation using function: {func_name}")
        if cases is not None and cases.ndim == 2:
            # The single value case
            safe.logger(logger, "debug", "Processing single-value case.")

            # build a data frame with the sampled data
            data_cases = np.asarray(self.series_data['case'])
            flat_cases = cases.flatten()
            values = self.series_data[np.in1d(data_cases, flat_cases)].to_numpy()
            safe.logger(logger, "debug", f"Number of values selected for single case: {len(values)}")
            # Calculate the statistic for each bootstrap iteration
            try:
                stat_value = globals()[func_name](values, self.column_names, logger=logger)
                stat_values.append([stat_value])
                safe.logger(logger, "info", f"Statistic calculated for bootstrap iteration: {stat_value}")
            except Exception as e:
                safe.logger(logger, "error", f"Error calculating statistic for bootstrap iteration: {e}", exc_info=True)
                raise
            
        elif cases is not None and cases.ndim == 3:
            # bootstrapped case
            stat_values = []
            for row in cases:
                values_ind = self.series_data['case'].isin(row.flatten())
                values = self.series_data[values_ind]
                safe.logger(logger, "debug", f"Number of values selected for bootstrap iteration: {len(values)}")
                # Calculate the statistic for each bootstrap iteration
                try:
                    stat_value = globals()[func_name](values, self.column_names, logger=logger)
                    stat_values.append([stat_value])
                    safe.logger(logger, "info", f"Statistic calculated for bootstrap iteration: {stat_value}")
                except Exception as e:
                    safe.logger(logger, "error", f"Error calculating statistic for bootstrap iteration: {e}", exc_info=True)
                    raise
        else:
            safe.logger(logger, "error", "Invalid input for cases. Cannot calculate statistic.")
            raise KeyError("can't calculate statistic")
        return stat_values

    def calculate_values(self):
        """ Performs EE if needed followed by  aggregation statistic logic
            Writes output data to the file
        """
        logger = self.logger
        safe.logger(logger, "info", "Starting calculation of values.")
        if not self.input_data.empty:
            safe.logger(logger, "debug", "Input data is not empty. Proceeding with calculations.")
            if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
                safe.logger(logger, "debug", f"Random seed set to: {self.params['random_seed']}")
                np.random.seed(self.params['random_seed'])

            # perform EE if needed
            is_event_equal = parse_bool(self.params['event_equal'])
            if is_event_equal:
                safe.logger(logger, "info", "Event equalization required. Performing event equalization.")
                self._perform_event_equalization()
                safe.logger(logger, "debug", "Event equalization completed.")

            # build the case information for each record
            safe.logger(logger, "debug", "Building case information for each record.")
            fcst_valid = self.input_data.loc[:, 'fcst_valid'].astype(str)
            indy_var = self.input_data.loc[:, self.params['indy_var']].astype(str)
            self.input_data['case'] = fcst_valid + '#' + indy_var
            safe.logger(logger, "debug", "Case information added to the input data.")

            # get results for axis1
            safe.logger(logger, "info", "Calculating results for axis 1.")
            out_frame = self._proceed_with_axis("1")
            if self.params['series_val_2']:
                safe.logger(logger, "info", "Series values for axis 2 detected. Calculating results for axis 2.")
                out_frame = pd.concat([out_frame, self._proceed_with_axis("2")])
                safe.logger(logger, "debug", "Results for axis 2 calculated and combined with axis 1.")

        else:
            safe.logger(logger, "warning", "Input data is empty. Returning an empty DataFrame.")
            out_frame = pd.DataFrame()

        header = True
        mode = 'w'
        safe.logger(logger, "info", f"Exporting results to {self.params['agg_stat_output']}")
        export_csv = out_frame.to_csv(self.params['agg_stat_output'],
                                      index=None, header=header, mode=mode,
                                      sep="\t", na_rep="NA")
        safe.logger(logger, "info", "Results successfully exported to CSV.")


    def _perform_event_equalization(self):
        """ Performs event equalisation on input data
        """
        agg_stat_event_eqz = AggStatEventEqz(self.params)

        self.input_data = agg_stat_event_eqz.calculate_values()
        file_name = self.params['agg_stat_input'].replace("agg_stat_bootstrap", "dataAfterEq")
        self.input_data.to_csv(file_name, index=None, header=True, mode='w', sep="\t", na_rep="NA")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat_bootstrap arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT_BOOTSTRAP = AggStatBootstrap(PARAMS)
    AGG_STAT_BOOTSTRAP.calculate_values()
