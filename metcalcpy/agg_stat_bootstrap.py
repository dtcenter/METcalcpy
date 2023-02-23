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
        result = pd.DataFrame()
        row_number = len(series)
        # fill series variables and values
        for field_ind, field in enumerate(series_fields):
            result[field] = [row[field_ind] for row in series]

        # fill the stats  and CI values placeholders with None
        result['fcst_var'] = [None] * row_number
        result['stat_value'] = [None] * row_number
        result['stat_btcl'] = [None] * row_number
        result['stat_btcu'] = [None] * row_number
        result['nstats'] = [None] * row_number
        return result

    def _proceed_with_axis(self, axis="1"):
        if not self.input_data.empty:
            # identify all possible points values by adding series values, indy values
            # and statistics and then permute them
            indy_vals = self.params['indy_vals']
            series_val = self.params['series_val_' + axis]
            all_fields_values = series_val.copy()
            all_fields_values[self.params['indy_var']] = indy_vals
            all_fields_values['stat_name'] = self.params['list_stat_' + axis]
            all_points = list(itertools.product(*all_fields_values.values()))
            fcst_var = None
            if len(self.params['fcst_var_val_' + axis]) > 0 and 'fcst_var' in self.input_data.columns:
                fcst_var = list(self.params['fcst_var_val_' + axis].keys())[0]

            cases = []
            out_frame = self._init_out_frame(all_fields_values.keys(), all_points)
            point_to_distrib = {}

            # run the bootstrap flow for each independent variable value
            for indy_val in indy_vals:
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

                    # build a list of cases to sample
                    fcst_valid = point_data.loc[:, 'fcst_valid'].astype(str)
                    indy_var = point_data.loc[:, self.params['indy_var']].astype(str)
                    case_cur = np.unique(fcst_valid + '#' + indy_var)
                    cases = np.sort(np.unique(np.append(cases, case_cur)))
                    cases = np.reshape(cases, (cases.shape[0], 1))
                # calculate bootstrap for cases
                for stat_upper in self.params['list_stat_' + axis]:
                    self.statistic = stat_upper.lower()
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
                        # save bootstrap results
                        point_to_distrib[point] = bootstrap_results
                        n_stats = len(point_data)

                        # find an index of this point in the output data frame
                        rows_with_mask = out_frame.loc[mask_out_frame]
                        rows_with_mask_indy_var = \
                            rows_with_mask[(rows_with_mask[self.params['indy_var']] == indy_val)]
                        index = rows_with_mask_indy_var.index[0]

                        # save results to the output data frame
                        out_frame['fcst_var'][index] = fcst_var
                        out_frame['stat_value'][index] = bootstrap_results.value
                        out_frame['stat_btcl'][index] = bootstrap_results.lower_bound
                        out_frame['stat_btcu'][index] = bootstrap_results.upper_bound
                        out_frame['nstats'][index] = n_stats
        else:
            out_frame = pd.DataFrame()
        return out_frame

    def _get_bootstrapped_stats(self, series_data, cases):
        self.series_data = sort_data(series_data)
        if self.params['num_iterations'] == 1:
            stat_val = self._calc_stats(cases)[0]
            results = BootstrapResults(lower_bound=None,
                                                   value=stat_val,
                                                   upper_bound=None)
        else:
            # need bootstrapping and CI calculation in addition to statistic
            try:
                results = bootstrap_and_value_mode(
                    self.series_data,
                    cases,
                    stat_func=self._calc_stats,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'])

            except KeyError as err:
                results = BootstrapResults(None, None, None)
                print(err)
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
        func_name = f'calculate_{self.statistic}'
        if cases is not None and cases.ndim == 2:
            # The single value case

            # build a data frame with the sampled data
            data_cases = np.asarray(self.series_data['case'])
            flat_cases = cases.flatten()
            values = self.series_data[np.in1d(data_cases, flat_cases)].to_numpy()
            stat_values = [globals()[func_name](values, self.column_names)]
        elif cases is not None and cases.ndim == 3:
            # bootstrapped case
            stat_values = []
            for row in cases:
                values_ind = self.series_data['case'].isin(row.flatten())
                values = self.series_data[values_ind]
                stat_values.append([globals()[func_name](values, self.column_names)])
        else:
            raise KeyError("can't calculate statistic")
        return stat_values

    def calculate_values(self):
        """ Performs EE if needed followed by  aggregation statistic logic
            Writes output data to the file
        """
        if not self.input_data.empty:
            if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
                np.random.seed(self.params['random_seed'])

            # perform EE if needed
            is_event_equal = parse_bool(self.params['event_equal'])
            if is_event_equal:
                self._perform_event_equalization()

            # build the case information for each record
            fcst_valid = self.input_data.loc[:, 'fcst_valid'].astype(str)
            indy_var = self.input_data.loc[:, self.params['indy_var']].astype(str)
            self.input_data['case'] = fcst_valid + '#' + indy_var

            # get results for axis1
            out_frame = self._proceed_with_axis("1")
            if self.params['series_val_2']:
                out_frame = pd.concat([out_frame, self._proceed_with_axis("2")])

        else:
            out_frame = pd.DataFrame()

        header = True
        mode = 'w'

        export_csv = out_frame.to_csv(self.params['agg_stat_output'],
                                      index=None, header=header, mode=mode,
                                      sep="\t", na_rep="NA")

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
