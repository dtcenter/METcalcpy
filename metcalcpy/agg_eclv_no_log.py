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

from metcalcpy.bootstrap import bootstrap_and_value, BootstrapResults
from metcalcpy.event_equalize import event_equalize
from metcalcpy.util.utils import PRECISION, is_string_strictly_float
from metcalcpy.util.eclv_statistics import *

from metcalcpy.util.utils import is_string_integer, parse_bool

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
        """Initialises the class by saving input parameters and reading data from file

            Args:
                in_params - input parameters as a dictionary
            Raises: EmptyDataError or ValueError when the input DataFrame is empty
                or doesn't have data
        """

        self.statistic = None
        self.current_thresh = None
        self.params = in_params
        self.steps = np.arange(self.params['cl_step'], 1, self.params['cl_step'])
        self.column_names = np.array(self.LINE_TYPE_COLUMNS[self.params['line_type']])

        if 'add_base_rate' in self.params.keys():
            self.add_base_rate = self.params['add_base_rate']
        else:
            self.add_base_rate = 0
        if self.add_base_rate != 0 or self.add_base_rate != 1:
            self.add_base_rate = 0

        try:
            self.input_data = pd.read_csv(
                self.params['agg_stat_input'],
                header=[0],
                sep='\t'
            )
        except pd.errors.EmptyDataError:
            raise
        except KeyError as er:
            print(f'ERROR: parameter with key {er} is missing')
            raise
        self.group_to_value = {}

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

        if values is not None and values.ndim == 2:
            # The single value case
            stat_values = [
                calculate_eclv(values, self.column_names, self.current_thresh, self.params['line_type'], self.steps,
                               self.add_base_rate)]

        elif values is not None and values.ndim == 3:
            # bootstrapped case
            stat_values = []
            for row in values:
                stat_value = [
                    calculate_eclv(row, self.column_names, self.current_thresh, self.params['line_type'], self.steps)]
                stat_values.append(stat_value)

        else:
            raise KeyError("can't calculate statistic")
        return stat_values

    def _get_bootstrapped_stats(self, series_data, thresholds):
        """ Calculates aggregation statistic value and CI intervals if needed for input data
            Args:
                series_data: pandas data frame
            Returns:
                BootstrapDistributionResults object

        """

        # if the data frame is empty - do nothing and return an empty object
        if series_data.empty:
            return BootstrapResults(lower_bound=None,
                                    value=None,
                                    upper_bound=None)

        data = series_data[self.column_names].to_numpy()
        boot_stat_thresh = {}
        for ind, thresh in enumerate(thresholds):
            self.current_thresh = thresh
            if self.params['num_iterations'] == 1:
                # don't need bootstrapping and CI calculation -
                # calculate the statistic and exit
                stat_val = self._calc_stats(data)[0]

                results = BootstrapResults(lower_bound=None,
                                           value=stat_val,
                                           upper_bound=None)

            else:
                # need bootstrapping and CI calculation in addition to statistic
                try:
                    block_length = 1
                    # to use circular block bootstrap or not
                    is_cbb = True
                    if 'circular_block_bootstrap' in self.params.keys():
                        is_cbb = parse_bool(self.params['circular_block_bootstrap'])

                    if is_cbb:
                        block_length = int(math.sqrt(len(data)))
                    results = bootstrap_and_value(
                        data,
                        stat_func=self._calc_stats,
                        num_iterations=self.params['num_iterations'],
                        num_threads=self.params['num_threads'],
                        ci_method=self.params['method'],
                        save_data=False,
                        block_length=block_length,
                        eclv=True
                    )

                except KeyError as err:
                    results = BootstrapResults(None, None, None)
                    print(err)
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
        result = pd.DataFrame()
        # fill series variables and values
        for field in fields:
            if field == 'nstats':
                result[field] = [0] * row_number
            else:
                result[field] = [None] * row_number
        return result

    def _proceed_with_axis(self):
        """Calculates stat values for the requested Y axis

             Returns:
                pandas dataframe  with calculated stat values and CI

        """
        if self.input_data.empty:
            return pd.DataFrame()

        series_val = self.params['series_val_1']
        if len(series_val) > 0:
            current_header = list(series_val.keys())
            current_header.extend(self.HEADER)
        else:
            current_header = self.HEADER.copy()

        all_points = list(itertools.product(*series_val.values()))

        out_frame = self._init_out_frame(current_header, 0)

        # for each point
        for point in all_points:
            out_frame_local = self._init_out_frame(current_header, len(self.steps) + self.add_base_rate)
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

            # use numpy to select the rows where any record evaluates to True
            mask = np.array(all_filters).all(axis=0)
            point_data = self.input_data.loc[mask]

            # calculate bootstrap results
            if 'thresh_i' in point_data.columns:
                thresholds = point_data['thresh_i'].unique().tolist()
                thresholds.sort()
            else:
                thresholds = [0]

            bootstrap_results = self._get_bootstrapped_stats(point_data, thresholds)

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

        out_frame.reset_index(inplace=True, drop=True)
        return out_frame

    def calculate_stats_and_ci(self):
        """ Calculates aggregated statistics and confidants intervals
            ( if parameter num_iterations > 1) for each series point
            Writes output data to the file

        """

        # set random seed if present
        if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
            np.random.seed(self.params['random_seed'])

        is_event_equal = parse_bool(self.params['event_equal'])
        # perform EE if needed
        if is_event_equal:
            fix_vals_permuted_list = []

            for key in self.params['fixed_vars_vals_input']:
                vals_permuted = list(itertools.product(*self.params['fixed_vars_vals_input'][key].values()))
                vals_permuted_list = [item for sublist in vals_permuted for item in sublist]
                fix_vals_permuted_list.append(vals_permuted_list)

            fix_vals_keys = list(self.params['fixed_vars_vals_input'].keys())
            is_equalize_by_indep = parse_bool(self.params['equalize_by_indep'])
            self.input_data = event_equalize(self.input_data, 'stat_name',
                                             self.params['series_val_1'],
                                             fix_vals_keys,
                                             fix_vals_permuted_list, is_equalize_by_indep, False)

        out_frame = self._proceed_with_axis()
        header = True
        mode = 'w'

        if 'append_to_file' in self.params.keys() and self.params['append_to_file'] == 'True':
            header = False
            mode = 'a'

        export_csv = out_frame.to_csv(self.params['agg_stat_output'],
                                      index=None, header=header, mode=mode,
                                      sep="\t", na_rep="NA", float_format='%.' + str(PRECISION) + 'f')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT = AggEclv(PARAMS)
    AGG_STAT.calculate_stats_and_ci()
