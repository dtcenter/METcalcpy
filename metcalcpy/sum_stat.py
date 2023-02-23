# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: sum_stat.py

How to use:
 - Call from other Python function
        SUM_STAT = SumStat(PARAMS)
        SUM_STAT.calculate_stats_and_ci()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python sum_stat.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_stat.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import sys
import itertools
import argparse
import time
import logging
import yaml
import pandas as pd
import warnings
from inspect import signature

from metcalcpy import GROUP_SEPARATOR
from metcalcpy.util.ctc_statistics import *
from metcalcpy.util.grad_statistics import *
from metcalcpy.util.sl1l2_statistics import *
from metcalcpy.util.val1l2_statistics import *
from metcalcpy.util.vcnt_statistics import *
from metcalcpy.util.vl1l2_statistics import *
from metcalcpy.util.pstd_statistics import *
from metcalcpy.util.nbrcnt_statistics import *
from metcalcpy.util.ecnt_statistics import *
from metcalcpy.util.nbrctc_statistics import *
from metcalcpy.util.rps_statistics import *

from metcalcpy.util.utils import is_string_integer, parse_bool, \
    aggregate_field_values, perform_event_equalization, is_string_strictly_float


class SumStat:
    """A class that performs event equalisation if needed and statistics calculation
        on each row of the input data frame.
        if one of the field values contain ':' (EAST:NMT) this class aggregate the values
        for these fields first and than calculate statistics on the aggregated values
           All parameters including data description and location is in the parameters' dictionary
        Usage:
            initialise this call with the parameters dictionary and then
            call calculate_stats method
            This method will crate and save to the file aggregation statistics
                sum_stat = SumStat(params)
                sum_stat.calculate_stats()
        Raises: EmptyDataError or ValueError when the input DataFrame is empty or doesn't have data
       """

    def __init__(self, in_params):
        """Initialises the class by saving input parameters and reading data from file
            Args:
                in_params - input parameters as a dictionary
            Raises: EmptyDataError or ValueError when the input DataFrame is empty
                or doesn't have data
        """

        self.params = in_params
        # import pandas
        try:
            self.input_data = pd.read_csv(
                self.params['sum_stat_input'],
                header=[0],
                sep='\t'
            )

            if self.input_data.empty:
                logging.warning('The input data is empty. The empty output file was created')
                export_csv = self.input_data.to_csv(self.params['sum_stat_output'],
                                                    index=None, header=True, mode="w",
                                                    sep="\t")
                raise ValueError("The input data is empty")

            self.column_names = self.input_data.columns.values
        except pd.errors.EmptyDataError:
            raise
        except KeyError as ex:
            logging.error('Parameter with key %s is missing', ex)
            raise

    STATISTIC_TO_FIELDS = {'ctc': {"total", "fy_oy", "fy_on", "fn_oy", "fn_on"},
                           'sl1l2': {"total", "fbar", "obar", "fobar", "ffbar", "oobar"},
                           'grad': {"total", "fgbar", "ogbar", "mgbar", "egbar"},
                           'sal1l2': {"total", "fbar", "obar", "fobar", "ffbar", "oobar"},
                           'vl1l2': {"total", "ufbar", "vfbar", "uobar", "vobar", "uvfobar",
                                     "uvffbar", "uvoobar", "f_speed_bar", "o_speed_bar"},
                           'val1l2': {"total", "ufabar", "vfabar", "uoabar", "voabar",
                                      "uvfoabar", "uvffabar", "uvooabar"},
                           'pct': {'total', 'oy_i', 'on_i', 'thresh_i'},
                           'nbrcnt': {'total', 'fbs', 'fss', 'afss', 'ufss', 'f_rate', 'o_rate'},
                           'ecnt': {'total', 'rmse', 'rmse_oerr', 'crps'},
                           'nbrctc': {"total", "fy_oy", "fy_on", "fn_oy", "fn_on"},
                           'rps': {"total", "rpss", "rps", "rps_comp"}
                           }

    def calculate_stats(self):
        """ Calculates summary statistics for each series point
            Writes output data to the file
        """
        fields = []
        try:
            fields = self.STATISTIC_TO_FIELDS[self.params['line_type']]

            # change names for sal1l2 so we can reuse sl1l2 statistics
            if self.params['line_type'] == 'sal1l2':
                self.input_data = self.input_data.rename(columns={'fabar': 'fbar',
                                                                  'oabar': 'obar',
                                                                  'foabar': 'fobar',
                                                                  'ffabar': 'ffbar',
                                                                  'ooabar': 'oobar'
                                                                  })
                self.column_names = self.input_data.columns.values

            # perform EE if needed
            is_event_equal = parse_bool(self.params['event_equal'])
            if is_event_equal:
                self.input_data = perform_event_equalization(self.params, self.input_data)

            if not self.input_data.empty:
                # perform aggregation on a special field - needed for scorecard
                self.aggregate_special_fields('1')
                self.aggregate_special_fields('2')

                start_time = time.time()
                # calculate statistic for each row
                self.process_rows()
                # self.input_data = self.parallelize_on_rows(self.input_data,
                # self.process_row, num_of_processes=mp.cpu_count())
                print("--- %s seconds ---" % (time.time() - start_time))
            else:
                logging.warning('Event equalisation removed all data. '
                                'The empty output file is created')

        except KeyError as ex:
            logging.error('Parameter with key %s is missing. The empty output file is created', ex)

        # remove the original fields to save the space
        for column in fields:
            if column in self.column_names:
                self.input_data = self.input_data.drop(column, axis=1)

        # remove 'equalize' to save the space
        if 'equalize' in self.input_data.columns:
            self.input_data = self.input_data.drop(labels=['equalize'], axis=1)

        # save the result to file
        self.input_data.to_csv(self.params['sum_stat_output'],
                               index=None, header=True, mode='w',
                               sep="\t", na_rep="NA")

    def aggregate_special_fields(self, axis='1'):
        """Finds the data for special fields - the field with ';' (EAST;NMT) -
        and the requested axis in the initial dataframe
        and aggregate it based on the line type rules.
        If there are no special fields that require aggregation - does nothing
        The result is stored in the initial dataframe

        Args:
            axis - y1 or y1 axis
        """
        warnings.filterwarnings('error')

        # check if indy_vals have a field that need to be aggregated - the field with ';'
        has_agg_indy_field = any(any(GROUP_SEPARATOR in i for i in item) for item in self.params['indy_vals'])

        # look if there is a field that need to be aggregated first - the field with ';'
        series_var_val = self.params['series_val_' + axis]
        has_agg_series_field = any(any(GROUP_SEPARATOR in i for i in item) for item in series_var_val)

        if series_var_val and (has_agg_indy_field or has_agg_series_field):
            # the special field was detected

            all_points = list(itertools.product(*series_var_val.values()))
            aggregated_values = pd.DataFrame()
            series_vars = list(series_var_val.keys())
            for indy_val in self.params['indy_vals']:
                # filter the input frame by each indy value
                if indy_val is None:
                    filtered_by_indy = self.input_data
                else:
                    # filter by value or split the value and filter by multiple values
                    filtered_by_indy = self.input_data[
                        self.input_data[self.params['indy_var']].isin(indy_val.split(';'))]

                for point in all_points:
                    point_data = filtered_by_indy

                    for index, series_var in enumerate(series_vars):
                        # get actual series value(s) and filter by them
                        if ';' in point[index]:
                            actual_series_vals = point[index].split(';')
                        else:
                            actual_series_vals = point[index].split(GROUP_SEPARATOR)
                        for ind, val in enumerate(actual_series_vals):
                            if is_string_integer(val):
                                actual_series_vals[ind] = int(val)
                            elif is_string_strictly_float(val):
                                actual_series_vals[ind] = float(val)
                        point_data = \
                            point_data[point_data[series_vars[index]].isin(actual_series_vals)]

                    # aggregate point data
                    if any(';' in series_val for series_val in point):
                        point_data = aggregate_field_values(series_var_val,
                                                            point_data,
                                                            self.params['line_type'])
                    elif ';' in indy_val:
                        # if aggregated value in indy val - add it to series values add aggregate
                        series_indy_var_val = series_var_val
                        series_indy_var_val[self.params['indy_var']] = [indy_val]
                        point_data = aggregate_field_values(series_indy_var_val,
                                                            point_data,
                                                            self.params['line_type'])

                    aggregated_values = pd.concat([aggregated_values, point_data])
            self.input_data = aggregated_values
            self.input_data.reset_index(inplace=True, drop=True)

    def process_rows(self):
        """For each row in the data frame finds the row statistic name,
            calculates it's value  and stores this value in the corresponding column
        """

        for index, row in self.input_data.iterrows():
            # statistic name
            stat = row['stat_name'].lower()

            # array of row's data
            row_array = np.expand_dims(row.to_numpy(), axis=0)

            # calculate the stat value
            stat_value = [calculate_statistic(row_array, self.column_names, stat)][0]

            # save the value to the 'stat_value' column
            self.input_data.at[index, 'stat_value'] = stat_value

    # def parallelize(self, data, func, num_of_processes=8):
    #     data_split = np.array_split(data, num_of_processes)
    #     pool = mp.Pool(num_of_processes)
    #     data = pd.concat(pool.map(func, data_split))
    #     pool.close()
    #     pool.join()
    #     return data
    #
    # def run_on_subset(self, func, data_subset):
    #     return data_subset.apply(func, axis=1)
    #
    # def parallelize_on_rows(self, data, func, num_of_processes=8):
    #     return self.parallelize(data, partial(self.run_on_subset, func), num_of_processes)


def calculate_statistic(values, columns_names, stat_name, aggregation=False):
    """Calculate the statistic of values
        Args:
            values: a np.array of values we want to calculate the statistic on
                    This is actually a 2d array (matrix) of values.
            stat_name: the name of the statistic
            aggregation: if the aggregation on fields was performed
        Returns:
            a  calculated statistics
        Raises:
            an error
        """
    func_name = f'calculate_{stat_name}'
    num_parameters = len(signature(globals()[func_name]).parameters)
    if num_parameters == 2:
        stat = globals()[func_name](values, columns_names)
    else:
        stat = globals()[func_name](values, columns_names, aggregation)
    return stat


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of sum_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    SUM_STAT = SumStat(PARAMS)
    SUM_STAT.calculate_stats()
